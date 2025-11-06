import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb
import os
from typing import Dict, Tuple, Optional
import argparse

from models.l_module import LModule
from models.jepa_components import JEPALoss, TemporalReconstructionLoss
from configs.l_module_config import get_phase1_config
from utils.dataset import create_synthetic_dataset, AudioDataset


def compute_jepa_loss(model_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute JEPA loss from model output
    """
    embedding, prediction, target = model_output
    
    # Use MSE for prediction loss (simpler than the complex JEPA loss for initial testing)
    pred_loss = F.mse_loss(prediction, target)
    
    # Add temporal smoothness regularization
    if embedding.dim() > 2 and embedding.shape[1] > 1:  # batch, seq, features
        temporal_diff = torch.mean(torch.abs(embedding[:, 1:] - embedding[:, :-1]))
    else:
        temporal_diff = torch.tensor(0.0, device=embedding.device)
    
    # Add sparsity penalty
    sparsity_penalty = torch.mean(torch.abs(embedding))
    
    # Total loss
    total_loss = pred_loss + 0.1 * temporal_diff + 0.01 * sparsity_penalty
    
    metrics = {
        'total_loss': total_loss.item(),
        'pred_loss': pred_loss.item(),
        'temporal_loss': temporal_diff.item(),
        'sparsity_loss': sparsity_penalty.item()
    }
    
    return total_loss, metrics


def train_epoch(model: LModule, 
                dataloader: DataLoader, 
                optimizer: optim.Optimizer, 
                device: torch.device,
                epoch: int = 0) -> Dict[str, float]:
    """
    Train for one epoch using self-supervised learning
    """
    model.train()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_temporal_loss = 0.0
    total_sparsity_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} - Training")
    for batch_idx, (audio_batch,) in enumerate(pbar):
        audio_batch = audio_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass for self-supervised learning
        embedding, prediction, target = model.self_supervised_forward(audio_batch)
        
        # Compute loss
        batch_loss, metrics = compute_jepa_loss((embedding, prediction, target))
        
        batch_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += metrics['total_loss']
        total_pred_loss += metrics['pred_loss']
        total_temporal_loss += metrics['temporal_loss']
        total_sparsity_loss += metrics['sparsity_loss']
        
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss/num_batches:.4f}',
            'Pred': f'{total_pred_loss/num_batches:.4f}'
        })
        
        # Limit batches for testing
        if batch_idx >= 10:  # Just test a few batches for now
            break
    
    return {
        'total_loss': total_loss / num_batches,
        'pred_loss': total_pred_loss / num_batches,
        'temporal_loss': total_temporal_loss / num_batches,
        'sparsity_loss': total_sparsity_loss / num_batches
    }


def validate(model: LModule, 
             dataloader: DataLoader, 
             device: torch.device) -> Dict[str, float]:
    """
    Validate the model
    """
    model.eval()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_temporal_loss = 0.0
    total_sparsity_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (audio_batch,) in enumerate(dataloader):
            audio_batch = audio_batch.to(device)
            
            # Forward pass
            embedding, prediction, target = model.self_supervised_forward(audio_batch)
            
            # Compute loss
            batch_loss, metrics = compute_jepa_loss((embedding, prediction, target))
            
            total_loss += metrics['total_loss']
            total_pred_loss += metrics['pred_loss']
            total_temporal_loss += metrics['temporal_loss']
            total_sparsity_loss += metrics['sparsity_loss']
            
            num_batches += 1
            
            # Limit batches for testing
            if batch_idx >= 3:  # Just test a few batches for now
                break
    
    return {
        'total_loss': total_loss / num_batches,
        'pred_loss': total_pred_loss / num_batches,
        'temporal_loss': total_temporal_loss / num_batches,
        'sparsity_loss': total_sparsity_loss / num_batches
    }


def main():
    parser = argparse.ArgumentParser(description='Train L-Module with JEPA for Phase 1')
    parser.add_argument('--batch-size', type=int, default=4, help='Input batch size for training')  # Reduced for testing
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, help='Enable CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log-interval', type=int, default=1, help='How many batches to wait before logging')
    parser.add_argument('--save-model', action='store_true', default=True, help='Save the model after training')
    parser.add_argument('--model-dir', type=str, default='./models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device configuration
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    config = get_phase1_config()
    model = LModule(config).to(device)
    
    print(f"Model parameters: {model._parameter_count:,}")
    print(f"Target parameters: {config.target_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create synthetic dataset for testing
    print("Setting up dataset...")
    synthetic_data_path = "./data/synthetic_audio"
    os.makedirs(synthetic_data_path, exist_ok=True)
    
    # Create a small synthetic dataset for quick testing
    create_synthetic_dataset(synthetic_data_path, num_samples=50, sample_rate=16000)
    
    # Create dataset and dataloaders
    dataset = AudioDataset(synthetic_data_path, sample_rate=16000, audio_length=2.0)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    print(f"Dataset created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Training history
    history = {
        'train': {'total_loss': [], 'pred_loss': [], 'temporal_loss': [], 'sparsity_loss': []},
        'val': {'total_loss': [], 'pred_loss': [], 'temporal_loss': [], 'sparsity_loss': []}
    }
    
    print("Starting training...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validation
        val_metrics = validate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        for key, value in train_metrics.items():
            history['train'][key].append(value)
        for key, value in val_metrics.items():
            history['val'][key].append(value)
        
        print(f"Train Loss: {train_metrics['total_loss']:.4f}, "
              f"Val Loss: {val_metrics['total_loss']:.4f}")
        
    # Save final model
    if args.save_model:
        os.makedirs(args.model_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['total_loss'],
            'config': config,
            'history': history
        }, f"{args.model_dir}/l_module_jepa_final.pth")
        print(f"Final model saved with {model._parameter_count:,} parameters")
    
    print(f"Training completed! Final validation loss: {val_metrics['total_loss']:.4f}")
    
    # Final validation
    print("\nFinal model validation:")
    model.eval()
    with torch.no_grad():
        sample_audio = torch.randn(1, 32000, device=device)
        embedding, prediction, target = model.self_supervised_forward(sample_audio)
        
        print(f"Sample embedding shape: {embedding.shape}")
        print(f"Prediction shape: {prediction.shape}")
        print(f"Target shape: {target.shape}")
        
        # Check that prediction and target have similar values (model should learn to predict)
        pred_mse = F.mse_loss(prediction, target)
        print(f"Sample prediction MSE: {pred_mse:.6f}")
    
    return model, history


if __name__ == "__main__":
    model, history = main()