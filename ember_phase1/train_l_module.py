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

from models.l_module import LModule, TemporalCoherenceLoss
from configs.l_module_config import get_phase1_config
from utils.dataset import AudioDataset  # We'll create this next


def train_epoch(model: LModule, 
                dataloader: DataLoader, 
                optimizer: optim.Optimizer, 
                criterion: nn.Module,
                device: torch.device,
                epoch: int = 0) -> Dict[str, float]:
    """
    Train for one epoch using self-supervised learning
    """
    model.train()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_temporal_loss = 0.0
    total_energy_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} - Training")
    for batch_idx, (audio_batch,) in enumerate(pbar):
        audio_batch = audio_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass for self-supervised learning
        embedding, prediction, target = model.self_supervised_forward(audio_batch)
        
        # Calculate losses
        pred_loss = F.mse_loss(prediction, target)
        
        # Temporal coherence and energy losses
        temporal_loss = torch.mean(torch.abs(embedding[:, 1:] - embedding[:, :-1])) if embedding.shape[1] > 1 else torch.tensor(0.0, device=device)
        energy_loss = torch.mean(torch.abs(embedding))
        
        # Total loss
        total_batch_loss = pred_loss + model.config.lambda_temporal * temporal_loss + model.config.lambda_energy * energy_loss
        
        total_batch_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += total_batch_loss.item()
        total_pred_loss += pred_loss.item()
        total_temporal_loss += temporal_loss.item() if isinstance(temporal_loss, torch.Tensor) else temporal_loss
        total_energy_loss += energy_loss.item()
        
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss/num_batches:.4f}',
            'Pred': f'{total_pred_loss/num_batches:.4f}',
            'Temporal': f'{total_temporal_loss/num_batches:.4f}',
            'Energy': f'{total_energy_loss/num_batches:.4f}'
        })
    
    return {
        'total_loss': total_loss / num_batches,
        'pred_loss': total_pred_loss / num_batches,
        'temporal_loss': total_temporal_loss / num_batches,
        'energy_loss': total_energy_loss / num_batches
    }


def validate(model: LModule, 
             dataloader: DataLoader, 
             criterion: nn.Module,
             device: torch.device) -> Dict[str, float]:
    """
    Validate the model
    """
    model.eval()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_temporal_loss = 0.0
    total_energy_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for audio_batch, in dataloader:
            audio_batch = audio_batch.to(device)
            
            # Forward pass
            embedding, prediction, target = model.self_supervised_forward(audio_batch)
            
            # Calculate losses
            pred_loss = F.mse_loss(prediction, target)
            temporal_loss = torch.mean(torch.abs(embedding[:, 1:] - embedding[:, :-1])) if embedding.shape[1] > 1 else torch.tensor(0.0, device=device)
            energy_loss = torch.mean(torch.abs(embedding))
            
            # Total loss
            total_batch_loss = pred_loss + model.config.lambda_temporal * temporal_loss + model.config.lambda_energy * energy_loss
            
            total_loss += total_batch_loss.item()
            total_pred_loss += pred_loss.item()
            total_temporal_loss += temporal_loss.item() if isinstance(temporal_loss, torch.Tensor) else temporal_loss
            total_energy_loss += energy_loss.item()
            
            num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'pred_loss': total_pred_loss / num_batches,
        'temporal_loss': total_temporal_loss / num_batches,
        'energy_loss': total_energy_loss / num_batches
    }


def main():
    parser = argparse.ArgumentParser(description='Train L-Module for Phase 1')
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, help='Enable CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log-interval', type=int, default=10, help='How many batches to wait before logging')
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
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss function
    criterion = TemporalCoherenceLoss(
        lambda_temporal=model.config.lambda_temporal,
        lambda_energy=model.config.lambda_energy
    )
    
    # Create synthetic dataset for testing
    # In a real implementation, you'd load actual audio data
    from utils.dataset import create_synthetic_dataset
    print("Creating synthetic dataset for testing...")
    synthetic_data_path = "./data/synthetic_audio"
    os.makedirs(synthetic_data_path, exist_ok=True)
    
    # Create synthetic data - in real training, replace with actual audio dataset
    # For now, let's create a simple dummy dataset
    class DummyDataset:
        def __init__(self, size=100, audio_length=32000):
            self.size = size
            self.audio_length = audio_length
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Generate random audio-like tensor
            audio = torch.randn(self.audio_length) * 0.1  # Small amplitude white noise
            return audio
    
    train_dataset = DummyDataset(size=200)  # Smaller for testing
    val_dataset = DummyDataset(size=50)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"Dataset created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Training history
    history = {
        'train': {'total_loss': [], 'pred_loss': [], 'temporal_loss': [], 'energy_loss': []},
        'val': {'total_loss': [], 'pred_loss': [], 'temporal_loss': [], 'energy_loss': []}
    }
    
    # Initialize wandb if available
    try:
        wandb.init(
            project="ember_phase1_l_module",
            config={
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "model_params": model._parameter_count,
                "embed_dim": config.embed_dim
            }
        )
    except:
        print("Wandb not available, proceeding without logging")
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print("Starting training...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validation
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        for key, value in train_metrics.items():
            history['train'][key].append(value)
        for key, value in val_metrics.items():
            history['val'][key].append(value)
        
        print(f"Train Loss: {train_metrics['total_loss']:.4f}, "
              f"Val Loss: {val_metrics['total_loss']:.4f}")
        
        # Log to wandb
        try:
            wandb.log({
                'epoch': epoch,
                'train_total_loss': train_metrics['total_loss'],
                'val_total_loss': val_metrics['total_loss'],
                'train_pred_loss': train_metrics['pred_loss'],
                'val_pred_loss': val_metrics['pred_loss'],
                'lr': optimizer.param_groups[0]['lr']
            })
        except:
            pass
        
        # Early stopping
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            patience_counter = 0
            
            if args.save_model:
                os.makedirs(args.model_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'config': config
                }, f"{args.model_dir}/l_module_best.pth")
                print(f"Best model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save final model
    if args.save_model:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['total_loss'],
            'config': config
        }, f"{args.model_dir}/l_module_final.pth")
        print("Final model saved")
    
    try:
        wandb.finish()
    except:
        pass
    
    print(f"Training completed! Final validation loss: {val_metrics['total_loss']:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return model, history


if __name__ == "__main__":
    model, history = main()