import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb
import os
import json
from typing import Dict, Tuple, Optional, List
import argparse
from datetime import datetime

from models.l_module import LModule
from configs.optimized_config import get_100m_optimized_config
from utils.dataset import create_synthetic_dataset, AudioDataset
from utils.validation_metrics import compute_all_validation_metrics, aggregate_metrics_across_batches


def compute_jepa_loss(model_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                      config) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute JEPA loss from model output with proper configuration
    """
    embedding, prediction, target = model_output
    
    # Prediction loss (MSE between prediction and target)
    pred_loss = F.mse_loss(prediction, target.detach())
    
    # Temporal smoothness regularization
    if embedding.dim() > 2 and embedding.shape[1] > 1:  # batch, seq, features or batch, features
        if embedding.dim() > 2:
            # For temporal embeddings, compute smoothness across time
            temporal_diff = torch.mean(torch.abs(embedding[:, 1:] - embedding[:, :-1]))
        else:
            temporal_diff = torch.tensor(0.0, device=embedding.device)
    else:
        temporal_diff = torch.tensor(0.0, device=embedding.device)
    
    # Sparsity penalty
    sparsity_penalty = torch.mean(torch.abs(embedding))
    
    # Total loss with configurable weights
    total_loss = (pred_loss + 
                  config.lambda_temporal * temporal_diff + 
                  config.lambda_energy * sparsity_penalty)
    
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
                config,
                device: torch.device,
                epoch: int = 0,
                compute_metrics: bool = True) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Train for one epoch with comprehensive metrics
    """
    model.train()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_temporal_loss = 0.0
    total_sparsity_loss = 0.0
    num_batches = 0
    
    # Store metrics for aggregation
    batch_metrics_list = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} - Training")
    for batch_idx, (audio_batch,) in enumerate(pbar):
        audio_batch = audio_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass for self-supervised learning
        embedding, prediction, target = model.self_supervised_forward(audio_batch)
        
        # Compute loss
        batch_loss, loss_metrics = compute_jepa_loss((embedding, prediction, target), config)
        
        batch_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss_metrics['total_loss']
        total_pred_loss += loss_metrics['pred_loss']
        total_temporal_loss += loss_metrics['temporal_loss']
        total_sparsity_loss += loss_metrics['sparsity_loss']
        
        num_batches += 1
        
        # Compute comprehensive validation metrics
        if compute_metrics and batch_idx % 5 == 0:  # Compute metrics every 5 batches to save time
            try:
                # Get spikes from model if available (would need to modify model to return spikes)
                spikes = None  # Placeholder - would come from SNN components
                batch_out = (embedding, prediction, target)
                val_metrics = compute_all_validation_metrics(batch_out, audio_batch, spikes)
                batch_metrics_list.append(val_metrics)
            except:
                # If validation metrics fail, just continue
                pass
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss/num_batches:.4f}',
            'Pred': f'{total_pred_loss/num_batches:.4f}'
        })
    
    avg_metrics = {
        'total_loss': total_loss / num_batches,
        'pred_loss': total_pred_loss / num_batches,
        'temporal_loss': total_temporal_loss / num_batches,
        'sparsity_loss': total_sparsity_loss / num_batches
    }
    
    return avg_metrics, batch_metrics_list


def validate(model: LModule, 
             dataloader: DataLoader, 
             config,
             device: torch.device,
             compute_metrics: bool = True) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Validate the model with comprehensive metrics
    """
    model.eval()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_temporal_loss = 0.0
    total_sparsity_loss = 0.0
    num_batches = 0
    
    # Store metrics for aggregation
    batch_metrics_list = []
    
    with torch.no_grad():
        for batch_idx, (audio_batch,) in enumerate(dataloader):
            audio_batch = audio_batch.to(device)
            
            # Forward pass
            embedding, prediction, target = model.self_supervised_forward(audio_batch)
            
            # Compute loss
            batch_loss, loss_metrics = compute_jepa_loss((embedding, prediction, target), config)
            
            total_loss += loss_metrics['total_loss']
            total_pred_loss += loss_metrics['pred_loss']
            total_temporal_loss += loss_metrics['temporal_loss']
            total_sparsity_loss += loss_metrics['sparsity_loss']
            
            num_batches += 1
            
            # Compute comprehensive validation metrics
            if compute_metrics and batch_idx % 2 == 0:  # Every 2 batches for validation
                try:
                    spikes = None  # Placeholder
                    batch_out = (embedding, prediction, target)
                    val_metrics = compute_all_validation_metrics(batch_out, audio_batch, spikes)
                    batch_metrics_list.append(val_metrics)
                except:
                    # If validation metrics fail, just continue
                    pass
    
    avg_metrics = {
        'total_loss': total_loss / num_batches,
        'pred_loss': total_pred_loss / num_batches,
        'temporal_loss': total_temporal_loss / num_batches,
        'sparsity_loss': total_sparsity_loss / num_batches
    }
    
    # Aggregate validation metrics
    aggregated_metrics = {}
    if batch_metrics_list:
        aggregated_metrics = aggregate_metrics_across_batches(batch_metrics_list)
    
    return avg_metrics, aggregated_metrics


def save_training_checkpoint(model: LModule, 
                           optimizer: optim.Optimizer, 
                           epoch: int, 
                           train_loss: float, 
                           val_loss: float,
                           config,
                           save_path: str):
    """
    Save a training checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config.__dict__ if hasattr(config, '__dict__') else str(config),
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)


def main():
    parser = argparse.ArgumentParser(description='Train 100M Parameter L-Module with JEPA')
    parser.add_argument('--batch-size', type=int, default=4, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, help='Enable CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model-dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--data-dir', type=str, default='./data/synthetic_audio', help='Directory for audio data')
    parser.add_argument('--save-freq', type=int, default=10, help='Save model every N epochs')
    parser.add_argument('--wandb', action='store_true', default=False, help='Use Weights & Biases logging')
    parser.add_argument('--project-name', type=str, default='ember_l_module_100m', help='Wandb project name')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device configuration
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create model with 100M optimized config
    config = get_100m_optimized_config()
    model = LModule(config).to(device)
    
    print(f"Model parameters: {model._parameter_count:,}")
    print(f"Target parameters: {config.target_params:,}")
    print(f"Embedding dimension: {config.embed_dim}")
    print(f"SNN hidden size: {config.snn_hidden_size}")
    print(f"JEPA hidden size: {config.jepa_hidden_size}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create synthetic dataset for training
    print(f"Setting up dataset in {args.data_dir}...")
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Create a larger synthetic dataset
    print("Creating synthetic audio dataset...")
    create_synthetic_dataset(args.data_dir, num_samples=200, sample_rate=16000)
    
    # Create dataset and dataloaders
    dataset = AudioDataset(args.data_dir, sample_rate=16000, audio_length=2.0)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"Dataset created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Training history
    history = {
        'train': {'total_loss': [], 'pred_loss': [], 'temporal_loss': [], 'sparsity_loss': []},
        'val': {'total_loss': [], 'pred_loss': [], 'temporal_loss': [], 'sparsity_loss': []},
        'val_metrics': []  # Comprehensive validation metrics
    }
    
    # Initialize wandb if requested
    if args.wandb:
        try:
            wandb.init(
                project=args.project_name,
                config={
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "epochs": args.epochs,
                    "model_params": model._parameter_count,
                    "embed_dim": config.embed_dim,
                    "snn_hidden_size": config.snn_hidden_size,
                    "jepa_hidden_size": config.jepa_hidden_size
                }
            )
        except Exception as e:
            print(f"Could not initialize wandb: {e}")
            args.wandb = False
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print("Starting training with 100M parameter model...")
    print(f"Model has {model._parameter_count:,} parameters ({model._parameter_count/config.target_params:.1%} of target)")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training
        train_metrics, train_val_metrics = train_epoch(
            model, train_loader, optimizer, config, device, epoch, compute_metrics=True
        )
        
        # Validation
        val_metrics, aggregated_val_metrics = validate(
            model, val_loader, config, device, compute_metrics=True
        )
        
        # Update learning rate
        scheduler.step()
        
        # Store metrics
        for key, value in train_metrics.items():
            history['train'][key].append(value)
        for key, value in val_metrics.items():
            history['val'][key].append(value)
        
        if aggregated_val_metrics:
            history['val_metrics'].append(aggregated_val_metrics)
        
        print(f"Train Loss: {train_metrics['total_loss']:.6f}, "
              f"Val Loss: {val_metrics['total_loss']:.6f}")
        
        # Log to wandb
        if args.wandb:
            log_dict = {
                'epoch': epoch,
                'train_total_loss': train_metrics['total_loss'],
                'val_total_loss': val_metrics['total_loss'],
                'train_pred_loss': train_metrics['pred_loss'],
                'val_pred_loss': val_metrics['pred_loss'],
                'lr': optimizer.param_groups[0]['lr']
            }
            # Add some key validation metrics if available
            if aggregated_val_metrics:
                for key in ['prediction_mse_mean', 'prediction_cosine_similarity_mean', 
                           'embedding_mean_mean', 'embedding_std_mean']:
                    if key in aggregated_val_metrics:
                        log_dict[key.replace('_mean', '')] = aggregated_val_metrics[key]
            
            wandb.log(log_dict)
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            patience_counter = 0
            
            # Save best model
            os.makedirs(args.model_dir, exist_ok=True)
            save_path = f"{args.model_dir}/l_module_100m_best.pth"
            save_training_checkpoint(
                model, optimizer, epoch, train_metrics['total_loss'], 
                val_metrics['total_loss'], config, save_path
            )
            print(f"New best model saved with validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save model periodically
        if (epoch + 1) % args.save_freq == 0:
            save_path = f"{args.model_dir}/l_module_100m_epoch_{epoch+1}.pth"
            save_training_checkpoint(
                model, optimizer, epoch, train_metrics['total_loss'], 
                val_metrics['total_loss'], config, save_path
            )
    
    # Save final model
    final_save_path = f"{args.model_dir}/l_module_100m_final.pth"
    save_training_checkpoint(
        model, optimizer, args.epochs-1, 
        train_metrics['total_loss'], val_metrics['total_loss'], 
        config, final_save_path
    )
    
    # Save training history
    history_path = f"{args.model_dir}/training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'config': config.__dict__,
            'final_params': model._parameter_count,
            'history': history
        }, f, indent=2)
    
    if args.wandb:
        wandb.finish()
    
    print(f"\nTraining completed!")
    print(f"Final validation loss: {val_metrics['total_loss']:.6f}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final model saved to: {final_save_path}")
    print(f"Training history saved to: {history_path}")
    
    # Final validation report
    print("\nFinal model validation report:")
    model.eval()
    with torch.no_grad():
        sample_audio = torch.randn(1, 32000, device=device)
        embedding, prediction, target = model.self_supervised_forward(sample_audio)
        
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Prediction shape: {prediction.shape}")
        print(f"  Target shape: {target.shape}")
        
        pred_mse = F.mse_loss(prediction, target)
        pred_cos_sim = F.cosine_similarity(prediction, target, dim=-1).mean()
        print(f"  Prediction MSE: {pred_mse:.6f}")
        print(f"  Prediction Cosine Similarity: {pred_cos_sim:.4f}")
        print(f"  Embedding mean: {embedding.mean():.4f}, std: {embedding.std():.4f}")
    
    return model, history


if __name__ == "__main__":
    model, history = main()