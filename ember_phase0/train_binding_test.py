import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
import os
from datetime import datetime
from typing import Dict, Tuple, List, Optional

from models.l_mini import LMini
from models.h_mini import HMini
from models.control_condition import SemanticBottleneckBaseline
from utils.dataset import create_dataloaders, create_text_dataloaders
from config import DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE


class BindingTestTrainer:
    """
    Trainer for the Phase 0 binding test: L-mini + H-mini vs control condition
    """
    def __init__(self, 
                 l_mini: LMini,
                 h_mini: HMini, 
                 control_model: SemanticBottleneckBaseline,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 control_train_loader: DataLoader,
                 control_val_loader: DataLoader,
                 control_test_loader: DataLoader,
                 learning_rate: float = 1e-4,
                 device: torch.device = DEVICE):
        
        self.l_mini = l_mini.to(device)
        self.h_mini = h_mini.to(device)
        self.control_model = control_model.to(device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.control_train_loader = control_train_loader
        self.control_val_loader = control_val_loader
        self.control_test_loader = control_test_loader
        
        self.device = device
        self.learning_rate = learning_rate
        
        # Optimizers
        self.full_model_params = list(l_mini.parameters()) + list(h_mini.parameters())
        self.full_optimizer = optim.AdamW(self.full_model_params, lr=learning_rate, weight_decay=1e-4)
        self.control_optimizer = optim.AdamW(control_model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'full_model': {'train_loss': [], 'val_loss': [], 'val_acc': []},
            'control_model': {'train_loss': [], 'val_loss': [], 'val_acc': []}
        }
        
        # Early stopping
        self.early_stopping_count = 0
        self.best_val_acc = 0.0
        self.early_stopping_patience = EARLY_STOPPING_PATIENCE
        
    def train_epoch(self, model_type: str = 'full', use_jepa: bool = True, jepa_weight: float = 0.1) -> float:
        """
        Train for one epoch
        
        Args:
            model_type: 'full' for L-mini + H-mini, 'control' for text-only baseline
            use_jepa: Whether to use JEPA-style temporal prediction loss for L-mini
            jepa_weight: Weight for JEPA loss component
            
        Returns:
            Average loss for the epoch
        """
        if model_type == 'full':
            model = nn.Sequential(self.l_mini, self.h_mini)
            dataloader = self.train_loader
            optimizer = self.full_optimizer
            model.train()
        else:
            model = self.control_model
            dataloader = self.control_train_loader
            optimizer = self.control_optimizer
            model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_audio, batch_labels in tqdm(dataloader, desc=f"Training {model_type}", leave=False):
            batch_audio = batch_audio.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            optimizer.zero_grad()
            
            if model_type == 'full':
                # Forward through L-mini then H-mini
                if use_jepa and hasattr(self.l_mini, 'compute_jepa_loss'):
                    # Compute both supervised and JEPA losses
                    l_mini_out = self.l_mini(batch_audio, use_jepa=True)  # Use JEPA-enabled forward pass
                    logits = self.h_mini(sensory_embed=l_mini_out, 
                                        text_embed=torch.zeros(batch_audio.size(0), 128, device=self.device),
                                        use_sensory=True)
                    
                    # Supervised loss
                    supervised_loss = self.criterion(logits, batch_labels)
                    
                    # JEPA loss for temporal prediction learning
                    jepa_loss, jepa_metrics = self.l_mini.compute_jepa_loss(batch_audio)
                    
                    # Combined loss
                    loss = supervised_loss + jepa_weight * jepa_loss
                else:
                    # Standard training without JEPA
                    l_mini_out = self.l_mini(batch_audio, use_jepa=False)  # Use SNN-only path
                    logits = self.h_mini(sensory_embed=l_mini_out, 
                                        text_embed=torch.zeros(batch_audio.size(0), 128, device=self.device),
                                        use_sensory=True)
                    
                    loss = self.criterion(logits, batch_labels)
            else:
                # Forward through control model
                logits = self.control_model(batch_audio)
                loss = self.criterion(logits, batch_labels)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self, model_type: str = 'full', val_set: bool = True) -> Tuple[float, float]:
        """
        Evaluate model on validation or test set
        
        Args:
            model_type: 'full' or 'control'
            val_set: True for validation, False for test
            
        Returns:
            (loss, accuracy)
        """
        if model_type == 'full':
            model = nn.Sequential(self.l_mini, self.h_mini)
            dataloader = self.val_loader if val_set else self.test_loader
            model.eval()
        else:
            model = self.control_model
            dataloader = self.control_val_loader if val_set else self.control_test_loader
            model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch_audio, batch_labels in dataloader:
                batch_audio = batch_audio.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                if model_type == 'full':
                    l_mini_out = self.l_mini(batch_audio, use_jepa=False)  # Use only SNN during evaluation
                    logits = self.h_mini(sensory_embed=l_mini_out,
                                        text_embed=torch.zeros(batch_audio.size(0), 128, device=self.device),
                                        use_sensory=True)
                else:
                    logits = self.control_model(batch_audio)
                
                loss = self.criterion(logits, batch_labels)
                total_loss += loss.item()
                
                # Calculate predictions
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def train(self) -> Dict:
        """
        Train both models and compare performance
        """
        print("Starting Phase 0 Binding Test Training...")
        print(f"Device: {self.device}")
        print(f"Full model parameters: {(sum(p.numel() for p in self.full_model_params)):,}")
        print(f"Control model parameters: {(sum(p.numel() for p in self.control_model.parameters())):,}")
        
        # Initialize wandb if available
        try:
            wandb.init(
                project="ember_phase0_binding_test",
                config={
                    "learning_rate": self.learning_rate,
                    "batch_size": BATCH_SIZE,
                    "num_epochs": NUM_EPOCHS
                }
            )
        except:
            print("Wandb not available, proceeding without logging")
        
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            
            # Train both models
            full_train_loss = self.train_epoch(model_type='full', use_jepa=True, jepa_weight=0.1)
            control_train_loss = self.train_epoch(model_type='control', use_jepa=False)  # Control doesn't use JEPA
            
            # Evaluate both models
            full_val_loss, full_val_acc = self.evaluate(model_type='full', val_set=True)
            control_val_loss, control_val_acc = self.evaluate(model_type='control', val_set=True)
            
            # Log to history
            self.history['full_model']['train_loss'].append(full_train_loss)
            self.history['full_model']['val_loss'].append(full_val_loss)
            self.history['full_model']['val_acc'].append(full_val_acc)
            
            self.history['control_model']['train_loss'].append(control_train_loss)
            self.history['control_model']['val_loss'].append(control_val_loss)
            self.history['control_model']['val_acc'].append(control_val_acc)
            
            # Print results
            print(f"Full Model  - Train Loss: {full_train_loss:.4f}, Val Loss: {full_val_loss:.4f}, Val Acc: {full_val_acc:.4f}")
            print(f"Control     - Train Loss: {control_train_loss:.4f}, Val Loss: {control_val_loss:.4f}, Val Acc: {control_val_acc:.4f}")
            
            # Log to wandb
            try:
                wandb.log({
                    'epoch': epoch,
                    'full_train_loss': full_train_loss,
                    'full_val_loss': full_val_loss,
                    'full_val_acc': full_val_acc,
                    'control_train_loss': control_train_loss,
                    'control_val_loss': control_val_loss,
                    'control_val_acc': control_val_acc
                })
            except:
                pass  # Wandb not available
            
            # Early stopping based on full model performance
            if full_val_acc > self.best_val_acc:
                self.best_val_acc = full_val_acc
                self.early_stopping_count = 0
                
                # Save best model
                self.save_checkpoint(f"best_model_epoch_{epoch+1}.pth")
            else:
                self.early_stopping_count += 1
                if self.early_stopping_count >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        try:
            wandb.finish()
        except:
            pass
        
        return self.history
    
    def evaluate_final_models(self) -> Dict[str, float]:
        """
        Evaluate both models on the test set and return comparison
        """
        print("\nEvaluating final models on test set...")
        
        # Evaluate full model (L-mini + H-mini)
        full_test_loss, full_test_acc = self.evaluate(model_type='full', val_set=False)
        
        # Evaluate control model
        control_test_loss, control_test_acc = self.evaluate(model_type='control', val_set=False)
        
        # Calculate performance gap
        performance_gap = full_test_acc - control_test_acc
        
        results = {
            'full_model_test_acc': full_test_acc,
            'control_model_test_acc': control_test_acc,
            'performance_gap': performance_gap,
            'binding_successful': performance_gap > 0.05  # Significant improvement
        }
        
        print(f"\nFinal Test Results:")
        print(f"Full Model (L-mini + H-mini): {full_test_acc:.4f}")
        print(f"Control Model (Text-only): {control_test_acc:.4f}")
        print(f"Performance Gap: {performance_gap:.4f}")
        print(f"Binding Successful: {results['binding_successful']}")
        
        return results
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'l_mini_state_dict': self.l_mini.state_dict(),
            'h_mini_state_dict': self.h_mini.state_dict(),
            'control_model_state_dict': self.control_model.state_dict(),
            'full_optimizer_state_dict': self.full_optimizer.state_dict(),
            'control_optimizer_state_dict': self.control_optimizer.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.l_mini.load_state_dict(checkpoint['l_mini_state_dict'])
        self.h_mini.load_state_dict(checkpoint['h_mini_state_dict'])
        self.control_model.load_state_dict(checkpoint['control_model_state_dict'])
        self.full_optimizer.load_state_dict(checkpoint['full_optimizer_state_dict'])
        self.control_optimizer.load_state_dict(checkpoint['control_optimizer_state_dict'])
        self.history = checkpoint['history']


def main():
    """
    Main execution function for the binding test
    """
    print("=== PROJECT EMBER: Phase 0 Binding Test ===")
    print(f"Using device: {DEVICE}")
    
    # Initialize models
    print("\n1. Initializing models...")
    l_mini = LMini(embed_dim=512)
    h_mini = HMini(sensory_dim=512, text_dim=128, embed_dim=512, num_classes=2)
    control_model = SemanticBottleneckBaseline(embed_dim=128, num_classes=2)
    
    print(f"L-mini parameters: {sum(p.numel() for p in l_mini.parameters()):,}")
    print(f"H-mini parameters: {sum(p.numel() for p in h_mini.parameters()):,}")
    print(f"Control model parameters: {sum(p.numel() for p in control_model.parameters()):,}")
    
    # Create datasets and dataloaders
    print("\n2. Creating datasets...")
    # Create synthetic data directory if it doesn't exist
    os.makedirs("ember_phase0/data", exist_ok=True)
    
    # Create dataloaders for both full model and control
    dataloaders = create_dataloaders("ember_phase0/data", batch_size=BATCH_SIZE)
    control_dataloaders = create_text_dataloaders("ember_phase0/data", batch_size=BATCH_SIZE)
    
    # Initialize trainer
    print("\n3. Initializing trainer...")
    trainer = BindingTestTrainer(
        l_mini=l_mini,
        h_mini=h_mini,
        control_model=control_model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        test_loader=dataloaders['test'],
        control_train_loader=control_dataloaders['train'],
        control_val_loader=control_dataloaders['val'],
        control_test_loader=control_dataloaders['test'],
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )
    
    # Train models
    print("\n4. Starting training...")
    history = trainer.train()
    
    # Evaluate final models
    print("\n5. Evaluating final models...")
    results = trainer.evaluate_final_models()
    
    # Summary
    print("\n=== PHASE 0 BINDING TEST RESULTS ===")
    print(f"Full Model Test Accuracy: {results['full_model_test_acc']:.4f}")
    print(f"Control Model Test Accuracy: {results['control_model_test_acc']:.4f}")
    print(f"Performance Gap: {results['performance_gap']:.4f}")
    
    if results['binding_successful']:
        print("\n✅ BINDING TEST SUCCESSFUL!")
        print("The H-module can effectively use subsymbolic sensory embeddings from L-module.")
        print("This validates the core architecture concept. Proceed to Phase 1.")
    else:
        print("\n❌ BINDING TEST FAILED!")
        print("The H-module cannot effectively use subsymbolic sensory embeddings.")
        print("Consider alternative approaches before continuing with full architecture.")
    
    print(f"\nPhase 0 complete. Results: {results}")
    return results


if __name__ == "__main__":
    results = main()