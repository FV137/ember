import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict


class TemporalMasking(nn.Module):
    """
    Implements temporal masking for self-supervised learning
    Similar to Masked Language Modeling but for temporal audio sequences
    """
    def __init__(self, 
                 mask_ratio: float = 0.15,
                 mask_type: str = 'random',  # 'random', 'block', 'temporal'
                 block_size: int = 10):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.block_size = block_size
        
    def forward(self, 
                x: torch.Tensor, 
                mask_ratio: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply temporal masking to input sequence
        
        Args:
            x: (batch, seq_len, features) - input sequence
            mask_ratio: Optional override for mask ratio
            
        Returns:
            visible: (batch, num_visible, features) - unmasked tokens
            masked: (batch, num_masked, features) - masked tokens  
            mask: (batch, seq_len) - boolean mask (True = masked)
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        batch_size, seq_len, features = x.shape
        
        # Determine number of positions to mask
        num_mask = int(seq_len * mask_ratio)
        
        # Generate mask indices based on type
        if self.mask_type == 'random':
            # Random masking - standard approach
            rand = torch.rand(batch_size, seq_len, device=x.device)
            mask_indices = rand.topk(num_mask, dim=1, largest=False).indices
        elif self.mask_type == 'block':
            # Block masking - mask consecutive blocks
            mask_indices = []
            for b in range(batch_size):
                # Random starting positions for blocks
                block_starts = torch.randperm(seq_len - self.block_size, device=x.device)[:num_mask//self.block_size + 1]
                block_indices = []
                for start in block_starts:
                    end = min(start + self.block_size, seq_len)
                    block_indices.extend(list(range(start.item(), end)))
                
                # Trim to exact number
                if len(block_indices) > num_mask:
                    block_indices = block_indices[:num_mask]
                
                mask_indices.append(torch.tensor(block_indices[:num_mask], device=x.device, dtype=torch.long))
            
            # Pad if needed
            max_len = max(len(m) for m in mask_indices)
            mask_tensor = torch.zeros(batch_size, max_len, dtype=torch.long, device=x.device)
            for i, indices in enumerate(mask_indices):
                mask_tensor[i, :len(indices)] = indices
            mask_indices = mask_tensor[:, :num_mask]
        else:  # 'temporal' - mask later time steps
            # Mask temporal positions from the end
            mask_indices = torch.arange(seq_len - num_mask, seq_len, device=x.device).repeat(batch_size, 1)
        
        # Create mask
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
        for b in range(batch_size):
            mask[b, mask_indices[b]] = True
        
        # Apply masking
        visible_idx = ~mask
        visible_tokens = x[visible_idx].view(batch_size, -1, features)
        masked_tokens = x[mask].view(batch_size, -1, features)
        
        return visible_tokens, masked_tokens, mask


class MaskedPredictionHead(nn.Module):
    """
    Prediction head for masked temporal prediction
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalReconstructionLoss(nn.Module):
    """
    Loss for temporal reconstruction tasks
    """
    def __init__(self, 
                 lambda_pred: float = 1.0,
                 lambda_temporal: float = 0.1,
                 lambda_energy: float = 0.01,
                 use_cosine_similarity: bool = True):
        super().__init__()
        self.lambda_pred = lambda_pred
        self.lambda_temporal = lambda_temporal
        self.lambda_energy = lambda_energy
        self.use_cosine_similarity = use_cosine_similarity
    
    def forward(self, 
                predictions: torch.Tensor,
                targets: torch.Tensor,
                embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute reconstruction loss
        
        Args:
            predictions: (batch, seq_len, features) - predicted tokens
            targets: (batch, seq_len, features) - target tokens
            embeddings: (batch, seq_len, features) - original embeddings
            
        Returns:
            loss: scalar - total loss
            metrics: dict - individual loss components
        """
        # Prediction loss (MSE or Cosine similarity)
        if self.use_cosine_similarity:
            pred_loss = 1 - F.cosine_similarity(predictions, targets, dim=-1).mean()
        else:
            pred_loss = F.mse_loss(predictions, targets)
        
        # Temporal smoothness loss - encourage gradual changes
        if embeddings.shape[1] > 1:  # seq_len > 1
            temporal_diff = embeddings[:, 1:] - embeddings[:, :-1]
            temporal_loss = torch.mean(torch.norm(temporal_diff, p=2, dim=-1))
        else:
            temporal_loss = torch.tensor(0.0, device=embeddings.device)
        
        # Energy efficiency (sparsity) loss
        energy_loss = torch.mean(torch.abs(embeddings))
        
        # Total loss
        total_loss = (self.lambda_pred * pred_loss + 
                     self.lambda_temporal * temporal_loss + 
                     self.lambda_energy * energy_loss)
        
        metrics = {
            'pred_loss': pred_loss,
            'temporal_loss': temporal_loss,
            'energy_loss': energy_loss
        }
        
        return total_loss, metrics


class PredictiveCodingModule(nn.Module):
    """
    JEPA-style predictive coding module
    Predicts future representations from current representations
    """
    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int,
                 prediction_steps: int = 3,  # Predict 3 time steps ahead
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.prediction_steps = prediction_steps
        
        # Prediction network
        layers = []
        layers.append(nn.Linear(embed_dim, hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, embed_dim * prediction_steps))
        
        self.predictor = nn.Sequential(*layers)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, 
                x: torch.Tensor, 
                return_intermediates: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict future embeddings from current embeddings
        
        Args:
            x: (batch, seq_len, embed_dim) - current embeddings
            return_intermediates: Whether to return intermediate predictions
            
        Returns:
            predictions: (batch, seq_len-prediction_steps, embed_dim*prediction_steps) - future predictions
            intermediates: Optional - intermediate states if return_intermediates=True
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Limit prediction to avoid going past sequence length
        effective_len = max(1, seq_len - self.prediction_steps)
        
        # Process through predictor
        x_norm = self.norm(x[:, :effective_len, :])  # Normalize current embeddings
        predictions = self.predictor(x_norm)  # (batch, effective_len, embed_dim*prediction_steps)
        
        # Reshape predictions to separate time steps
        predictions = predictions.view(batch_size, effective_len, self.prediction_steps, embed_dim)
        
        if return_intermediates:
            return predictions, x_norm
        return predictions, None


class CrossTemporalAttention(nn.Module):
    """
    Cross-attention between different temporal positions for self-supervised learning
    """
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with cross-temporal attention
        
        Args:
            x: (batch, seq_len, embed_dim) - input sequence
            mask: (batch, seq_len) - boolean mask for masked positions
            
        Returns:
            output: (batch, seq_len, embed_dim) - attended output
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for attention heads
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N)
            attn.masked_fill_(mask_expanded == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.proj_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class JEPALoss(nn.Module):
    """
    Contrastive loss for JEPA-style learning
    Unlike traditional contrastive learning, JEPA learns representations that are 
    predictive of future states without requiring negative samples
    """
    def __init__(self, 
                 alpha: float = 1.0,  # Weight for prediction loss
                 beta: float = 1.0,   # Weight for regularization
                 temperature: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                representations: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute JEPA loss
        
        Args:
            predictions: (batch, seq_len, embed_dim) - predicted representations
            targets: (batch, seq_len, embed_dim) - target representations  
            representations: (batch, seq_len, embed_dim) - original representations
            
        Returns:
            loss: scalar - JEPA loss
            metrics: dict - individual components
        """
        # Prediction loss (MSE between predictions and targets)
        pred_loss = F.mse_loss(predictions, targets.detach())
        
        # Representation smoothness (regularization)
        rep_smoothness = torch.mean(torch.norm(representations[:, 1:] - representations[:, :-1], p=2, dim=-1))
        
        # Total loss
        total_loss = self.alpha * pred_loss + self.beta * rep_smoothness
        
        metrics = {
            'pred_loss': pred_loss,
            'smoothness_loss': rep_smoothness
        }
        
        return total_loss, metrics


# Test the JEPA components
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Testing JEPA components...")
    
    # Test Temporal Masking
    masking = TemporalMasking(mask_ratio=0.2, mask_type='random')
    x = torch.randn(2, 100, 512, device=device)  # batch, seq_len, features
    visible, masked, mask = masking(x)
    print(f"Temporal masking - Input: {x.shape}, Visible: {visible.shape}, Masked: {masked.shape}, Mask: {mask.shape}")
    print(f"  Mask ratio: {mask.float().mean():.2f}")
    
    # Test Masked Prediction Head
    pred_head = MaskedPredictionHead(512, 1024, 512).to(device)
    pred_output = pred_head(visible)
    print(f"Prediction head - Input: {visible.shape}, Output: {pred_output.shape}")
    
    # Test Predictive Coding Module
    pred_coder = PredictiveCodingModule(512, 1024, prediction_steps=3).to(device)
    pred_seq, _ = pred_coder(x)
    print(f"Predictive coding - Input: {x.shape}, Output: {pred_seq.shape}")
    
    # Test Cross-temporal Attention
    attn = CrossTemporalAttention(512, num_heads=8).to(device)
    attended = attn(x, mask=None)
    print(f"Cross-temporal attention - Input: {x.shape}, Output: {attended.shape}")
    
    # Test losses
    recon_loss = TemporalReconstructionLoss()
    targets = torch.randn_like(x)
    total_loss, metrics = recon_loss(pred_output, targets[:, :pred_output.shape[1]], x)
    print(f"Reconstruction loss: {total_loss:.4f}")
    
    jepa_loss = JEPALoss()
    jepa_total, jepa_metrics = jepa_loss(pred_seq.mean(2), x[:, :pred_seq.shape[1]], x[:, :pred_seq.shape[1]])
    print(f"JEPA loss: {jepa_total:.4f}")
    
    print("All JEPA components working correctly!")