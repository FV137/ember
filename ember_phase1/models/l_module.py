import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from configs.l_module_config import LModuleConfig
from .cochlear_processing import CochlearFilterBank
from .spiking_layers import SpikingJEPAEncoder, PhaseEncodingSNN
from .jepa_components import TemporalMasking, PredictiveCodingModule, CrossTemporalAttention, JEPALoss


class JEPAStylePredictor(nn.Module):
    """
    JEPA-style predictor that learns to predict temporal dynamics
    Unlike traditional autoencoders, JEPA learns representations by predicting
    future states from present states, without semantic labels
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Multi-layer predictor with residual connections
        layers = []
        layers.append(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ))
        
        for _ in range(n_layers - 2):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.predictor = nn.ModuleList(layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate scaling"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict future/past states from current state
        
        Args:
            x: (batch, sequence_length, features) or (batch, features)
            
        Returns:
            predictions: (batch, sequence_length, output_dim) or (batch, output_dim)
        """
        if x.dim() == 2:
            # Single feature vector
            for layer in self.predictor:
                x = layer(x)
        else:
            # Sequence of feature vectors
            batch_size, seq_len, features = x.shape
            x = x.view(-1, features)  # Reshape to (batch*seq_len, features)
            
            for layer in self.predictor:
                x = layer(x)
            
            x = x.view(batch_size, seq_len, -1)  # Reshape back
        
        return x


from configs.l_module_config import LModuleConfig


class LModule(nn.Module):
    """
    Phase 1 L-Module: SNN-based audio processor (100M params)
    Combines biological cochlear processing with spiking neural networks
    and JEPA-style self-supervised learning
    """
    def __init__(self, config: LModuleConfig = None):
        super().__init__()
        
        if config is None:
            config = LModuleConfig()
        
        self.config = config
        self.embed_dim = config.embed_dim
        self.sample_rate = config.sample_rate
        self.n_filters = config.n_filters
        
        # Cochlear-inspired preprocessing
        self.cochlear_filter = CochlearFilterBank(
            sample_rate=config.sample_rate,
            n_filters=config.n_filters,
            low_freq=config.low_freq,
            high_freq=config.high_freq,
            compression_factor=config.compression_factor
        )
        
        # Phase-of-firing encoding SNN with increased capacity
        self.phase_encoder = PhaseEncodingSNN(
            input_size=config.n_filters,
            hidden_size=config.snn_hidden_size,
            output_size=config.snn_output_size,  # Changed to support larger intermediate representations
            n_time_steps=config.target_time_steps,
            beta=config.snn_beta,
            threshold=config.snn_threshold
        )
        
        # Additional projection to match embed_dim
        self.phase_to_embed = nn.Linear(config.snn_output_size, config.embed_dim)
        
        # JEPA-style encoder with increased capacity
        self.jepa_encoder = SpikingJEPAEncoder(
            input_size=config.n_filters,
            hidden_size=config.jepa_hidden_size,
            embed_dim=config.embed_dim,
            n_layers=config.n_jepa_layers,
            beta=config.snn_beta,
            threshold=config.snn_threshold
        )
        
        # Self-supervised predictor (for temporal dynamics prediction)
        self.predictor = JEPAStylePredictor(
            input_dim=config.embed_dim,
            hidden_dim=config.predictor_hidden_dim,
            output_dim=config.embed_dim,
            n_layers=config.predictor_layers,
            dropout=config.dropout
        )
        
        # Additional processing layers to increase parameter count
        self.processing_layers = nn.ModuleList()
        for _ in range(2):  # Add 2 additional processing layers
            self.processing_layers.append(nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim * 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.embed_dim * 2, config.embed_dim),
                nn.Dropout(config.dropout)
            ))
        
        # JEPA-specific components for enhanced self-supervised learning
        self.temporal_masking = TemporalMasking(
            mask_ratio=config.mask_ratio,
            mask_type='random'
        )
        
        # Predictive coding module for temporal prediction
        self.predictive_coder = PredictiveCodingModule(
            embed_dim=config.embed_dim,
            hidden_dim=config.predictor_hidden_dim // 2,  # Reduced to manage params
            prediction_steps=3,
            num_layers=3
        )
        
        # Cross-temporal attention for better temporal relationships
        self.cross_temporal_attn = CrossTemporalAttention(
            embed_dim=config.embed_dim,
            num_heads=max(4, config.embed_dim // 128),  # Ensure at least 4 heads
            dropout=config.dropout
        )
        
        # Additional projection layers for JEPA
        self.jepa_proj = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, config.embed_dim)
        )
        
        # Final projection to ensure consistent output dimension
        self.final_proj = nn.Linear(config.embed_dim, config.embed_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize parameters
        self._init_weights()
        
        # Count parameters to aim for ~100M
        self._parameter_count = self._count_parameters()
        print(f"L-Module initialized with {self._parameter_count:,} parameters (target: {config.target_params:,})")
        
        # Check if we're close to target parameters
        if abs(self._parameter_count - config.target_params) > config.target_params * 0.3:  # 30% tolerance
            print(f"Warning: Parameter count is {(self._parameter_count / config.target_params * 100):.1f}% of target. "
                  f"Consider adjusting configuration.")
        
    def _init_weights(self):
        """Initialize weights with appropriate scaling"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _count_parameters(self):
        """Count total parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, 
                audio_input: torch.Tensor, 
                return_intermediates: bool = False,
                mask_ratio: float = 0.0,
                apply_masking: bool = False) -> torch.Tensor:
        """
        Forward pass through the L-module
        
        Args:
            audio_input: (batch, time_samples) - raw audio waveform
            return_intermediates: Whether to return intermediate representations
            mask_ratio: Ratio of temporal positions to mask for self-supervised training
            apply_masking: Whether to apply temporal masking during forward pass
            
        Returns:
            embedding: (batch, embed_dim) - temporal-feature preserving embedding
        """
        batch_size = audio_input.shape[0]
        
        # Step 1: Cochlear filtering
        # Output: (batch, n_filters, time_steps)
        cochlear_out = self.cochlear_filter(audio_input)
        
        # Transpose for SNN processing: (batch, time_steps, n_filters)
        cochlear_out = cochlear_out.transpose(1, 2)
        
        # Pad or truncate to fixed length for consistent processing
        target_length = self.config.target_time_steps  # Use config value
        current_length = cochlear_out.shape[1]
        
        if current_length < target_length:
            # Pad with zeros
            pad_length = target_length - current_length
            cochlear_out = F.pad(cochlear_out, (0, 0, 0, pad_length), mode='constant', value=0)
        elif current_length > target_length:
            # Truncate
            cochlear_out = cochlear_out[:, :target_length, :]
        
        # Step 2: Phase encoding via SNN
        phase_encoded = self.phase_encoder(cochlear_out)  # (batch, snn_output_size)
        phase_encoded = self.phase_to_embed(phase_encoded)  # (batch, embed_dim)
        
        # Step 3: JEPA-style encoding
        jepa_encoded = self.jepa_encoder(cochlear_out)  # (batch, embed_dim)
        
        # Combine both representations
        combined_embed = phase_encoded + jepa_encoded  # (batch, embed_dim)
        
        # Reshape for temporal processing: (batch, time_steps, embed_dim)
        seq_len = self.config.target_time_steps
        # Reshape combined_embed to have temporal dimension
        combined_embed = combined_embed.unsqueeze(1).expand(-1, seq_len, -1)  # Broadcast to temporal dimension
        
        # Apply cross-temporal attention
        combined_embed = self.cross_temporal_attn(combined_embed)
        
        # Apply additional processing layers with temporal awareness
        for layer in self.processing_layers:
            residual = combined_embed
            combined_embed = layer(combined_embed)
            combined_embed = combined_embed + residual  # Residual connection
        
        # Apply JEPA projection
        jepa_output = self.jepa_proj(combined_embed)  # (batch, time_steps, embed_dim)
        
        # Average over time dimension to get final representation
        final_embed = jepa_output.mean(dim=1)  # (batch, embed_dim)
        
        # Apply dropout
        final_embed = self.dropout(final_embed)
        
        # Final projection
        final_embed = self.final_proj(final_embed)  # (batch, embed_dim)
        
        if return_intermediates:
            return {
                'cochlear_output': cochlear_out,
                'phase_encoded': phase_encoded,
                'jepa_encoded': jepa_encoded,
                'temporal_features': combined_embed,
                'jepa_output': jepa_output,
                'final_embedding': final_embed
            }
        
        return final_embed
    
    def self_supervised_forward(self, audio_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for self-supervised learning (JEPA training)
        
        Args:
            audio_input: (batch, time_samples) - raw audio waveform
            
        Returns:
            embedding: (batch, embed_dim) - representation
            prediction: (batch, embed_dim) - predicted representation
            target: (batch, embed_dim) - target representation
        """
        batch_size = audio_input.shape[0]
        
        # Step 1: Cochlear filtering
        cochlear_out = self.cochlear_filter(audio_input)
        cochlear_out = cochlear_out.transpose(1, 2)
        
        # Pad or truncate to fixed length
        target_length = self.config.target_time_steps
        current_length = cochlear_out.shape[1]
        if current_length < target_length:
            pad_length = target_length - current_length
            cochlear_out = F.pad(cochlear_out, (0, 0, 0, pad_length), mode='constant', value=0)
        elif current_length > target_length:
            cochlear_out = cochlear_out[:, :target_length, :]
        
        # Get temporal features through the network
        phase_encoded = self.phase_encoder(cochlear_out)
        phase_encoded = self.phase_to_embed(phase_encoded)
        jepa_encoded = self.jepa_encoder(cochlear_out)
        
        # Combine representations
        combined_embed = phase_encoded + jepa_encoded  # (batch, embed_dim)
        
        # Expand to temporal dimension: (batch, time_steps, embed_dim)
        seq_len = self.config.target_time_steps
        temporal_features = combined_embed.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Apply cross-temporal attention
        temporal_features = self.cross_temporal_attn(temporal_features)
        
        # Apply processing layers
        for layer in self.processing_layers:
            residual = temporal_features
            temporal_features = layer(temporal_features)
            temporal_features = temporal_features + residual
        
        # Apply JEPA projection
        jepa_features = self.jepa_proj(temporal_features)  # (batch, time_steps, embed_dim)
        
        # Get main embedding (average over time)
        main_embedding = jepa_features.mean(dim=1)  # (batch, embed_dim)
        
        # For predictive coding - predict future temporal steps
        predictions, _ = self.predictive_coder(jepa_features)  # (batch, time_steps-pred_steps, pred_steps, embed_dim)
        
        # Reshape predictions to match target format: (batch, time_steps, embed_dim)
        pred_steps = predictions.shape[2]
        pred_reshaped = predictions.mean(dim=2)  # Average over prediction steps: (batch, effective_time, embed_dim)
        
        # Target is the actual features (current or shifted)
        target_features = jepa_features.detach()  # Detach to not backprop through target
        
        # For prediction, we can use a shifted version as target
        # Predict next time steps based on current
        if pred_reshaped.shape[1] <= target_features.shape[1]:
            prediction = self.predictor(pred_reshaped.mean(dim=1))  # (batch, embed_dim)
            target = target_features.mean(dim=1).detach()  # (batch, embed_dim)
        else:
            # Handle case where prediction time is different
            prediction = self.predictor(main_embedding)
            target = main_embedding.detach()
        
        return main_embedding, prediction, target


class TemporalCoherenceLoss(nn.Module):
    """
    Loss function that encourages temporal coherence in the embeddings
    """
    def __init__(self, lambda_temporal: float = 1.0, lambda_energy: float = 0.1):
        super().__init__()
        self.lambda_temporal = lambda_temporal
        self.lambda_energy = lambda_energy
    
    def forward(self, embeddings: torch.Tensor, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss combining prediction accuracy and temporal coherence
        
        Args:
            embeddings: (batch, seq_len, embed_dim) - main embeddings
            predictions: (batch, seq_len, embed_dim) - predicted embeddings
            targets: (batch, seq_len, embed_dim) - target embeddings
            
        Returns:
            loss: scalar - combined loss
        """
        # Prediction loss (MSE between predictions and targets)
        pred_loss = F.mse_loss(predictions, targets)
        
        # Temporal coherence loss - encourage smooth transitions
        if embeddings.dim() == 3 and embeddings.shape[1] > 1:  # batch, seq_len, embed_dim
            temporal_diff = embeddings[:, 1:, :] - embeddings[:, :-1, :]
            temporal_loss = torch.mean(torch.norm(temporal_diff, p=2, dim=2))
        else:
            temporal_loss = 0.0
        
        # Energy efficiency loss - encourage sparse representations
        energy_loss = torch.mean(torch.abs(embeddings))
        
        # Combine losses
        total_loss = pred_loss + self.lambda_temporal * temporal_loss + self.lambda_energy * energy_loss
        
        return total_loss


# Test the L-Module
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Testing L-Module...")
    
    # Create L-Module
    l_module = LModule(
        embed_dim=768,
        n_filters=128,
        snn_hidden_size=512,
        n_snn_layers=4,
        n_jepa_layers=3
    ).to(device)
    
    # Test with dummy audio
    dummy_audio = torch.randn(2, 32000, device=device)  # 2s at 16kHz
    
    print(f"Input audio shape: {dummy_audio.shape}")
    
    # Forward pass
    output = l_module(dummy_audio)
    print(f"Output embedding shape: {output.shape}")
    
    # Test self-supervised forward pass
    embedding, prediction, target = l_module.self_supervised_forward(dummy_audio)
    print(f"Self-supervised shapes - Embedding: {embedding.shape}, "
          f"Prediction: {prediction.shape}, Target: {target.shape}")
    
    # Test with intermediates
    intermediates = l_module(dummy_audio, return_intermediates=True)
    print(f"Intermediate shapes:")
    print(f"  Cochlear output: {intermediates['cochlear_output'].shape}")
    print(f"  Phase encoded: {intermediates['phase_encoded'].shape}")
    print(f"  JEPA encoded: {intermediates['jepa_encoded'].shape}")
    print(f"  Final embedding: {intermediates['final_embedding'].shape}")
    
    print(f"\nL-Module successfully created with {l_module._parameter_count:,} parameters")
    print("L-Module working correctly!")