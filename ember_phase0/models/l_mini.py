import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio.Spectrogram import MelSpectrogram
import math
from einops import rearrange


class Patchify(nn.Module):
    """
    Convert spectrogram into patches for JEPA processing
    """
    def __init__(self, patch_height=8, patch_width=8, stride_h=None, stride_w=None):
        super().__init__()
        
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.stride_h = stride_h or patch_height
        self.stride_w = stride_w or patch_width
        
    def forward(self, spec):
        """
        Convert spectrogram into patches
        
        Args:
            spec: (batch, freq_bins, time_steps) - mel spectrogram
            
        Returns:
            patches: (batch, num_patches, patch_dim) - flattened patches
        """
        batch_size, freq_bins, time_steps = spec.shape
        
        # Calculate number of patches
        num_freq_patches = (freq_bins - self.patch_height) // self.stride_h + 1
        num_time_patches = (time_steps - self.patch_width) // self.stride_w + 1
        
        # Extract patches using unfold
        patches = spec.unfold(1, self.patch_height, self.stride_h) \
                     .unfold(2, self.patch_width, self.stride_w)  # (batch, num_freq_patches, num_time_patches, patch_h, patch_w)
        
        # Reshape to (batch, num_patches, patch_dim)
        patches = patches.contiguous().view(batch_size, num_freq_patches * num_time_patches, -1)
        
        return patches


class JEPAContextEncoder(nn.Module):
    """
    JEPA-style encoder that creates context representations from masked input
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.1, mask_ratio=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mask_ratio = mask_ratio
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers for temporal processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection to final embedding
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, mask_ratio=None):
        """
        Forward pass with masking to create context representation
        
        Args:
            x: (batch, seq_len, input_dim) - input spectrogram patches
            mask_ratio: ratio of patches to mask (not see)
            
        Returns:
            context_repr: (batch, seq_len, output_dim) - masked context representation
            target_repr: (batch, seq_len, output_dim) - full target representation (for comparison)
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        batch_size, seq_len, input_dim = x.shape
        
        # Encode full sequence to get target representation
        full_x = self.input_proj(x)
        full_x = self.dropout(full_x)
        full_x = self.norm(full_x)
        encoded_full = self.encoder(full_x)  # (batch, seq_len, hidden_dim)
        target_repr = self.output_proj(encoded_full)  # (batch, seq_len, output_dim)
        
        # Create random mask for context encoder
        mask = torch.rand(batch_size, seq_len, device=x.device) < mask_ratio
        mask = mask.unsqueeze(-1).expand(-1, -1, self.input_dim)  # (batch, seq_len, input_dim)
        
        # Mask the input for context encoder and encode
        masked_x = x.clone()
        masked_x.masked_fill_(mask, 0)  # Zero out masked positions
        
        masked_x_proj = self.input_proj(masked_x)
        masked_x_proj = self.dropout(masked_x_proj)
        masked_x_proj = self.norm(masked_x_proj)
        encoded_masked = self.encoder(masked_x_proj)  # (batch, seq_len, hidden_dim)
        context_repr = self.output_proj(encoded_masked)  # (batch, seq_len, output_dim)
        
        return context_repr, target_repr


class JEPAContextPredictor(nn.Module):
    """
    Predictor network that predicts target representations from context representations
    """
    def __init__(self, context_dim=512, target_dim=512, hidden_dim=1024, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.context_dim = context_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        
        # Small predictor network to avoid representation collapse
        layers = []
        layers.append(nn.Linear(context_dim, hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, target_dim))
        
        self.predictor = nn.Sequential(*layers)
        
    def forward(self, context_repr):
        """
        Predict target representations from context representations
        
        Args:
            context_repr: (batch, seq_len, context_dim) - context representations
            
        Returns:
            pred_repr: (batch, seq_len, target_dim) - predicted target representations
        """
        return self.predictor(context_repr)


class DistressJEPALoss(nn.Module):
    """
    JEPA-style loss function for distress vocalization temporal prediction
    """
    def __init__(self, alpha=0.7, beta=0.3, temperature=0.2):
        super().__init__()
        self.alpha = alpha  # weight for prediction loss
        self.beta = beta    # weight for variance loss (prevent collapse)
        self.temperature = temperature
        
    def forward(self, pred_repr, target_repr):
        """
        Compute JEPA loss
        
        Args:
            pred_repr: (batch, seq_len, dim) - predicted representations
            target_repr: (batch, seq_len, dim) - target representations
            
        Returns:
            loss: scalar loss value
            metrics: dict with individual loss components
        """
        # Normalize representations
        pred_norm = F.normalize(pred_repr, dim=-1)
        target_norm = F.normalize(target_repr, dim=-1)
        
        # Compute cosine similarity loss
        similarity = (pred_norm * target_norm).sum(dim=-1)  # (batch, seq_len)
        pred_loss = -similarity.mean()
        
        # Variance loss to prevent representation collapse
        target_var = torch.var(target_norm, dim=0).mean()
        var_loss = torch.clamp(self.temperature - target_var, min=0.0)
        
        # Total loss
        total_loss = self.alpha * pred_loss + self.beta * var_loss
        
        metrics = {
            'prediction_loss': pred_loss,
            'variance_loss': var_loss,
            'target_variance': target_var
        }
        
        return total_loss, metrics

class CochlearFilterBank(nn.Module):
    """Cochlear-inspired filter bank using gammatone filters"""
    def __init__(self, n_filters=32, sample_rate=16000, n_fft=512):
        super().__init__()
        self.mel_spec = MelSpectrogram(
            sr=sample_rate,
            n_mels=n_filters,
            n_fft=n_fft,
            win_length=int(0.025 * sample_rate),  # 25ms window
            hop_length=int(0.01 * sample_rate)    # 10ms hop
        )
        
    def forward(self, x):
        # x shape: (batch, time)
        # Output shape: (batch, n_filters, time_steps)
        return self.mel_spec(x)


class SNNPhaseEncoder(nn.Module):
    """SNN encoder that uses phase-of-firing encoding"""
    def __init__(self, input_size, hidden_size, threshold=1.0, tau=2.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.tau = tau
        
        # Linear transformation to hidden layer
        self.linear = nn.Linear(input_size, hidden_size)
        
        # Simulate LIF dynamics
        self.beta = 1.0 / tau
    
    def forward(self, x):
        # x shape: (batch, time_steps, input_features)
        batch_size, time_steps, _ = x.shape
        
        # Reshape for temporal processing
        # We'll process each time step separately for phase encoding
        spikes = []
        mem = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        for t in range(time_steps):
            input_t = x[:, t, :]  # (batch, input_features)
            hidden_t = self.linear(input_t)  # (batch, hidden_size)
            
            # Simulate LIF neuron dynamics
            # Membrane potential update: mem_new = beta * mem_old + input
            mem = self.beta * mem + hidden_t
            
            # Generate spikes when membrane potential exceeds threshold
            spk = (mem > self.threshold).float()
            
            # Reset membrane potential where spikes occurred (subtractive reset)
            mem = mem * (1 - spk)  # Reset to 0 where there was a spike
            
            spikes.append(spk)
        
        # Stack spikes: (batch, time_steps, hidden_size)
        spike_tensor = torch.stack(spikes, dim=1)
        
        return spike_tensor


class SNNTemporalProcessor(nn.Module):
    """SNN layer that processes temporal relationships in spike trains"""
    def __init__(self, input_size, output_size, tau=2.0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tau = tau
        
        # Temporal convolution to capture timing relationships
        self.temporal_conv = nn.Conv1d(
            input_size, 
            output_size, 
            kernel_size=3, 
            padding=1
        )
        
        # Simple recurrent layer to capture temporal dynamics
        self.recurrent_layer = nn.GRU(
            input_size=output_size,
            hidden_size=output_size,
            batch_first=True,
            num_layers=1
        )
        
        # Linear projection to final embedding
        self.projection = nn.Linear(output_size, output_size)
    
    def forward(self, spike_trains):
        # spike_trains shape: (batch, time_steps, hidden_size)
        batch_size, time_steps, hidden_size = spike_trains.shape
        
        # Transpose for convolution: (batch, hidden_size, time_steps)
        spike_trains_t = spike_trains.transpose(1, 2)
        
        # Apply temporal convolution: (batch, output_size, time_steps)
        conv_out = self.temporal_conv(spike_trains_t)
        
        # Transpose back: (batch, time_steps, output_size)
        conv_out = conv_out.transpose(1, 2)
        
        # Process with recurrent layer to capture temporal dynamics
        recurrent_out, _ = self.recurrent_layer(conv_out)
        
        # Average over time to get final embedding: (batch, output_size)
        final_embedding = recurrent_out.mean(dim=1)  # (batch, output_size)
        
        # Apply final projection
        final_embedding = self.projection(final_embedding)
        
        return final_embedding


class LMini(nn.Module):
    """L-mini: SNN-based audio processor (20M params)"""
    def __init__(self, 
                 embed_dim=512,
                 n_filters=32,
                 sample_rate=16000,
                 n_fft=512,
                 snn_tau=2.0,
                 snn_threshold=1.0):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.sample_rate = sample_rate
        
        # Cochlear-inspired filter bank
        self.cochlear_filter = CochlearFilterBank(
            n_filters=n_filters,
            sample_rate=sample_rate,
            n_fft=n_fft
        )
        
        # SNN encoder with phase-of-firing encoding
        self.snn_encoder = SNNPhaseEncoder(
            input_size=n_filters,
            hidden_size=embed_dim // 2,
            threshold=snn_threshold,
            tau=snn_tau
        )
        
        # SNN temporal processor
        self.snn_temporal = SNNTemporalProcessor(
            input_size=embed_dim // 2,
            output_size=embed_dim,
            tau=snn_tau
        )
        
        # JEPA components for temporal prediction learning
        self.patchify = Patchify(patch_height=4, patch_width=4)
        
        # Calculate patch dimension for JEPA: n_filters * patch_height * patch_width
        patch_dim = n_filters * 4 * 4  # Assuming 4x4 patch size from Patchify initialization
        
        # JEPA context encoder (sees masked input)
        self.jepa_context_encoder = JEPAContextEncoder(
            input_dim=patch_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim
        )
        
        # JEPA predictor (predicts target from context)
        self.jepa_predictor = JEPAContextPredictor(
            context_dim=embed_dim,
            target_dim=embed_dim,
            hidden_dim=embed_dim * 2
        )
        
        # Final projection combining SNN and JEPA features
        self.final_proj = nn.Linear(embed_dim * 2, embed_dim)
        
        # Normalization and regularization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Regularization to maintain temporal structure
        self.dropout = nn.Dropout(0.1)
        
        # Initialize parameters
        self._init_weights()
        
        # Loss function for JEPA training
        self.jepa_loss = DistressJEPALoss()
    
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
    
    def forward(self, audio_input, use_jepa=True):
        """
        Process audio input through SNN + JEPA pipeline
        
        Args:
            audio_input: (batch, time_samples) - raw audio waveform
            use_jepa: Whether to use JEPA components (for ablation studies)
            
        Returns:
            embedding: (batch, embed_dim) - temporal-feature preserving embedding
        """
        # Step 1: Cochlear filtering
        # Output: (batch, n_filters, time_steps)
        cochlear_out = self.cochlear_filter(audio_input)
        
        # Get SNN embedding
        # Transpose for SNN processing: (batch, time_steps, n_filters)
        cochlear_out_t = cochlear_out.transpose(1, 2)
        # Output: (batch, time_steps, embed_dim//2) - spike trains
        spike_trains = self.snn_encoder(cochlear_out_t)
        # Output: (batch, embed_dim) - SNN embedding
        snn_embedding = self.snn_temporal(spike_trains)
        
        if use_jepa:
            # Process with JEPA components for temporal prediction learning
            # Patchify the mel spectrogram: (batch, num_patches, patch_dim)
            patches = self.patchify(cochlear_out)
            
            # Get context and target representations via JEPA
            context_repr, target_repr = self.jepa_context_encoder(patches)
            
            # Predict target from context
            pred_repr = self.jepa_predictor(context_repr)  # (batch, num_patches, embed_dim)
            
            # Average over patches to get final JEPA embedding: (batch, embed_dim)
            jepa_embedding = pred_repr.mean(dim=1)
            
            # Combine SNN and JEPA features
            combined_features = torch.cat([snn_embedding, jepa_embedding], dim=1)  # (batch, embed_dim * 2)
            embedding = self.final_proj(combined_features)  # (batch, embed_dim)
        else:
            # Use only SNN embedding (for control experiments)
            embedding = snn_embedding
        
        # Apply normalization and dropout
        embedding = self.norm(embedding)
        embedding = self.dropout(embedding)
        
        return embedding

    def compute_jepa_loss(self, audio_input):
        """
        Compute JEPA self-supervised loss for temporal prediction learning
        
        Args:
            audio_input: (batch, time_samples) - raw audio waveform
            
        Returns:
            loss: scalar loss value
            metrics: dict with additional metrics
        """
        # Get mel spectrogram
        mel_spec = self.cochlear_filter(audio_input)
        
        # Patchify: (batch, num_patches, patch_dim)
        patches = self.patchify(mel_spec)
        
        # Get context and target representations
        context_repr, target_repr = self.jepa_context_encoder(patches)
        
        # Predict target from context
        pred_repr = self.jepa_predictor(context_repr)  # (batch, num_patches, embed_dim)
        
        # Compute JEPA loss
        loss, metrics = self.jepa_loss(pred_repr, target_repr)
        
        return loss, metrics

    def get_spike_statistics(self, audio_input):
        """Get statistics about spiking activity for monitoring"""
        with torch.no_grad():
            cochlear_out = self.cochlear_filter(audio_input)
            cochlear_out = cochlear_out.transpose(1, 2)
            spike_trains = self.snn_encoder(cochlear_out)
            
            spike_rate = spike_trains.sum() / spike_trains.numel()
            max_spike_rate = spike_trains.sum(dim=[1, 2]).max() / spike_trains.shape[1]
            
            return {
                'mean_spike_rate': spike_rate.item(),
                'max_frame_spike_rate': max_spike_rate.item(),
                'spike_trains_shape': spike_trains.shape
            }


# Example usage and testing
if __name__ == "__main__":
    # Test the L-mini model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    l_mini = LMini(embed_dim=512)
    l_mini.to(device)
    
    # Generate dummy audio input (batch=2, 16kHz * 2sec = 32000 samples)
    dummy_audio = torch.randn(2, 32000, device=device)
    
    # Forward pass
    embedding = l_mini(dummy_audio)
    
    print(f"Audio input shape: {dummy_audio.shape}")
    print(f"Output embedding shape: {embedding.shape}")
    
    # Check spike statistics
    stats = l_mini.get_spike_statistics(dummy_audio)
    print(f"Spike statistics: {stats}")
    
    # Count parameters
    total_params = sum(p.numel() for p in l_mini.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Target: ~20M, Actual: {total_params/1_000_000:.1f}M")