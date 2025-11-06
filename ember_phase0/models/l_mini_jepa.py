"""
Enhanced L-mini with JEPA (Joint-Embedding Predictive Architecture) components
for learning temporal dynamics in distress vocalization data without semantic labels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from nnAudio.Spectrogram import MelSpectrogram


class JEPAEncoder(nn.Module):
    """
    JEPA-style encoder that creates context and target representations
    for self-supervised temporal prediction learning.
    """
    def __init__(self, input_dim=32, hidden_dim=512, output_dim=512, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
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
        
    def forward(self, x):
        """
        Forward pass through encoder
        
        Args:
            x: (batch, seq_len, input_dim) - input spectrogram patches
            
        Returns:
            context_repr: (batch, seq_len, output_dim) - representations for context
            target_repr: (batch, seq_len, output_dim) - representations for targets
        """
        # Project input
        x = self.input_proj(x)
        x = self.dropout(x)
        x = self.norm(x)
        
        # Process with transformer
        encoded = self.encoder(x)  # (batch, seq_len, hidden_dim)
        
        # Project to output dimension
        output = self.output_proj(encoded)  # (batch, seq_len, output_dim)
        
        # Return separate representations for context and target
        # This allows for asymmetric encoding if needed
        context_repr = output
        target_repr = output  # Initially same; can be made different through masking
        
        return context_repr, target_repr


class JEPAContextEncoder(JEPAEncoder):
    """
    Context encoder that only sees a subset of the input (masked patches)
    """
    def forward(self, x, mask_ratio=0.5):
        """
        Forward pass with masking to create context representation
        
        Args:
            x: (batch, seq_len, input_dim) - input spectrogram patches
            mask_ratio: ratio of patches to mask (not see)
            
        Returns:
            context_repr: (batch, seq_len, output_dim) - masked context representation
            target_repr: (batch, seq_len, output_dim) - full target representation
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Create random mask
        mask = torch.rand(batch_size, seq_len, device=x.device) < mask_ratio
        mask = mask.unsqueeze(-1).expand(-1, -1, self.output_dim)  # (batch, seq_len, output_dim)
        
        # Encode full sequence to get target representation
        _, target_repr = super().forward(x)
        
        # Mask the input for context encoder
        # Use zeros to indicate masked positions
        masked_x = x.masked_fill(mask[..., :self.input_dim].expand(-1, -1, self.input_dim), 0)
        
        # Encode masked input to get context representation
        context_repr, _ = super().forward(masked_x)
        
        return context_repr, target_repr


class JPAPredictor(nn.Module):
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


class DistressJEPALoss(nn.Module):
    """
    JEPA-style loss function for distress vocalization temporal prediction
    """
    def __init__(self, alpha=0.5, beta=0.1, temperature=0.2):
        super().__init__()
        self.alpha = alpha  # weight for prediction loss
        self.beta = beta    # weight for variance loss (prevent collapse)
        self.temperature = temperature
        
    def forward(self, pred_repr, target_repr, pos_mask):
        """
        Compute JEPA loss
        
        Args:
            pred_repr: (batch, seq_len, dim) - predicted representations
            target_repr: (batch, seq_len, dim) - target representations
            pos_mask: (batch, seq_len) - mask indicating positive pairs
            
        Returns:
            loss: scalar loss value
        """
        # Prediction loss - cosine similarity between predicted and target
        pred_flat = rearrange(pred_repr, 'b s d -> (b s) d')
        target_flat = rearrange(target_repr, 'b s d -> (b s) d')
        
        # Normalize representations
        pred_norm = F.normalize(pred_flat, dim=-1)
        target_norm = F.normalize(target_flat, dim=-1)
        
        # Compute cosine similarity
        similarities = torch.sum(pred_norm * target_norm, dim=-1)  # (batch * seq_len,)
        
        # Apply position mask (only consider positive pairs)
        mask_flat = rearrange(pos_mask, 'b s -> (b s)')
        pos_similarities = similarities[mask_flat]
        
        # Prediction loss (negative cosine similarity)
        pred_loss = -pos_similarities.mean()
        
        # Variance loss to prevent representation collapse
        target_var = torch.var(target_norm, dim=0).mean()
        var_loss = torch.clamp(self.temperature - target_var, min=0.0)
        
        # Total loss
        total_loss = self.alpha * pred_loss + self.beta * var_loss
        
        return total_loss, {'pred_loss': pred_loss, 'var_loss': var_loss}


class LMiniJEPA(nn.Module):
    """
    L-mini with integrated JEPA components for learning temporal dynamics
    in distress vocalization data without semantic labels.
    """
    def __init__(self, 
                 embed_dim=512,
                 n_filters=32,
                 sample_rate=16000,
                 n_fft=512,
                 snn_tau=2.0,
                 snn_threshold=1.0,
                 jepa_mask_ratio=0.3,
                 jepa_hidden_dim=1024):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.sample_rate = sample_rate
        self.jepa_mask_ratio = jepa_mask_ratio
        
        # Cochlear-inspired filter bank
        self.cochlear_filter = MelSpectrogram(
            sr=sample_rate,
            n_mels=n_filters,
            n_fft=n_fft,
            win_length=int(0.025 * sample_rate),  # 25ms window
            hop_length=int(0.01 * sample_rate)    # 10ms hop
        )
        
        # Patchify the spectrogram for JEPA processing
        self.patchify = Patchify(patch_height=4, patch_width=4)
        
        # SNN components for phase-of-firing encoding
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
        
        # JEPA components
        self.jepa_context_encoder = JEPAContextEncoder(
            input_dim=n_filters,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            num_layers=4
        )
        
        self.jepa_predictor = JPAPredictor(
            context_dim=embed_dim,
            target_dim=embed_dim,
            hidden_dim=jepa_hidden_dim
        )
        
        # Final projection to embedding dimension
        self.final_proj = nn.Linear(embed_dim * 2, embed_dim)  # Combine SNN and JEPA features
        
        # Regularization
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, audio_input, return_jepa_components=False):
        """
        Forward pass through L-mini with JEPA
        
        Args:
            audio_input: (batch, time_samples) - raw audio waveform
            return_jepa_components: Whether to return intermediate JEPA representations
            
        Returns:
            embedding: (batch, embed_dim) - final embedding
            If return_jepa_components=True:
                - embedding
                - context_repr
                - target_repr  
                - pred_repr
        """
        # Step 1: Cochlear filtering to get mel spectrogram
        # audio_input: (batch, time_samples)
        # Output: (batch, n_filters, time_steps)
        mel_spec = self.cochlear_filter(audio_input)
        
        # Step 2: Apply SNN processing for phase-of-firing encoding
        # Transpose for SNN: (batch, time_steps, n_filters)
        mel_spec_t = mel_spec.transpose(1, 2)
        
        # Process with SNN
        spike_trains = self.snn_encoder(mel_spec_t)  # (batch, time_steps, embed_dim//2)
        snn_embedding = self.snn_temporal(spike_trains)  # (batch, embed_dim)
        
        # Step 3: Apply JEPA processing for temporal prediction
        # Patchify the mel spectrogram for JEPA
        patches = self.patchify(mel_spec)  # (batch, num_patches, patch_dim)
        
        # Get context and target representations via JEPA
        context_repr, target_repr = self.jepa_context_encoder(patches, mask_ratio=self.jepa_mask_ratio)
        
        # Predict target from context
        pred_repr = self.jepa_predictor(context_repr)  # (batch, num_patches, embed_dim)
        
        # Combine SNN and JEPA features
        combined_features = torch.cat([snn_embedding, pred_repr.mean(dim=1)], dim=1)  # (batch, embed_dim * 2)
        embedding = self.final_proj(combined_features)  # (batch, embed_dim)
        
        # Apply normalization and dropout
        embedding = self.norm(embedding)
        embedding = self.dropout(embedding)
        
        if return_jepa_components:
            return embedding, context_repr, target_repr, pred_repr
        else:
            return embedding
    
    def jepa_loss(self, audio_input):
        """
        Compute JEPA loss for self-supervised learning
        
        Args:
            audio_input: (batch, time_samples) - raw audio waveform
            
        Returns:
            total_loss: scalar loss value
            loss_components: dict with individual loss components
        """
        # Get representations
        _, context_repr, target_repr, pred_repr = self.forward(audio_input, return_jepa_components=True)
        
        # Create positive mask (all positions are positive for this simplified version)
        batch_size, seq_len, _ = target_repr.shape
        pos_mask = torch.ones(batch_size, seq_len, device=audio_input.device, dtype=torch.bool)
        
        # Compute JEPA loss
        loss_fn = DistressJEPALoss()
        total_loss, loss_components = loss_fn(pred_repr, target_repr, pos_mask)
        
        return total_loss, loss_components
    
    def get_spike_statistics(self, audio_input):
        """Get statistics about spiking activity for monitoring"""
        with torch.no_grad():
            mel_spec = self.cochlear_filter(audio_input)
            mel_spec_t = mel_spec.transpose(1, 2)
            spike_trains = self.snn_encoder(mel_spec_t)
            
            spike_rate = spike_trains.sum() / spike_trains.numel()
            max_frame_spike_rate = spike_trains.sum(dim=[1, 2]).max() / spike_trains.shape[1]
            
            return {
                'mean_spike_rate': spike_rate.item(),
                'max_frame_spike_rate': max_frame_spike_rate.item(),
                'spike_trains_shape': spike_trains.shape
            }


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
        
        # Process each time step separately for phase encoding
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


if __name__ == "__main__":
    # Test the JEPA-enhanced L-mini
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    l_mini_jepa = LMiniJEPA(embed_dim=512)
    l_mini_jepa.to(device)
    
    # Generate dummy audio input (batch=2, 16kHz * 2sec = 32000 samples)
    dummy_audio = torch.randn(2, 32000, device=device)
    
    # Forward pass
    embedding = l_mini_jepa(dummy_audio)
    
    print(f"Audio input shape: {dummy_audio.shape}")
    print(f"Output embedding shape: {embedding.shape}")
    
    # Test JEPA loss
    jepa_loss, loss_components = l_mini_jepa.jepa_loss(dummy_audio)
    print(f"JEPA loss: {jepa_loss.item():.4f}")
    print(f"Loss components: {loss_components}")
    
    # Test spike statistics
    stats = l_mini_jepa.get_spike_statistics(dummy_audio)
    print(f"Spike statistics: {stats}")
    
    # Count parameters
    total_params = sum(p.numel() for p in l_mini_jepa.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Target: ~20M, Actual: {total_params/1_000_000:.1f}M")
    
    print("L-mini with JEPA components created successfully!")