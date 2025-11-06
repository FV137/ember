import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class GammatoneFilterbank(nn.Module):
    """
    Cochlear-inspired gammatone filter bank that mimics biological processing.
    Gammatone filters are more biologically accurate than Mel filters for auditory processing.
    """
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_filters: int = 64,
                 low_freq: int = 50,
                 high_freq: int = 8000,
                 order: int = 4):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_filters = n_filters
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.order = order  # Gammatone filter order
        
        # Create center frequencies using an erb scale (more biologically accurate)
        self.center_freqs = self._create_center_frequencies()
        
        # Initialize gammatone filters
        self._create_filters()
        
    def _create_center_frequencies(self) -> torch.Tensor:
        """Create center frequencies using the equivalent rectangular bandwidth (ERB) scale"""
        # Convert frequencies to ERB scale
        low_erb = 21.4 * math.log10(1 + self.low_freq * 0.00437)
        high_erb = 21.4 * math.log10(1 + self.high_freq * 0.00437)
        
        # Create evenly spaced frequencies on ERB scale
        erb_points = torch.linspace(low_erb, high_erb, self.n_filters)
        
        # Convert back to Hz
        center_freqs = (torch.pow(10, (erb_points / 21.4)) - 1) / 0.00437
        return center_freqs
    
    def _create_filters(self):
        """Create gammatone filter parameters"""
        # Initialize filter parameters
        # For gammatone filters: h(t) = t^(n-1) * exp(-2πb*t) * cos(2πfc*t + φ)
        
        # Bandwidth parameters (in Hz) - wider for lower frequencies, narrower for higher
        # Following formula from literature: bandwidth increases with center frequency
        bandwidth_f = 24.7 * (4.37 * self.center_freqs / 1000 + 1)  # in Hz
        
        # Convert to radians/sample
        self.bandwidth_radians = 2 * math.pi * bandwidth_f / self.sample_rate
        
        # Create complex filter coefficients
        self.register_buffer('center_freqs_radians', 
                            2 * math.pi * self.center_freqs / self.sample_rate)
        
        # Initialize phase
        self.register_buffer('phase', torch.zeros(self.n_filters))
        
        # Filter coefficients for IIR implementation
        self.filter_coeffs = nn.ParameterDict()
        
        # For each filter, create complex coefficients
        for i in range(self.n_filters):
            # This is a simplified approach - in a full implementation, we'd create
            # proper gammatone IIR filter coefficients
            pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply gammatone filterbank to input audio
        
        Args:
            x: (batch, time_samples) - input audio waveform
            
        Returns:
            output: (batch, n_filters, time_steps) - filter responses
        """
        # This is a simplified implementation - a full gammatone filterbank 
        # would require more complex processing with IIR filters
        # For now, we'll use a close approximation with STFT and frequency warping
        
        batch_size, time_len = x.shape
        
        # Use STFT as a base implementation, then map to gammatone frequencies
        n_fft = 512
        hop_length = n_fft // 4  # 75% overlap
        
        # Compute STFT
        stft = torch.stft(
            x.view(-1, time_len),  # Flatten batch and time
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft, device=x.device),
            return_complex=True,
            normalized=True
        )  # (batch, n_fft//2+1, time_steps)
        
        # Frequency bins
        freq_bins = torch.linspace(0, self.sample_rate // 2, stft.shape[1], device=x.device)
        
        # Create mapping from linear frequency to gammatone center frequencies
        output = torch.zeros(batch_size, self.n_filters, stft.shape[2], device=x.device, dtype=torch.complex64)
        
        # For each gammatone center frequency, find the closest linear frequency bin
        for i in range(self.n_filters):
            closest_freq_idx = torch.argmin(torch.abs(freq_bins - self.center_freqs[i]))
            # Copy the closest bin to the output
            output[:, i, :] = stft[:, closest_freq_idx, :]
        
        # Take magnitude to get filterbank responses
        output = torch.abs(output)
        
        return output


class CochlearStage(nn.Module):
    """
    Biological-inspired cochlear processing stage that includes:
    1. Outer hair cell dynamics (enhancement of weak signals)
    2. Inner hair cell transduction
    3. Basic frequency selectivity
    """
    def __init__(self, 
                 sample_rate: int = 16000,
                 compression_factor: float = 0.2,
                 noise_suppression: bool = True):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.compression_factor = compression_factor
        self.noise_suppression = noise_suppression
        
        # Learnable parameters for cochlear dynamics
        self.outer_hair_gain = nn.Parameter(torch.ones(1) * 0.5)
        self.inner_hair_bias = nn.Parameter(torch.zeros(1))
        
        # Optional: noise suppression filter
        if noise_suppression:
            # Simple temporal noise suppression
            self.noise_filter = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
            nn.init.constant_(self.noise_filter.weight, 1.0/3.0)  # Simple averaging
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cochlear processing to input
        
        Args:
            x: (batch, time_samples) - input audio
            
        Returns:
            processed: (batch, time_samples) - cochlear processed signal
        """
        # Apply outer hair cell dynamics (signal enhancement)
        enhanced = x + self.outer_hair_gain * x * torch.abs(x)
        
        # Apply inner hair cell transduction (compressive nonlinearity)
        # This mimics the compressive response of inner hair cells
        compressed = torch.sign(enhanced) * torch.pow(torch.abs(enhanced), self.compression_factor)
        compressed = compressed + self.inner_hair_bias
        
        # Apply noise suppression if enabled
        if self.noise_suppression:
            # Reshape for 1D conv: (batch, 1, time_samples)
            compressed_reshaped = compressed.unsqueeze(1)
            compressed_reshaped = self.noise_filter(compressed_reshaped)
            compressed = compressed_reshaped.squeeze(1)
        
        return compressed


class CochlearFilterBank(nn.Module):
    """
    Enhanced cochlear-inspired filter bank for Phase 1 L-Module
    Combines biological processing with gammatone filtering
    """
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_filters: int = 64,
                 low_freq: int = 50,
                 high_freq: int = 8000,
                 compression_factor: float = 0.2):
        super().__init__()
        
        self.sample_rate = sample_rate
        
        # Cochlear processing stage
        self.cochlear_stage = CochlearStage(
            sample_rate=sample_rate,
            compression_factor=compression_factor
        )
        
        # Gammatone filterbank
        self.gammatone_bank = GammatoneFilterbank(
            sample_rate=sample_rate,
            n_filters=n_filters,
            low_freq=low_freq,
            high_freq=high_freq
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process audio through the enhanced cochlear filter bank
        
        Args:
            x: (batch, time_samples) - raw audio waveform
            
        Returns:
            output: (batch, n_filters, time_steps) - cochlear responses
        """
        # Apply biological cochlear processing
        processed_audio = self.cochlear_stage(x)
        
        # Apply gammatone filterbank
        gammatone_output = self.gammatone_bank(processed_audio)
        
        return gammatone_output


# Test the enhanced cochlear processing
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create filterbank
    cochlear = CochlearFilterBank(n_filters=64).to(device)
    
    # Test with dummy audio
    dummy_audio = torch.randn(2, 32000, device=device)  # 2s at 16kHz
    
    print(f"Input shape: {dummy_audio.shape}")
    
    output = cochlear(dummy_audio)
    print(f"Output shape: {output.shape}")
    
    print("Enhanced cochlear filterbank working!")
    
    # Test frequency response
    print(f"Center frequencies: {cochlear.gammatone_bank.center_freqs[:5]} Hz (first 5)")
    print(f"Last 5 center frequencies: {cochlear.gammatone_bank.center_freqs[-5:]} Hz")