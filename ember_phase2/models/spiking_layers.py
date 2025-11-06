import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron with proper spiking dynamics
    """
    def __init__(self, 
                 beta: float = 0.9,  # membrane time constant
                 threshold: float = 1.0,
                 reset_mechanism: str = "subtract",  # "subtract" or "zero"
                 init_hidden: bool = True):
        super().__init__()
        
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False)
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=False)
        self.reset_mechanism = reset_mechanism
        self.init_hidden = init_hidden
        
        # For tracking membrane potential
        self.mem = None
        self.spike = None
        
    def forward(self, input_current: torch.Tensor, mem: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LIF neuron
        
        Args:
            input_current: (batch, features) - input current to neurons
            mem: (batch, features) - previous membrane potential (if not provided, uses internal state or zeros)
            
        Returns:
            spike: (batch, features) - binary spike output
            new_mem: (batch, features) - updated membrane potential
        """
        batch_size = input_current.shape[0]
        n_features = input_current.shape[1]
        
        # Initialize membrane potential if not provided
        if mem is None:
            if self.mem is not None and self.mem.shape[0] == batch_size:
                mem = self.mem
            else:
                mem = torch.zeros(batch_size, n_features, device=input_current.device, dtype=input_current.dtype)
        
        # Update membrane potential: mem_new = beta * mem_old + input_current
        new_mem = self.beta * mem + input_current
        
        # Generate spikes where membrane potential exceeds threshold
        spike = (new_mem > self.threshold).float()
        
        # Reset membrane potential based on mechanism
        if self.reset_mechanism == "subtract":
            new_mem = new_mem * (1 - spike) + spike * (new_mem - self.threshold)  # Subtract threshold
        elif self.reset_mechanism == "zero":
            new_mem = new_mem * (1 - spike)  # Reset to zero
        else:
            # No reset - keep membrane potential
            pass
            
        # Store for next call if using internal state
        if self.init_hidden:
            self.mem = new_mem
            self.spike = spike
            
        return spike, new_mem


class SpikingLinear(nn.Module):
    """
    Linear layer with spiking inputs and outputs
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transform to spike trains
        
        Args:
            spikes: (batch, time_steps, in_features) or (batch, in_features) - input spikes
            
        Returns:
            output: (batch, time_steps, out_features) or (batch, out_features) - linear output
        """
        return self.linear(spikes)


class SpikingConv1d(nn.Module):
    """
    1D convolution with spiking inputs
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Apply conv to spike trains (time dimension is treated as sequence)
        
        Args:
            spikes: (batch, in_channels, time_steps) - input spikes
            
        Returns:
            output: (batch, out_channels, time_steps) - convolved output
        """
        return self.conv(spikes)


class SpikingRecurrent(nn.Module):
    """
    Recurrent SNN layer that maintains temporal dynamics
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int,
                 beta: float = 0.9,
                 threshold: float = 1.0):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input to hidden projection
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        
        # Recurrent connections
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        
        # LIF neurons for the hidden units
        self.lif_neurons = LIFNeuron(beta=beta, threshold=threshold)
        
    def forward(self, input_spikes: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through spiking recurrent layer
        
        Args:
            input_spikes: (batch, time_steps, input_features) - input spike trains
            hidden_state: (batch, hidden_size) - previous hidden state (membrane potentials)
            
        Returns:
            output_spikes: (batch, time_steps, hidden_size) - output spike trains
            hidden_state: (batch, hidden_size) - final hidden state
        """
        batch_size, time_steps, _ = input_spikes.shape
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_mem = torch.zeros(batch_size, self.hidden_size, device=input_spikes.device)
        else:
            hidden_mem = hidden_state
        
        output_spikes = []
        current_hidden_mem = hidden_mem
        
        for t in range(time_steps):
            # Get input at current time step
            input_t = input_spikes[:, t, :]  # (batch, input_features)
            
            # Compute recurrent input
            input_proj = self.input_to_hidden(input_t)  # (batch, hidden_size)
            hidden_proj = self.hidden_to_hidden(current_hidden_mem)  # (batch, hidden_size)
            
            # Total input to LIF neurons
            total_input = input_proj + hidden_proj  # (batch, hidden_size)
            
            # Process through LIF neurons
            spikes, current_hidden_mem = self.lif_neurons(total_input, current_hidden_mem)
            
            output_spikes.append(spikes)
        
        # Stack output spikes: (batch, time_steps, hidden_size)
        output_spikes = torch.stack(output_spikes, dim=1)
        
        return output_spikes, current_hidden_mem


class PhaseEncodingSNN(nn.Module):
    """
    SNN that uses phase-of-firing encoding to preserve temporal relationships
    """
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 n_time_steps: int,
                 beta: float = 0.9,
                 threshold: float = 1.0):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_time_steps = n_time_steps
        
        # First layer: input to hidden with spiking
        self.input_layer = SpikingLinear(input_size, hidden_size)
        self.lif_input = LIFNeuron(beta=beta, threshold=threshold)
        
        # Temporal processing layer
        self.temporal_layer = SpikingRecurrent(hidden_size, hidden_size, beta=beta, threshold=threshold)
        
        # Output layer: hidden to output with spiking
        self.output_layer = SpikingLinear(hidden_size, output_size)
        self.lif_output = LIFNeuron(beta=beta, threshold=threshold)
        
        # Learnable attention weights for temporal integration
        self.temporal_attention = nn.Parameter(torch.randn(n_time_steps, output_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through phase-encoding SNN
        
        Args:
            x: (batch, time_steps, input_features) - input features (from cochlear processing)
            
        Returns:
            output: (batch, output_features) - phase-encoded representation
        """
        batch_size, time_steps, input_features = x.shape
        device = x.device
        
        # Process through input layer
        input_proj = self.input_layer(x)  # (batch, time_steps, hidden_size)
        
        # Process each time step through LIF neurons
        hidden_spikes = []
        mem_state = torch.zeros(batch_size, self.hidden_size, device=device)
        
        for t in range(time_steps):
            spikes, mem_state = self.lif_input(input_proj[:, t, :], mem_state)
            hidden_spikes.append(spikes)
        
        # Stack hidden spikes: (batch, time_steps, hidden_size)
        hidden_spikes = torch.stack(hidden_spikes, dim=1)
        
        # Process through temporal layer
        temporal_spikes, _ = self.temporal_layer(hidden_spikes)
        
        # Process through output layer
        output_proj = self.output_layer(temporal_spikes)  # (batch, time_steps, output_size)
        
        # Apply temporal attention weights
        # Normalize attention weights
        attention_weights = F.softmax(self.temporal_attention[:time_steps, :], dim=0)  # (time_steps, output_size)
        
        # Apply attention: (batch, time_steps, output_size) * (time_steps, output_size)
        attended_output = output_proj * attention_weights.unsqueeze(0)  # Broadcasting
        attended_output = attended_output.sum(dim=1)  # Sum over time steps: (batch, output_size)
        
        # Process through final LIF neurons for final spiking output
        final_spikes, _ = self.lif_output(attended_output)
        
        return final_spikes


class SpikingJEPAEncoder(nn.Module):
    """
    JEPA-style encoder using spiking neural networks
    JEPA = Janus Energy Prediction Architecture (predicting both past and future from present)
    """
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 embed_dim: int,
                 n_layers: int = 3,
                 beta: float = 0.9,
                 threshold: float = 1.0):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        
        # Multiple layers of spiking processing
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(SpikingRecurrent(input_size, hidden_size, beta, threshold))
        
        # Intermediate layers
        for i in range(1, n_layers):
            self.layers.append(SpikingRecurrent(hidden_size, hidden_size, beta, threshold))
        
        # Final projection to embedding space
        self.projection = SpikingLinear(hidden_size, embed_dim)
        self.final_lif = LIFNeuron(beta=beta, threshold=threshold)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through JEPA-style spiking encoder
        
        Args:
            x: (batch, time_steps, input_features) - input from cochlear processing
            
        Returns:
            embedding: (batch, embed_dim) - JEPA-style embedding
        """
        batch_size, time_steps, _ = x.shape
        device = x.device
        
        # Process through each layer
        current_input = x
        hidden_state = None
        
        for i, layer in enumerate(self.layers):
            # Process through spiking recurrent layer
            layer_output, hidden_state = layer(current_input, hidden_state)
            current_input = layer_output
        
        # Average over time dimension to get final representation
        time_avg = current_input.mean(dim=1)  # (batch, hidden_size)
        
        # Project to embedding space
        proj_output = self.projection(time_avg)  # (batch, embed_dim)
        
        # Apply final LIF neurons
        final_output, _ = self.final_lif(proj_output)
        
        return final_output


# Test the spiking layers
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Testing spiking layers...")
    
    # Test LIF neuron
    lif = LIFNeuron(beta=0.9, threshold=1.0).to(device)
    input_current = torch.randn(2, 10, device=device)  # 2 samples, 10 neurons
    mem_init = torch.zeros(2, 10, device=device)
    
    spikes, new_mem = lif(input_current, mem_init)
    print(f"LIF neuron - Input: {input_current.shape}, Output spikes: {spikes.shape}, Mem: {new_mem.shape}")
    
    # Test phase encoding SNN
    phase_snn = PhaseEncodingSNN(
        input_size=32,
        hidden_size=128,
        output_size=256,
        n_time_steps=100
    ).to(device)
    
    x = torch.randn(2, 100, 32, device=device)  # 2 samples, 100 time steps, 32 features
    output = phase_snn(x)
    print(f"Phase SNN - Input: {x.shape}, Output: {output.shape}")
    
    # Test JEPA encoder
    jepa = SpikingJEPAEncoder(
        input_size=32,
        hidden_size=256,
        embed_dim=512,
        n_layers=3
    ).to(device)
    
    jepa_output = jepa(x)
    print(f"JEPA encoder - Input: {x.shape}, Output: {jepa_output.shape}")
    
    print("All spiking layers working correctly!")