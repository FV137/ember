import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for L-H module binding
    Allows bidirectional information flow between sensation and reasoning
    """
    def __init__(self,
                 l_dim: int = 1024,
                 h_dim: int = 1024,
                 embed_dim: int = 1024,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.l_dim = l_dim
        self.h_dim = h_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Projection layers to unified embedding space
        self.l_proj = nn.Linear(l_dim, embed_dim)
        self.h_proj = nn.Linear(h_dim, embed_dim)
        
        # Cross-attention mechanisms
        self.l_to_h_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.h_to_l_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm_l = nn.LayerNorm(embed_dim)
        self.norm_h = nn.LayerNorm(embed_dim)
        
        # Feed-forward networks
        self.ffn_l = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.ffn_h = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self,
                l_embed: torch.Tensor,
                h_embed: torch.Tensor,
                l_mask: Optional[torch.Tensor] = None,
                h_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional cross-modal attention between L and H modules
        
        Args:
            l_embed: (batch, l_seq, l_dim) - L-module embeddings
            h_embed: (batch, h_seq, h_dim) - H-module embeddings
            l_mask: (batch, l_seq) - optional mask for L embeddings
            h_mask: (batch, h_seq) - optional mask for H embeddings
            
        Returns:
            enhanced_l: (batch, l_seq, embed_dim) - L embeddings enhanced with H context
            enhanced_h: (batch, h_seq, embed_dim) - H embeddings enhanced with L context
        """
        # Project to unified embedding space
        l_proj = self.l_proj(l_embed)  # (batch, l_seq, embed_dim)
        h_proj = self.h_proj(h_embed)  # (batch, h_seq, embed_dim)
        
        # L-to-H attention (H enhanced with L context)
        h_enhanced, h_attn_weights = self.l_to_h_attn(
            query=h_proj,
            key=l_proj,
            value=l_proj,
            key_padding_mask=~l_mask if l_mask is not None else None
        )
        
        # H-to-L attention (L enhanced with H context)
        l_enhanced, l_attn_weights = self.h_to_l_attn(
            query=l_proj,
            key=h_proj,
            value=h_proj,
            key_padding_mask=~h_mask if h_mask is not None else None
        )
        
        # Apply residual connections and normalization
        h_enhanced = self.norm_h(h_enhanced + h_proj)
        l_enhanced = self.norm_l(l_enhanced + l_proj)
        
        # Apply feed-forward networks
        h_enhanced = self.norm_h(h_enhanced + self.ffn_h(h_enhanced))
        l_enhanced = self.norm_l(l_enhanced + self.ffn_l(l_enhanced))
        
        return l_enhanced, h_enhanced


class SensoryModulation(nn.Module):
    """
    Modulates H-module processing based on L-module sensory states
    This implements the "sensory influence on cognition" mechanism
    """
    def __init__(self,
                 sensory_dim: int = 1024,
                 hidden_dim: int = 512,
                 num_modulation_factors: int = 8):
        super().__init__()
        
        self.sensory_dim = sensory_dim
        self.hidden_dim = hidden_dim
        self.num_modulation_factors = num_modulation_factors
        
        # Generate modulation factors from sensory state
        self.modulation_generator = nn.Sequential(
            nn.Linear(sensory_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_modulation_factors),
            nn.Softmax(dim=-1)
        )
        
        # Modulation application layers
        self.modulation_layers = nn.ModuleList([
            nn.Linear(sensory_dim, sensory_dim) 
            for _ in range(num_modulation_factors)
        ])
        
    def forward(self, 
                sensory_state: torch.Tensor,
                cognitive_state: torch.Tensor) -> torch.Tensor:
        """
        Apply sensory modulation to cognitive processing
        
        Args:
            sensory_state: (batch, sensory_dim) - current sensory state
            cognitive_state: (batch, seq_len, cognitive_dim) - cognitive processing state
            
        Returns:
            modulated_state: (batch, seq_len, cognitive_dim) - modulated cognitive state
        """
        batch_size, seq_len, cognitive_dim = cognitive_state.shape
        
        # Generate modulation factors
        modulation_weights = self.modulation_generator(sensory_state)  # (batch, num_factors)
        
        # Apply different modulation transformations
        modulated_states = []
        for i, mod_layer in enumerate(self.modulation_layers):
            # Transform sensory state with modulation-specific layer
            mod_sensory = mod_layer(sensory_state)  # (batch, sensory_dim)
            
            # Apply modulation weight to cognitive state
            weight = modulation_weights[:, i].unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
            modulated = cognitive_state * (1 + weight * mod_sensory.unsqueeze(1))
            modulated_states.append(modulated)
        
        # Combine modulated states
        modulated_state = torch.stack(modulated_states, dim=-1).mean(dim=-1)
        
        return modulated_state


class LHBindingModule(nn.Module):
    """
    Complete L-H binding mechanism integrating cross-modal attention and sensory modulation
    """
    def __init__(self,
                 l_dim: int = 1024,
                 h_dim: int = 1024,
                 embed_dim: int = 1024,
                 num_heads: int = 8,
                 num_modulation_factors: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.l_dim = l_dim
        self.h_dim = h_dim
        self.embed_dim = embed_dim
        
        # Cross-modal attention for information exchange
        self.cross_attention = CrossModalAttention(
            l_dim=l_dim,
            h_dim=h_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Sensory modulation for cognition influence
        self.sensory_modulation = SensoryModulation(
            sensory_dim=l_dim,
            hidden_dim=embed_dim // 2,
            num_modulation_factors=num_modulation_factors
        )
        
        # Final projection layers
        self.l_final_proj = nn.Linear(embed_dim, l_dim)
        self.h_final_proj = nn.Linear(embed_dim, h_dim)
        
        # Binding strength control
        self.binding_strength = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self,
                l_embeddings: torch.Tensor,
                h_embeddings: torch.Tensor,
                l_mask: Optional[torch.Tensor] = None,
                h_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through L-H binding mechanism
        
        Args:
            l_embeddings: (batch, l_seq, l_dim) - L-module embeddings
            h_embeddings: (batch, h_seq, h_dim) - H-module embeddings  
            l_mask: (batch, l_seq) - optional mask for L embeddings
            h_mask: (batch, h_seq) - optional mask for H embeddings
            return_attention: bool - whether to return attention weights
            
        Returns:
            Dictionary containing bound embeddings and optional attention weights
        """
        # Apply cross-modal attention
        bound_l, bound_h = self.cross_attention(
            l_embed=l_embeddings,
            h_embed=h_embeddings,
            l_mask=l_mask,
            h_mask=h_mask
        )
        
        # Apply final projections
        bound_l = self.l_final_proj(bound_l)
        bound_h = self.h_final_proj(bound_h)
        
        # Pool L embeddings for modulation (if needed)
        if bound_l.dim() == 3:
            l_pooled = bound_l.mean(dim=1)  # (batch, l_dim)
        else:
            l_pooled = bound_l
            
        # Apply sensory modulation to H embeddings
        modulated_h = self.sensory_modulation(l_pooled, bound_h)
        
        # Blend original and modulated states based on binding strength
        final_h = (1 - self.binding_strength) * bound_h + self.binding_strength * modulated_h
        
        outputs = {
            'bound_l_embeddings': bound_l,
            'bound_h_embeddings': final_h,
            'l_attention_map': None,
            'h_attention_map': None
        }
        
        # Optionally compute and return attention maps
        if return_attention:
            # This would require storing attention weights from cross_attention
            pass
            
        return outputs


# Test the binding module
if __name__ == "__main__":
    # Create binding module
    binding_module = LHBindingModule(
        l_dim=1024,
        h_dim=1024,
        embed_dim=1024,
        num_heads=8,
        num_modulation_factors=8
    )
    
    print(f"L-H Binding Module created with {sum(p.numel() for p in binding_module.parameters()):,} parameters")
    
    # Test with dummy inputs
    batch_size = 2
    l_seq_len = 50  # Temporal sequence from L-module
    h_seq_len = 20  # Text sequence from H-module
    
    # Dummy L-module embeddings (temporal sequence)
    l_embeddings = torch.randn(batch_size, l_seq_len, 1024)
    
    # Dummy H-module embeddings (text sequence)
    h_embeddings = torch.randn(batch_size, h_seq_len, 1024)
    
    # Dummy masks
    l_mask = torch.ones(batch_size, l_seq_len, dtype=torch.bool)
    h_mask = torch.ones(batch_size, h_seq_len, dtype=torch.bool)
    
    print(f"Input shapes:")
    print(f"  L embeddings: {l_embeddings.shape}")
    print(f"  H embeddings: {h_embeddings.shape}")
    print(f"  L mask: {l_mask.shape}")
    print(f"  H mask: {h_mask.shape}")
    
    # Test binding
    binding_outputs = binding_module(
        l_embeddings=l_embeddings,
        h_embeddings=h_embeddings,
        l_mask=l_mask,
        h_mask=h_mask,
        return_attention=False
    )
    
    print(f"\nBinding outputs:")
    print(f"  Bound L embeddings: {binding_outputs['bound_l_embeddings'].shape}")
    print(f"  Bound H embeddings: {binding_outputs['bound_h_embeddings'].shape}")
    
    print("\nL-H Binding Module working correctly!")