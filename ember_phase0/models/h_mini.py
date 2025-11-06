import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Tuple


class SensoryFusionLayer(nn.Module):
    """Fusion layer that combines sensory embeddings with text embeddings"""
    def __init__(self, sensory_dim: int, text_dim: int, output_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.sensory_dim = sensory_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # Linear projections for multi-head attention
        self.sensory_proj = nn.Linear(sensory_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)
        
        # Multi-head attention for fusion
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Layer norm and feed-forward
        self.layer_norm = nn.LayerNorm(output_dim)
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 4, output_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, sensory_embed: torch.Tensor, text_embed: torch.Tensor):
        """
        Fuse sensory and text embeddings
        
        Args:
            sensory_embed: (batch, sensory_dim)
            text_embed: (batch, text_dim)
            
        Returns:
            fused_embed: (batch, output_dim)
        """
        # Project both embeddings to same dimensionality
        sensory_proj = self.sensory_proj(sensory_embed).unsqueeze(1)  # (batch, 1, output_dim)
        text_proj = self.text_proj(text_embed).unsqueeze(1)  # (batch, 1, output_dim)
        
        # Concatenate: (batch, 2, output_dim)
        combined = torch.cat([sensory_proj, text_proj], dim=1)
        
        # Apply multi-head attention for fusion
        fused, attn_weights = self.fusion_attn(
            query=combined,
            key=combined,
            value=combined
        )
        
        # Take mean of fused representations
        fused_mean = fused.mean(dim=1)  # (batch, output_dim)
        
        # Apply layer norm and FFN
        fused_norm = self.layer_norm(fused_mean)
        output = fused_norm + self.ffn(fused_norm)
        
        return output


class HTransformer(nn.Module):
    """Transformer module for H-mini with customizable parameters"""
    def __init__(self, 
                 embed_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 ff_dim: int = 2048,
                 dropout: float = 0.1,
                 num_classes: int = 2):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_classes = num_classes
        
        # Input layer norm
        self.input_norm = nn.LayerNorm(embed_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialize parameters
        self._init_weights()
    
    def _init_weights(self):
        """Initialize transformer weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through transformer
        
        Args:
            x: (batch, seq_len, embed_dim) or (batch, embed_dim) for single embedding
            
        Returns:
            logits: (batch, num_classes)
        """
        # Handle both single embeddings and sequences
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, embed_dim)
        
        # Normalize input
        x = self.input_norm(x)
        
        # Pass through transformer
        output = self.transformer(x)  # (batch, seq_len, embed_dim)
        
        # Take mean over sequence dimension
        pooled = output.mean(dim=1)  # (batch, embed_dim)
        
        # Apply classification head
        logits = self.classifier(pooled)  # (batch, num_classes)
        
        return logits


class HMini(nn.Module):
    """H-mini: Transformer-based reasoning module with sensory fusion (100M params)"""
    def __init__(self,
                 sensory_dim: int = 512,
                 text_dim: int = 128,
                 embed_dim: int = 512,
                 num_classes: int = 2,
                 use_sensory: bool = True):
        super().__init__()
        
        self.sensory_dim = sensory_dim
        self.text_dim = text_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_sensory = use_sensory
        
        # Tokenizer and text encoder (using a small pre-trained model)
        # We'll use a lightweight model for text processing
        self.tokenizer = None  # Will be initialized in setup
        self.text_encoder = nn.Linear(text_dim, text_dim)  # Placeholder for text encoding
        
        # Sensory fusion layer
        self.sensory_fusion = SensoryFusionLayer(
            sensory_dim=sensory_dim,
            text_dim=text_dim,
            output_dim=embed_dim
        )
        
        # Main transformer
        self.transformer = HTransformer(
            embed_dim=embed_dim,
            num_layers=6,  # Adjusted for ~100M params
            num_heads=8,
            ff_dim=2048,
            num_classes=num_classes
        )
        
        # Alternative path for text-only processing (control condition)
        self.text_only_head = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(0.1)
        )
        
        # Initialize parameters to achieve ~100M total
        self._init_weights()
        
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
    
    def setup_text_encoder(self, model_name: str = "distilbert-base-uncased"):
        """Setup text encoder with pre-trained model"""
        from transformers import AutoTokenizer, AutoModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_model = AutoModel.from_pretrained(model_name)
        
        # Adapt the text model to our needs
        self.text_encoder = nn.Sequential(
            text_model,
            nn.Linear(text_model.config.hidden_size, self.text_dim),
            nn.Dropout(0.1)
        )
        
    def process_text(self, text_input: torch.Tensor) -> torch.Tensor:
        """
        Process text input to embedding
        For simplicity, we'll use a placeholder here
        """
        # In a real implementation, this would use the tokenizer and encoder
        # For now, we'll assume text_input is already a processed embedding
        return text_input
    
    def forward(self, 
                sensory_embed: Optional[torch.Tensor] = None,
                text_embed: Optional[torch.Tensor] = None,
                use_sensory: Optional[bool] = None) -> torch.Tensor:
        """
        Forward pass combining sensory and text information
        
        Args:
            sensory_embed: (batch, sensory_dim) - from L-mini
            text_embed: (batch, text_dim) - text representation
            use_sensory: Override self.use_sensory flag for control experiments
            
        Returns:
            logits: (batch, num_classes) - classification logits
        """
        if use_sensory is None:
            use_sensory = self.use_sensory
        
        if use_sensory and sensory_embed is not None:
            # Use sensory fusion
            if text_embed is None:
                # If no text provided, use a learned embedding
                text_embed = torch.zeros(sensory_embed.size(0), self.text_dim, 
                                       device=sensory_embed.device, dtype=sensory_embed.dtype)
            
            # Fuse sensory and text embeddings
            fused_embed = self.sensory_fusion(sensory_embed, text_embed)
            
            # Process through transformer
            logits = self.transformer(fused_embed)
        else:
            # Control condition: use text embedding only (possibly placeholder)
            if text_embed is None:
                # Create a learned embedding for classification-only mode
                text_embed = torch.zeros(sensory_embed.size(0), self.text_dim,
                                       device=sensory_embed.device, dtype=sensory_embed.dtype)
            
            # Project text to embedding space
            text_processed = self.text_only_head(text_embed)
            
            # Process through transformer
            logits = self.transformer(text_processed)
        
        return logits
    
    def get_parameter_count(self):
        """Get total number of parameters"""
        total = sum(p.numel() for p in self.parameters())
        return total


# Example usage and testing
if __name__ == "__main__":
    # Test the H-mini model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    h_mini = HMini(
        sensory_dim=512,
        text_dim=128,
        embed_dim=512,
        num_classes=2,
        use_sensory=True
    )
    h_mini.to(device)
    
    # Generate dummy embeddings
    dummy_sensory = torch.randn(2, 512, device=device)
    dummy_text = torch.randn(2, 128, device=device)
    
    # Forward pass with both sensory and text
    logits_both = h_mini(sensory_embed=dummy_sensory, text_embed=dummy_text)
    print(f"Logits with sensory+text: {logits_both.shape}")
    
    # Forward pass with text only (control condition)
    logits_text_only = h_mini(sensory_embed=dummy_sensory, text_embed=dummy_text, use_sensory=False)
    print(f"Logits with text only: {logits_text_only.shape}")
    
    # Count parameters
    total_params = h_mini.get_parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Target: ~100M, Actual: {total_params/1_000_000:.1f}M")