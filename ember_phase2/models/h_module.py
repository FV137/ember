import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from transformers import AutoModel, AutoConfig
from einops import rearrange, repeat


class SensoryFusionLayer(nn.Module):
    """
    Fusion layer that combines sensory embeddings with text/context embeddings
    This is the critical binding mechanism between L and H modules
    """
    def __init__(self, 
                 sensory_dim: int = 1024,
                 text_dim: int = 768, 
                 embed_dim: int = 1024,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.sensory_dim = sensory_dim
        self.text_dim = text_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Projection layers to unify dimensions
        self.sensory_proj = nn.Linear(sensory_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        
        # Cross-attention for sensory-text fusion
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization and feed-forward
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, 
                sensory_embed: torch.Tensor, 
                text_embed: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fuse sensory and text embeddings
        
        Args:
            sensory_embed: (batch, sensory_dim) - from L-module
            text_embed: (batch, seq_len, text_dim) - from text encoder
            attention_mask: (batch, seq_len) - optional attention mask for text
            
        Returns:
            fused_embed: (batch, seq_len, embed_dim) - fused representations
        """
        batch_size = sensory_embed.size(0)
        
        # Project embeddings to unified dimension
        sensory_proj = self.sensory_proj(sensory_embed)  # (batch, embed_dim)
        text_proj = self.text_proj(text_embed)  # (batch, seq_len, embed_dim)
        
        # Add sensory information to each text position
        sensory_expanded = repeat(sensory_proj, 'b d -> b s d', s=text_proj.size(1))
        combined = text_proj + sensory_expanded  # (batch, seq_len, embed_dim)
        
        # Apply cross-attention for fusion
        fused, attn_weights = self.fusion_attn(
            query=combined,
            key=combined,
            value=combined,
            key_padding_mask=attention_mask
        )
        
        # Apply normalization and FFN
        fused = self.norm1(fused + combined)
        fused = self.norm2(fused + self.ffn(fused))
        
        return fused


class ReasoningTransformer(nn.Module):
    """
    Transformer-based reasoning module that works with fused sensory representations
    """
    def __init__(self,
                 embed_dim: int = 1024,
                 num_layers: int = 12,
                 num_heads: int = 16,
                 ff_dim: int = 4096,
                 dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Positional embeddings
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Output normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, 
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through reasoning transformer
        
        Args:
            x: (batch, seq_len, embed_dim) - fused sensory-text embeddings
            attention_mask: (batch, seq_len) - optional attention mask
            
        Returns:
            output: (batch, seq_len, embed_dim) - reasoned representations
        """
        batch_size, seq_len, _ = x.shape
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0)  # (1, seq_len, embed_dim)
        x = x + pos_enc  # (batch, seq_len, embed_dim)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to the format expected by PyTorch transformer
            src_key_padding_mask = ~attention_mask.bool()  # Invert for padding mask
        else:
            src_key_padding_mask = None
        
        # Apply transformer
        output = self.transformer(
            src=x,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Apply final normalization
        output = self.norm(output)
        
        return output


class HModule(nn.Module):
    """
    H-Module: High-level reasoning with sensory fusion
    The thinking and speaking component that works with L-module sensations
    """
    def __init__(self,
                 sensory_dim: int = 1024,
                 text_model_name: str = "distilbert-base-uncased",
                 embed_dim: int = 1024,
                 num_reasoning_layers: int = 12,
                 num_reasoning_heads: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        
        self.sensory_dim = sensory_dim
        self.embed_dim = embed_dim
        
        # Text encoder (frozen or fine-tuned)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_dim = self.text_encoder.config.hidden_size
        
        # Freeze text encoder for stability (optional)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Sensory fusion layer
        self.sensory_fusion = SensoryFusionLayer(
            sensory_dim=sensory_dim,
            text_dim=self.text_dim,
            embed_dim=embed_dim,
            num_heads=max(4, embed_dim // 128),
            dropout=dropout
        )
        
        # Reasoning transformer
        self.reasoning_transformer = ReasoningTransformer(
            embed_dim=embed_dim,
            num_layers=num_reasoning_layers,
            num_heads=num_reasoning_heads,
            ff_dim=embed_dim * 4,
            dropout=dropout
        )
        
        # Task-specific heads
        self.language_modeling_head = nn.Linear(embed_dim, self.text_encoder.config.vocab_size)
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2)  # Binary classification for now
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with appropriate scaling"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self,
                sensory_embed: torch.Tensor,
                text_input_ids: torch.Tensor,
                text_attention_mask: Optional[torch.Tensor] = None,
                task: str = "language_modeling") -> Dict[str, torch.Tensor]:
        """
        Forward pass through H-module
        
        Args:
            sensory_embed: (batch, sensory_dim) - embeddings from L-module
            text_input_ids: (batch, seq_len) - tokenized text input
            text_attention_mask: (batch, seq_len) - attention mask for text
            task: str - task type ("language_modeling", "classification", etc.)
            
        Returns:
            output: Dict containing task-specific outputs
        """
        batch_size, seq_len = text_input_ids.shape
        
        # Encode text
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        text_embed = text_outputs.last_hidden_state  # (batch, seq_len, text_dim)
        
        # Fuse sensory and text embeddings
        fused_embed = self.sensory_fusion(
            sensory_embed=sensory_embed,
            text_embed=text_embed,
            attention_mask=text_attention_mask
        )  # (batch, seq_len, embed_dim)
        
        # Apply reasoning
        reasoned_embed = self.reasoning_transformer(
            x=fused_embed,
            attention_mask=text_attention_mask
        )  # (batch, seq_len, embed_dim)
        
        # Task-specific outputs
        outputs = {}
        
        if task == "language_modeling":
            # Language modeling head
            lm_logits = self.language_modeling_head(reasoned_embed)
            outputs['logits'] = lm_logits
            
        elif task == "classification":
            # Classification: pool and classify
            # Use attention-weighted average pooling
            if text_attention_mask is not None:
                # Mask out padding tokens
                masked_embed = reasoned_embed * text_attention_mask.unsqueeze(-1)
                pooled = masked_embed.sum(dim=1) / text_attention_mask.sum(dim=1, keepdim=True)
            else:
                pooled = reasoned_embed.mean(dim=1)
                
            class_logits = self.classification_head(pooled)
            outputs['logits'] = class_logits
            
        # Always return the embeddings for potential downstream use
        outputs['embeddings'] = reasoned_embed
        outputs['pooled_embedding'] = reasoned_embed.mean(dim=1)  # Simple pooling
        
        return outputs
    
    def generate_response(self,
                         sensory_embed: torch.Tensor,
                         prompt_input_ids: torch.Tensor,
                         prompt_attention_mask: torch.Tensor,
                         max_length: int = 50,
                         temperature: float = 1.0,
                         top_k: int = 50) -> torch.Tensor:
        """
        Generate text response conditioned on sensory input
        """
        # This would be a simplified generation method
        # In practice, you'd use beam search or sampling
        
        generated_tokens = []
        current_input_ids = prompt_input_ids.clone()
        current_attention_mask = prompt_attention_mask.clone()
        
        for _ in range(max_length):
            # Forward pass
            outputs = self.forward(
                sensory_embed=sensory_embed,
                text_input_ids=current_input_ids,
                text_attention_mask=current_attention_mask,
                task="language_modeling"
            )
            
            # Get logits for last token
            logits = outputs['logits'][:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Append to generated tokens
            generated_tokens.append(next_token)
            
            # Update input for next iteration
            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
            current_attention_mask = torch.cat([
                current_attention_mask, 
                torch.ones_like(next_token)
            ], dim=1)
        
        return torch.cat(generated_tokens, dim=1)


# Test the H-Module
if __name__ == "__main__":
    # Create H-Module
    h_module = HModule(
        sensory_dim=1024,
        text_model_name="distilbert-base-uncased",
        embed_dim=1024,
        num_reasoning_layers=6,  # Smaller for testing
        num_reasoning_heads=8
    )
    
    print(f"H-Module created with {sum(p.numel() for p in h_module.parameters()):,} parameters")
    
    # Test with dummy inputs
    batch_size = 2
    seq_len = 20
    
    # Dummy sensory embedding from L-module
    sensory_embed = torch.randn(batch_size, 1024)
    
    # Dummy text inputs
    text_input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    text_attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"Input shapes:")
    print(f"  Sensory embedding: {sensory_embed.shape}")
    print(f"  Text input IDs: {text_input_ids.shape}")
    print(f"  Attention mask: {text_attention_mask.shape}")
    
    # Test language modeling
    lm_outputs = h_module(
        sensory_embed=sensory_embed,
        text_input_ids=text_input_ids,
        text_attention_mask=text_attention_mask,
        task="language_modeling"
    )
    
    print(f"\nLanguage modeling outputs:")
    print(f"  Logits: {lm_outputs['logits'].shape}")
    print(f"  Embeddings: {lm_outputs['embeddings'].shape}")
    print(f"  Pooled embedding: {lm_outputs['pooled_embedding'].shape}")
    
    # Test classification
    class_outputs = h_module(
        sensory_embed=sensory_embed,
        text_input_ids=text_input_ids,
        text_attention_mask=text_attention_mask,
        task="classification"
    )
    
    print(f"\nClassification outputs:")
    print(f"  Logits: {class_outputs['logits'].shape}")
    
    print("\nH-Module working correctly!")