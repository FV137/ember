import torch
import torch.nn as nn
from typing import Dict, Callable, Any
from models.l_module import LModule
from configs.l_module_config import LModuleConfig


def ensure_multiples_of_heads(embed_dim: int, min_heads: int = 4) -> tuple[int, int]:
    """
    Ensure embed_dim is divisible by number of attention heads
    Returns (adjusted_embed_dim, num_heads)
    """
    # Find the number of heads that divides embed_dim evenly, starting from min_heads
    num_heads = min_heads
    
    # Increase embed_dim until it's divisible by num_heads
    while embed_dim % num_heads != 0:
        # Try next even number for num_heads up to 16
        while num_heads <= 16 and embed_dim % num_heads != 0:
            num_heads += 1
        if embed_dim % num_heads != 0:
            # If none of 4-16 work, adjust embed_dim to be divisible by 4
            embed_dim = ((embed_dim // 4) + 1) * 4
            num_heads = 4
    
    return embed_dim, num_heads


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model_with_target_params(target_params: int = 100_000_000, 
                                   tolerance: float = 0.05) -> tuple[LModule, LModuleConfig]:
    """
    Create an LModule with parameters as close as possible to the target
    """
    print(f"Creating L-Module with target parameters: {target_params:,}")
    
    # Start with Phase 1 config as base
    base_embed_dim = 1024  # Start value
    base_embed_dim, num_heads = ensure_multiples_of_heads(base_embed_dim)
    
    config = LModuleConfig(
        n_filters=384,
        embed_dim=base_embed_dim,
        snn_hidden_size=1536,
        snn_output_size=1536,
        n_snn_layers=8,
        n_jepa_layers=8,
        jepa_hidden_size=1536,
        predictor_hidden_dim=2048,
        predictor_layers=6,
        target_params=target_params
    )
    
    def calculate_params(cfg: LModuleConfig) -> int:
        # Create a temporary model to count parameters
        temp_model = LModule(cfg)
        params = count_parameters(temp_model)
        del temp_model  # Free memory
        return params
    
    # Get current parameter count
    current_params = calculate_params(config)
    print(f"Initial parameters: {current_params:,} ({current_params/target_params:.2%} of target)")
    
    # Adjust parameters to get closer to target
    # First try adjusting the largest contributing components
    attempts = 0
    max_attempts = 20
    
    while abs(current_params - target_params) > target_params * tolerance and attempts < max_attempts:
        param_ratio = target_params / current_params
        
        # Adjust the largest parameter contributors
        if param_ratio > 1.1:  # Need more parameters
            config.snn_hidden_size = int(config.snn_hidden_size * 1.1)
            config.jepa_hidden_size = int(config.jepa_hidden_size * 1.1)
            config.predictor_hidden_dim = int(config.predictor_hidden_dim * 1.1)
            new_embed_dim = min(2048, int(config.embed_dim * 1.05))  # Cap embed_dim
            new_embed_dim, _ = ensure_multiples_of_heads(new_embed_dim)
            config.embed_dim = new_embed_dim
        elif param_ratio < 0.9:  # Need fewer parameters
            config.snn_hidden_size = int(config.snn_hidden_size * 0.95)
            config.jepa_hidden_size = int(config.jepa_hidden_size * 0.95)
            config.predictor_hidden_dim = int(config.predictor_hidden_dim * 0.95)
            new_embed_dim = max(768, int(config.embed_dim * 0.95))  # Min embed_dim
            new_embed_dim, _ = ensure_multiples_of_heads(new_embed_dim)
            config.embed_dim = new_embed_dim
        else:
            # Fine-tune with smaller adjustments
            adjustment = 1 + (target_params - current_params) / target_params / 3
            config.snn_hidden_size = int(config.snn_hidden_size * adjustment)
            config.jepa_hidden_size = int(config.jepa_hidden_size * adjustment)
            config.predictor_hidden_dim = int(config.predictor_hidden_dim * adjustment)
            new_embed_dim = int(config.embed_dim * adjustment)
            new_embed_dim, _ = ensure_multiples_of_heads(new_embed_dim)
            config.embed_dim = new_embed_dim
        
        # Ensure parameters are reasonable
        config.snn_hidden_size = max(256, min(3072, config.snn_hidden_size))
        config.jepa_hidden_size = max(256, min(3072, config.jepa_hidden_size))
        config.predictor_hidden_dim = max(512, min(4096, config.predictor_hidden_dim))
        new_embed_dim, _ = ensure_multiples_of_heads(config.embed_dim, 4)
        config.embed_dim = new_embed_dim
        
        current_params = calculate_params(config)
        print(f"Attempt {attempts+1}: {current_params:,} parameters ({current_params/target_params:.2%} of target)")
        attempts += 1
    
    # Final model creation
    final_model = LModule(config)
    final_params = count_parameters(final_model)
    
    print(f"Final parameters: {final_params:,} ({final_params/target_params:.2%} of target)")
    print(f"Diff from target: {final_params - target_params:,}")
    
    return final_model, config


def analyze_parameter_breakdown(model: LModule) -> Dict[str, int]:
    """
    Analyze where parameters are allocated in the model
    """
    breakdown = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
            if param_count > 0:
                breakdown[name] = param_count
    
    # Summarize by major components
    summary = {
        'cochlear_processing': 0,
        'snn_components': 0,
        'jepa_components': 0,
        'attention_components': 0,
        'prediction_components': 0,
        'other_components': 0
    }
    
    for name, count in breakdown.items():
        if 'cochlear' in name.lower() or 'filterbank' in name.lower():
            summary['cochlear_processing'] += count
        elif 'snn' in name.lower() or 'lif' in name.lower():
            summary['snn_components'] += count
        elif 'jepa' in name.lower() or 'predictive' in name.lower():
            summary['jepa_components'] += count
        elif 'attn' in name.lower() or 'attention' in name.lower():
            summary['attention_components'] += count
        elif 'pred' in name.lower() or 'predictor' in name.lower():
            summary['prediction_components'] += count
        else:
            summary['other_components'] += count
    
    # Clean up zero entries
    summary = {k: v for k, v in summary.items() if v > 0}
    
    return summary


def optimize_for_compute_efficiency(original_config: LModuleConfig) -> LModuleConfig:
    """
    Optimize configuration for compute efficiency based on available hardware
    """
    # For laptop optimization, we might want to reduce some parameters
    # but maintain the same overall architecture
    optimized_config = LModuleConfig(
        n_filters=original_config.n_filters,
        embed_dim=original_config.embed_dim,
        snn_hidden_size=original_config.snn_hidden_size,
        snn_output_size=original_config.snn_output_size,
        n_snn_layers=original_config.n_snn_layers,
        n_jepa_layers=original_config.n_jepa_layers,
        jepa_hidden_size=original_config.jepa_hidden_size,
        predictor_hidden_dim=original_config.predictor_hidden_dim,
        predictor_layers=original_config.predictor_layers,
        target_params=original_config.target_params,
        
        # Additional efficiency parameters
        dropout=0.1,  # Keep dropout for regularization
        target_time_steps=150,  # Reduce time steps to reduce memory
        mask_ratio=0.15  # Standard masking ratio
    )
    
    return optimized_config


if __name__ == "__main__":
    # Test the parameter optimizer
    print("Testing parameter optimizer...")
    
    # Create model with target 100M parameters
    model, config = create_model_with_target_params(target_params=100_000_000, tolerance=0.05)
    
    # Analyze parameter breakdown
    breakdown = analyze_parameter_breakdown(model)
    print("\nParameter breakdown by major components:")
    for component, count in breakdown.items():
        print(f"  {component}: {count:,} ({count/model._parameter_count:.1%})")
    
    print(f"\nTotal parameters: {model._parameter_count:,}")
    
    # Test efficiency optimization
    efficient_config = optimize_for_compute_efficiency(config)
    print(f"\nOptimized config parameters: {count_parameters(LModule(efficient_config)):,}")
    
    print("Parameter optimizer working correctly!")