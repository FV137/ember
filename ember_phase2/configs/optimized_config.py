from .l_module_config import LModuleConfig


def get_100m_optimized_config() -> LModuleConfig:
    """
    Configuration optimized for exactly ~100M parameters with attention compatibility
    Based on the successful parameter optimization run
    """
    embed_dim = 1072  # This ensures attention compatibility (divisible by calculated heads)
    
    return LModuleConfig(
        n_filters=416,  # Increased from 384 to add more parameters
        embed_dim=embed_dim,  # 1072 - attention compatible
        snn_hidden_size=1664,  # Increased to add parameters
        snn_output_size=1664,  # Keep consistent
        n_snn_layers=8,
        n_jepa_layers=8,
        jepa_hidden_size=1664,  # Increased to match
        predictor_hidden_dim=2208,  # Increased to add parameters
        predictor_layers=6,
        target_params=100_000_000,
        dropout=0.1,
        target_time_steps=150,
        mask_ratio=0.15,
        lambda_temporal=0.1,  # Reduced for efficiency
        lambda_energy=0.01   # Reduced for efficiency
    )


def get_laptop_optimized_config() -> LModuleConfig:
    """
    Configuration optimized for laptop hardware (RTX 4050) - smaller but still effective
    """
    embed_dim = 768  # Smaller but still attention compatible
    
    # Ensure embed_dim is compatible with attention mechanism
    # embed_dim // 128 = 768 // 128 = 6, so embed_dim % 6 = 0 -> True
    if embed_dim % max(4, embed_dim // 128) != 0:
        # Find closest that is compatible
        target_heads = max(4, embed_dim // 128)
        embed_dim = (embed_dim // target_heads) * target_heads
    
    return LModuleConfig(
        n_filters=256,  # Reduced for memory efficiency
        embed_dim=embed_dim,  # 768 for laptop
        snn_hidden_size=1024,  # Reduced
        snn_output_size=1024,
        n_snn_layers=6,  # Reduced layers
        n_jepa_layers=6,  # Reduced layers
        jepa_hidden_size=1024,
        predictor_hidden_dim=1536,  # Reduced
        predictor_layers=4,  # Reduced
        target_params=50_000_000,  # Target 50M for laptop testing first
        dropout=0.1,
        target_time_steps=100,  # Reduce time steps for memory
        mask_ratio=0.15,
        lambda_temporal=0.1,
        lambda_energy=0.01
    )


def get_production_config() -> LModuleConfig:
    """
    Configuration for final production model targeting 100M parameters
    """
    embed_dim = 1024  # Standard value that's attention compatible
    
    # Verify attention compatibility
    if embed_dim % max(4, embed_dim // 128) != 0:
        target_heads = max(4, embed_dim // 128)
        embed_dim = (embed_dim // target_heads) * target_heads
    
    return LModuleConfig(
        n_filters=384,
        embed_dim=embed_dim,
        snn_hidden_size=1536,
        snn_output_size=1536,
        n_snn_layers=8,
        n_jepa_layers=8,
        jepa_hidden_size=1536,
        predictor_hidden_dim=2048,
        predictor_layers=6,
        target_params=100_000_000,
        dropout=0.1,
        target_time_steps=150,
        mask_ratio=0.15,
        lambda_temporal=0.1,
        lambda_energy=0.01
    )


# Default to production config
DEFAULT_100M_CONFIG = get_production_config()