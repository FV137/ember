from dataclasses import dataclass
from typing import Tuple


@dataclass
class LModuleConfig:
    """
    Configuration for the L-Module (Phase 1)
    """
    # Audio processing parameters
    sample_rate: int = 16000
    n_filters: int = 256  # Increased for higher capacity
    low_freq: int = 50
    high_freq: int = 8000
    compression_factor: float = 0.2
    
    # Embedding dimensions
    embed_dim: int = 768  # Standard for transformer compatibility
    snn_hidden_size: int = 1024  # Increased for higher capacity
    snn_output_size: int = 1024  # For phase encoding output
    
    # SNN architecture parameters
    n_snn_layers: int = 6  # More layers for increased capacity
    snn_beta: float = 0.9  # LIF neuron parameter
    snn_threshold: float = 1.0  # LIF neuron threshold
    
    # JEPA architecture parameters
    n_jepa_layers: int = 6  # More layers for increased capacity
    jepa_hidden_size: int = 1024  # Increased for higher capacity
    
    # Network architecture
    predictor_hidden_dim: int = 1536  # For JEPA predictor
    predictor_layers: int = 4  # Predictor layers
    
    # Regularization
    dropout: float = 0.1
    target_params: int = 100_000_000  # 100M parameters target
    
    # Training parameters
    mask_ratio: float = 0.15  # For masked prediction training
    lambda_temporal: float = 1.0  # Weight for temporal coherence loss
    lambda_energy: float = 0.1  # Weight for energy efficiency loss
    
    # Audio processing
    target_time_steps: int = 200  # Target time steps for processing
    window_size: float = 2.0  # Window size in seconds


# Default configuration
DEFAULT_CONFIG = LModuleConfig()


def get_large_config() -> LModuleConfig:
    """Configuration for a larger L-Module"""
    return LModuleConfig(
        n_filters=512,
        embed_dim=1024,
        snn_hidden_size=2048,
        snn_output_size=2048,
        n_snn_layers=8,
        n_jepa_layers=8,
        jepa_hidden_size=2048,
        predictor_hidden_dim=2048,
        predictor_layers=6,
        target_params=100_000_000
    )


def get_xlarge_config() -> LModuleConfig:
    """Configuration for an extra-large L-Module to reach 100M parameters"""
    return LModuleConfig(
        n_filters=512,
        embed_dim=1536,
        snn_hidden_size=2048,
        snn_output_size=2048,
        n_snn_layers=10,
        n_jepa_layers=10,
        jepa_hidden_size=2048,
        predictor_hidden_dim=3072,
        predictor_layers=8,
        target_params=100_000_000
    )


def get_phase1_config() -> LModuleConfig:
    """Configuration optimized for Phase 1 to achieve ~100M parameters"""
    return LModuleConfig(
        n_filters=384,  # Reduced from 512
        embed_dim=1024,  # Standard transformer size
        snn_hidden_size=1536,  # Reduced from 2048
        snn_output_size=1536,  # Reduced from 2048
        n_snn_layers=8,  # Keep at 8
        n_jepa_layers=8,  # Keep at 8
        jepa_hidden_size=1536,  # Reduced from 2048
        predictor_hidden_dim=2048,  # Keep at 2048 for prediction power
        predictor_layers=6,  # Keep at 6
        target_params=100_000_000
    )


def get_medium_config() -> LModuleConfig:
    """Configuration for a medium L-Module"""
    return LModuleConfig(
        n_filters=256,
        embed_dim=768,
        snn_hidden_size=1024,
        snn_output_size=1024,
        n_snn_layers=6,
        n_jepa_layers=6,
        jepa_hidden_size=1024,
        predictor_hidden_dim=1536,
        predictor_layers=4,
        target_params=100_000_000
    )


def get_small_config() -> LModuleConfig:
    """Configuration for a smaller L-Module (for testing)"""
    return LModuleConfig(
        n_filters=64,
        embed_dim=512,
        snn_hidden_size=512,
        snn_output_size=512,
        n_snn_layers=4,
        n_jepa_layers=3,
        jepa_hidden_size=512,
        predictor_hidden_dim=768,
        predictor_layers=3,
        target_params=20_000_000
    )