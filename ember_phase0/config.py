# Configuration for Phase 0 Binding Test
import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# Model dimensions
L_MINI_EMBED_DIM = 512
H_MINI_EMBED_DIM = 512
TEXT_EMBED_DIM = 128
AUDIO_SAMPLE_RATE = 16000
AUDIO_WINDOW_SIZE = 2  # seconds
AUDIO_WINDOW_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_SIZE

# Model parameters
L_MINI_PARAMS = 20_000_000  # 20M
H_MINI_PARAMS = 100_000_000  # 100M
TOTAL_PARAMS = L_MINI_PARAMS + H_MINI_PARAMS

# SNN parameters
SNN_LIF_THRESHOLD = 1.0
SNN_LIF_TAU = 2.0
SPIKE_SPARSITY_TARGET = 0.15  # 15% activation rate

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Dataset parameters
DATASET_PATH = "data/"
CONTROL_CONDITION = True  # Whether to run text-only baseline

# Experiment tracking
WANDB_PROJECT = "ember_phase0_binding_test"
EXPERIMENT_NAME = "binding_test_v1"