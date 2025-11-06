# PROJECT EMBER - Phase 1: L-Module Development

## Overview

Phase 1 focuses on developing the full-scale L-module (100M parameters) that processes raw sensory input using spiking neural networks with JEPA-style self-supervised learning.

## Architecture

The L-module combines:

1. **Biological Cochlear Processing**:
   - Gammatone filterbank with ERB scale frequencies
   - Cochlear stage with outer/inner hair cell dynamics
   - Signal compression modeling real cochlear response

2. **Spiking Neural Networks**:
   - Phase-of-firing encoding to preserve temporal relationships
   - Leaky Integrate-and-Fire (LIF) neurons
   - Spiking recurrent layers for temporal processing

3. **JEPA-Style Learning**:
   - Self-supervised prediction of temporal dynamics
   - Focus on preserving subsymbolic features rather than semantic labels
   - Temporal coherence and energy efficiency objectives

## Model Configuration

- **Target Parameters**: ~100M
- **Embedding Dimension**: 1024 (compatible with transformer architectures)
- **Audio Processing**: 16kHz sample rate, 2-second windows
- **SNN Architecture**: 8-layer phase encoding + 8-layer JEPA encoder
- **Hidden Sizes**: 1536-dimensional representations

## File Structure

```
ember_phase1/
├── models/
│   ├── l_module.py          # Main L-module architecture
│   ├── cochlear_processing.py  # Biological cochlear models
│   ├── spiking_layers.py    # SNN components
├── configs/
│   └── l_module_config.py   # Model configurations
├── utils/
│   └── dataset.py           # Audio dataset utilities
├── train_l_module.py        # Training script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Key Features

- **Temporal Coherence**: Preserves phase relationships in spike trains
- **Energy Efficiency**: Sparse spiking activity (~5-10% of neurons active)
- **Self-Supervised Learning**: Learns representations without semantic labels
- **Transformer Compatibility**: 1024-dim embeddings for H-module integration
- **Biological Fidelity**: Gammatone filters and cochlear processing models

## Training Objectives

1. **Predictive Coding**: Predict masked temporal positions
2. **Temporal Coherence**: Encourage smooth transitions in embeddings
3. **Energy Efficiency**: Maintain sparse spiking representations
4. **Subsymbolic Preservation**: Maintain temporal/dynamic information

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train the L-module
python train_l_module.py --batch-size 8 --epochs 50 --lr 1e-4

# Or with CUDA support
python train_l_module.py --cuda --batch-size 16
```

## Validation

The model will be validated through:
- Temporal coherence metrics
- Spike rate analysis
- Self-supervised loss convergence
- Compatibility with H-module interface
- Behavioral tests on audio discrimination tasks

## Next Steps

After successful training of Phase 1:
1. Validate temporal dynamics preservation
2. Test integration with H-module
3. Benchmark against semantic baselines
4. Prepare for Phase 2 (H-module integration)