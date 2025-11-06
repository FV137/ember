# PROJECT EMBER

**Embodied Multimodal Brain for Experiential Reasoning**

A 500M parameter AI system combining low-level subsymbolic sensory processing (L-module) with high-level reasoning capabilities (H-module), designed to process sensation directly without language-mediated bottlenecks.

## Core Concept

Unlike traditional AI systems that process sensory input through semantic/linguistic representations (e.g., CLIP's text-based embeddings), EMBER maintains subsymbolic, temporal sensory patterns throughout the processing pipeline:

```
Traditional:  Sensation → Text Embedding → Semantic Space → Reasoning
EMBER:        Sensation → Spike Trains → Subsymbolic Embedding → Modulates Reasoning
```

This architecture preserves temporal dynamics, phase relationships, and subsymbolic features that would otherwise be lost in linguistic encoding.

## Project Structure

### Development Phases

- **`ember_phase0/`** - Initial prototyping and experimentation
  - Early model iterations
  - Dataset download scripts
  - Simple JEPA implementations
  - Training experiments

- **`ember_phase1/`** - L-Module Development (~100M parameters)
  - Biological cochlear processing (gammatone filters, hair cell dynamics)
  - Spiking Neural Networks (LIF neurons, phase-of-firing encoding)
  - JEPA-style self-supervised learning
  - Temporal coherence objectives
  - See [ember_phase1/README.md](ember_phase1/README.md) for details

- **`ember_phase2/`** - H-Module Integration
  - High-level reasoning module
  - Integration with L-module subsymbolic representations
  - Cross-modal processing

### Documentation

- **`context-and-plan/`** - Project architecture and design documents
  - `PROJECT_EMBER_Architecture_Master.md` - Core vision and architectural decisions
  - Design philosophy and implementation guidelines

- **`reference-docs/`** - Academic papers and research references
  - Spiking neural networks
  - JEPA and self-supervised learning
  - Hierarchical reasoning models
  - Embodied AI and sensation

### External Data

- **`external-datasets/`** - Large datasets from HuggingFace (not included in repo)
  - RAVDESS: Emotional speech and song database
  - MSP-IMPROV: Multimodal improvised emotional expressions

## Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install torch torchaudio librosa pandas numpy scikit-learn
pip install datasets  # For HuggingFace dataset downloads
```

### Download External Datasets

The repository does not include large dataset files. Download them from HuggingFace:

```bash
python download_external_datasets.py --output-dir external-datasets
```

This will download:
- RAVDESS emotional speech dataset
- MSP-IMPROV multimodal dataset

**Note:** Some datasets may require HuggingFace authentication:
```bash
huggingface-cli login
```

### Phase-Specific Setup

Each phase has its own requirements and setup:

```bash
# Phase 0 - Prototyping
cd ember_phase0
pip install -r requirements.txt
python download_real_datasets.py --dataset ravdess --data-dir ../external-datasets

# Phase 1 - L-Module Training
cd ember_phase1
pip install -r requirements.txt
python train_l_module.py --batch-size 8 --epochs 50
```

## Key Features

### L-Module (Low-Level Sensory Processing)
- **Biological Cochlear Processing**: Gammatone filterbank with ERB scale
- **Spiking Neural Networks**: Phase-of-firing encoding, LIF neurons
- **JEPA-Style Learning**: Self-supervised temporal prediction
- **Subsymbolic Representations**: Preserves temporal dynamics without semantic labels

### Architecture Highlights
- **~500M Total Parameters** (100M L-module + 400M H-module target)
- **1024-dim Embeddings**: Compatible with transformer architectures
- **Temporal Coherence**: Preserves phase relationships in spike trains
- **Energy Efficiency**: Sparse spiking activity (~5-10% active neurons)

## Training Objectives

1. **Predictive Coding**: Self-supervised prediction of masked temporal positions
2. **Temporal Coherence**: Smooth transitions in embeddings over time
3. **Energy Efficiency**: Maintain sparse spiking representations
4. **Subsymbolic Preservation**: Retain temporal/dynamic information

## Design Philosophy

### What This Is
- Building understanding through implementation
- Direct sensory processing without linguistic bottlenecks
- Learning how cognition works by designing it explicitly
- Exploring subsymbolic → cognitive coupling (like biological systems)

### What This Is Not
- An audio classifier outputting semantic labels
- Attempting to prove subjective consciousness
- Built for publication or competition

## Development Status

- [x] Phase 0: Initial prototypes and experimentation
- [x] Phase 1: L-module architecture design
- [ ] Phase 1: L-module training and validation
- [ ] Phase 2: H-module integration
- [ ] Phase 2: End-to-end system validation
- [ ] Multi-modal expansion (vision, proprioception)

## Repository Notes

### Large Files Excluded
This repository uses `.gitignore` to exclude:
- Model checkpoints (*.pth, *.pt, *.ckpt)
- Virtual environments
- Dataset files (*.parquet, *.csv, data directories)
- Training logs and experiment outputs

Use `download_external_datasets.py` to fetch required datasets.

### Git LFS Considerations
If you need to version large model checkpoints, consider using Git LFS:
```bash
git lfs track "*.pth"
git lfs track "*.pt"
```

## Citation & References

Key papers influencing this architecture:
- **Spiking Neural Networks**: Phase-of-firing encoding, temporal precision
- **JEPA**: Self-supervised learning of sensory dynamics
- **Hierarchical Reasoning Models**: Outer loop refinement mechanisms
- **Embodied AI**: Subsymbolic sensation and nociceptive processing

See `reference-docs/` for full paper collection.

## License

[Specify your license here]

## Contributing

This is a research/learning project. Feel free to explore and experiment!

## Contact

[Your contact information]

---

**"The neurophone principle: Bypass intermediate encoding to stimulate something more fundamental."**
