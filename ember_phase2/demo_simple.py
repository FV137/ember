#!/usr/bin/env python3
"""
Simple demonstration of Phase 2 concepts
"""

import torch
import torch.nn as nn
from models.l_module import LModule
from models.h_module import HModule
from models.lh_binding import LHBindingModule
from configs.l_module_config import get_phase1_config


def main():
    print("=" * 60)
    print("PROJECT EMBER - PHASE 2 CONCEPT DEMONSTRATION")
    print("=" * 60)
    
    # Create models
    print("\n1. Creating models...")
    
    # L-Module (Phase 1)
    l_config = get_phase1_config()
    l_module = LModule(l_config)
    
    # H-Module
    h_module = HModule(
        sensory_dim=1024,
        text_model_name="distilbert-base-uncased",
        embed_dim=1024,
        num_reasoning_layers=6,
        num_reasoning_heads=8
    )
    
    # L-H Binding
    binding_module = LHBindingModule(
        l_dim=1024,
        h_dim=1024,
        embed_dim=1024,
        num_heads=8
    )
    
    print(f"   L-Module: {sum(p.numel() for p in l_module.parameters()):,} parameters")
    print(f"   H-Module: {sum(p.numel() for p in h_module.parameters()):,} parameters")
    print(f"   Binding:  {sum(p.numel() for p in binding_module.parameters()):,} parameters")
    
    # Test with dummy data
    print("\n2. Testing with dummy data...")
    
    # Dummy audio input (2 seconds at 16kHz)
    dummy_audio = torch.randn(1, 32000)
    print(f"   Audio input: {dummy_audio.shape}")
    
    # Dummy text
    dummy_text = ["Describe this sound."]
    print(f"   Text input: {dummy_text}")
    
    # Process through L-Module
    print("\n3. L-Module processing...")
    with torch.no_grad():
        l_outputs = l_module.self_supervised_forward(dummy_audio)
        l_embeddings, l_predictions, l_targets = l_outputs
        print(f"   L embeddings: {l_embeddings.shape}")
    
    # Process through H-Module
    print("\n4. H-Module processing...")
    # Simplified text processing for demo
    dummy_text_embed = torch.randn(1, 1024)  # Simulated text embedding
    with torch.no_grad():
        h_outputs = h_module(
            sensory_embed=l_embeddings,
            text_input_ids=torch.randint(0, 1000, (1, 10)),  # Dummy token IDs
            text_attention_mask=torch.ones(1, 10),  # Dummy attention mask
            task="language_modeling"
        )
        print(f"   H embeddings: {h_outputs['embeddings'].shape}")
    
    # Process through Binding
    print("\n5. L-H Binding...")
    with torch.no_grad():
        # Add sequence dimension to L embeddings
        l_embeddings_seq = l_embeddings.unsqueeze(1)
        
        binding_outputs = binding_module(
            l_embeddings=l_embeddings_seq,
            h_embeddings=h_outputs['embeddings'],
            h_mask=torch.ones(1, h_outputs['embeddings'].size(1)).bool()
        )
        print(f"   Bound L embeddings: {binding_outputs['bound_l_embeddings'].shape}")
        print(f"   Bound H embeddings: {binding_outputs['bound_h_embeddings'].shape}")
    
    print("\n" + "=" * 60)
    print("PHASE 2 CONCEPTS DEMONSTRATED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("Key Concepts Showcased:")
    print("  ✓ L-Module: Spiking neural network sensory processing")
    print("  ✓ H-Module: Transformer-based cognitive reasoning")
    print("  ✓ L-H Binding: Cross-modal integration mechanism")
    print("  ✓ Pipeline: Audio → Sensation → Cognition → Language")
    print()
    print("Next Steps:")
    print("  1. Full training with real datasets")
    print("  2. Sensorimotor loop implementation")
    print("  3. Cross-modal extension (vision, touch)")
    print("  4. Real-world deployment and testing")
    print()
    print("PROJECT EMBER - Bridging Sensation and Cognition")


if __name__ == "__main__":
    main()