#!/usr/bin/env python3
"""
PROJECT EMBER - Phase 2: L-H Module Integration
===============================================

Main entry point for Phase 2 implementation that integrates:
1. L-Module (sensory processing) 
2. H-Module (reasoning and language)
3. L-H Binding (cross-modal integration)

This represents the core architecture of PROJECT EMBER where sensation meets cognition.
"""

import torch
import argparse
import os
from datetime import datetime

from models.l_module import LModule
from models.h_module import HModule  
from models.lh_binding import LHBindingModule
from configs.l_module_config import get_phase1_config
from train_phase2 import Phase2Trainer


def create_phase2_models():
    """
    Create all models for Phase 2 integration
    """
    print("Creating Phase 2 models...")
    
    # L-Module (reuse from Phase 1)
    l_config = get_phase1_config()
    l_module = LModule(l_config)
    
    # H-Module (cognitive processing)
    h_module = HModule(
        sensory_dim=1024,
        text_model_name="distilbert-base-uncased",
        embed_dim=1024,
        num_reasoning_layers=6,
        num_reasoning_heads=8,
        dropout=0.1
    )
    
    # L-H Binding Module (integration mechanism)
    binding_module = LHBindingModule(
        l_dim=1024,
        h_dim=1024,
        embed_dim=1024,
        num_heads=8,
        num_modulation_factors=8,
        dropout=0.1
    )
    
    return l_module, h_module, binding_module


def test_complete_pipeline(l_module, h_module, binding_module):
    """
    Test the complete L-H integration pipeline
    """
    print("Testing complete integration pipeline...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move models to device
    l_module = l_module.to(device)
    h_module = h_module.to(device)
    binding_module = binding_module.to(device)
    
    # Create sample inputs
    batch_size = 1
    audio_length = 32000  # 2 seconds at 16kHz
    
    sample_audio = torch.randn(batch_size, audio_length, device=device)
    sample_text = ["Describe the sound you hear."]
    
    print(f"Input shapes:")
    print(f"  Audio: {sample_audio.shape}")
    print(f"  Text: {sample_text}")
    
    # Test L-Module processing
    print("\n1. L-Module processing...")
    l_outputs = l_module.self_supervised_forward(sample_audio)
    l_embeddings, l_predictions, l_targets = l_outputs
    l_embeddings = l_embeddings.unsqueeze(1)  # Add sequence dimension
    print(f"  L embeddings: {l_embeddings.shape}")
    
    # Test H-Module processing  
    print("\n2. H-Module processing...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    text_inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32
    )
    
    input_ids = text_inputs['input_ids'].to(device)
    attention_mask = text_inputs['attention_mask'].to(device)
    
    h_outputs = h_module(
        sensory_embed=l_embeddings.squeeze(1),
        text_input_ids=input_ids,
        text_attention_mask=attention_mask,
        task="language_modeling"
    )
    h_embeddings = h_outputs['embeddings']
    print(f"  H embeddings: {h_embeddings.shape}")
    
    # Test L-H Binding
    print("\n3. L-H Binding processing...")
    binding_outputs = binding_module(
        l_embeddings=l_embeddings,
        h_embeddings=h_embeddings,
        h_mask=attention_mask.bool()
    )
    
    print(f"  Bound L embeddings: {binding_outputs['bound_l_embeddings'].shape}")
    print(f"  Bound H embeddings: {binding_outputs['bound_h_embeddings'].shape}")
    
    print("\n✓ Complete pipeline working successfully!")
    return True


def run_phase2_demo():
    """
    Run a demonstration of Phase 2 capabilities
    """
    print("=" * 60)
    print("PROJECT EMBER - PHASE 2: L-H MODULE INTEGRATION")
    print("=" * 60)
    print()
    print("Architecture Overview:")
    print("  L-Module: Sensory processing with spiking neural networks")
    print("  H-Module: Cognitive reasoning with language capabilities")  
    print("  Binding: Cross-modal integration and mutual influence")
    print()
    
    # Create models
    l_module, h_module, binding_module = create_phase2_models()
    
    print(f"Model Statistics:")
    print(f"  L-Module: {sum(p.numel() for p in l_module.parameters()):,} parameters")
    print(f"  H-Module: {sum(p.numel() for p in h_module.parameters()):,} parameters")
    print(f"  Binding:  {sum(p.numel() for p in binding_module.parameters()):,} parameters")
    print(f"  Total:    {sum(p.numel() for p in l_module.parameters()) + sum(p.numel() for p in h_module.parameters()) + sum(p.numel() for p in binding_module.parameters()):,} parameters")
    print()
    
    # Test pipeline
    success = test_complete_pipeline(l_module, h_module, binding_module)
    
    if success:
        print()
        print("🎉 PHASE 2 DEMONSTRATION COMPLETE!")
        print()
        print("Key Achievements:")
        print("  ✓ L-Module processes raw audio with SNNs")
        print("  ✓ H-Module handles language reasoning")
        print("  ✓ L-H Binding enables cross-modal integration")
        print("  ✓ Complete sensation-to-cognition pipeline")
        print()
        print("Next Steps:")
        print("  1. Full training with real datasets")
        print("  2. Sensorimotor loop implementation")
        print("  3. Embodied cognition experiments")
        print("  4. Real-world deployment testing")
        print()
        return True
    else:
        print("❌ Pipeline test failed!")
        return False


def main():
    parser = argparse.ArgumentParser(description='PROJECT EMBER - Phase 2')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--test', action='store_true', help='Run tests')
    
    args = parser.parse_args()
    
    if args.demo or (not args.train and not args.test):
        # Run demo by default
        success = run_phase2_demo()
        return 0 if success else 1
    elif args.train:
        print("Training mode - to be implemented")
        return 0
    elif args.test:
        print("Test mode - to be implemented")
        return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)