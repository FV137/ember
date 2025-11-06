#!/usr/bin/env python3
"""
Run PROJECT EMBER Phase 0 training with real distress vocalization detection.
This script can use either synthetic data (for initial validation) or real data
(when available) for training the binding test.
"""
import os
import sys
import torch
import argparse
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from train_binding_test import main as run_binding_test
from utils.dataset import DistressVocalizationDataset
from utils.synthetic_audio import create_synthetic_dataset


def check_real_dataset_availability(data_dir):
    """
    Check if real dataset files are available in the data directory.
    
    Args:
        data_dir: Directory to check for real dataset files
        
    Returns:
        bool: True if real files exist, False otherwise
    """
    data_path = Path(data_dir)
    
    # Check for real audio files (not synthetic ones)
    real_audio_patterns = [
        # EmoDB patterns
        data_path / "03a01Fa.wav",
        data_path / "03a01Nc.wav",
        # RAVDESS patterns  
        data_path / "Actor_01/03-01-01-01-01-01-01.wav",
        # CREMA patterns
        data_path / "1001_DFA_ANG_XX.wav",
    ]
    
    # Count actual audio files in the directory
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(data_path.rglob(f'*{ext}')))
        audio_files.extend(list(data_path.rglob(f'*{ext.upper()}')))
    
    # Filter out synthetic files (files starting with 'distress_')
    real_files = [f for f in audio_files if not f.name.startswith('distress_')]
    
    print(f"Found {len(audio_files)} total audio files in {data_path}")
    print(f"Found {len(real_files)} potentially real (non-synthetic) audio files")
    
    # If we have more than synthetic data, we likely have real data
    has_real_data = len(real_files) > 0
    has_synthetic_data = any(f.name.startswith('distress_') for f in audio_files)
    
    return has_real_data, has_synthetic_data


def main():
    parser = argparse.ArgumentParser(description='Run PROJECT EMBER Phase 0 with real or synthetic data')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing audio dataset (default: data)')
    parser.add_argument('--force-synthetic', action='store_true',
                        help='Force using synthetic data even if real data is available')
    parser.add_argument('--create-synthetic-if-missing', action='store_true',
                        help='Create synthetic dataset if no data is found')
    
    args = parser.parse_args()
    
    print(f"Running PROJECT EMBER Phase 0 training with data from: {args.data_dir}")
    
    # Check dataset availability
    has_real_data, has_synthetic_data = check_real_dataset_availability(args.data_dir)
    
    print(f"Dataset status:")
    print(f"  Has real data: {has_real_data}")
    print(f"  Has synthetic data: {has_synthetic_data}")
    
    if args.force_synthetic:
        print("  Using synthetic data (forced by --force-synthetic)")
    elif has_real_data and not args.force_synthetic:
        print("  Using real data detected in directory")
    elif has_synthetic_data:
        print("  Using existing synthetic data")
    elif args.create_synthetic_if_missing:
        print("  No data found, creating synthetic dataset...")
        data_path = Path(args.data_dir)
        data_path.mkdir(exist_ok=True)
        create_synthetic_dataset(data_path, num_samples=200)
        print("  Created synthetic dataset")
    else:
        print("ERROR: No audio data found and --create-synthetic-if-missing not specified!")
        print("Please either:")
        print("  1. Place real audio files in the data directory")
        print("  2. Use --create-synthetic-if-missing to create synthetic data")
        print("  3. Download a real dataset (EmoDB, RAVDESS, CREMA-D) and place it here")
        sys.exit(1)
    
    # Verify dataset before running training
    try:
        dataset = DistressVocalizationDataset(
            data_dir=args.data_dir, 
            mode='train'
        )
        print(f"Dataset successfully loaded with {len(dataset)} samples")
        print(f"Sample shape: {dataset[0][0].shape}")
        print(f"Sample label: {dataset[0][1]}")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        sys.exit(1)
    
    # Run the actual training
    print("="*60)
    print("Starting PROJECT EMBER Phase 0 Binding Test Training")
    print("="*60)
    
    # Run the binding test training
    results = run_binding_test()
    
    print("="*60)
    print("PROJECT EMBER Phase 0 Completed!")
    print(f"Results: {results}")
    
    # Check if binding was successful
    binding_successful = results.get('binding_successful', False)
    performance_gap = results.get('performance_gap', 0.0)
    
    print(f"\nBinding Test Result: {'SUCCESS' if binding_successful else 'FAILED'}")
    print(f"Performance Gap: {performance_gap:.4f}")
    
    if binding_successful:
        print("\n✅ The H-module can effectively use subsymbolic sensory embeddings from L-module!")
        print("This validates the core architecture concept. Proceeding to Phase 1 is recommended.")
    else:
        print("\n❌ The binding test failed. The H-module cannot effectively use L-module embeddings.")
        print("Consider adjusting the architecture before proceeding.")
    
    return results


if __name__ == "__main__":
    main()