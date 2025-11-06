#!/usr/bin/env python3
"""
Script to download external datasets from HuggingFace for PROJECT EMBER.

This script downloads the required datasets that are too large to store in git:
- RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- MSP-IMPROV Full dataset

Usage:
    python download_external_datasets.py --output-dir external-datasets
"""

import os
import argparse
from pathlib import Path
from datasets import load_dataset


def download_ravdess(output_dir):
    """Download RAVDESS dataset from HuggingFace."""
    print("Downloading RAVDESS dataset from HuggingFace...")
    print("Dataset: RAVDESS (Emotional speech and song database)")

    output_path = Path(output_dir) / "HF" / "ravdess"
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Download from HuggingFace
        # Note: Replace with the actual HuggingFace dataset identifier
        dataset = load_dataset("narad/ravdess", cache_dir=str(output_path))

        print(f"✓ RAVDESS dataset downloaded successfully to {output_path}")
        print(f"  Train samples: {len(dataset.get('train', []))}")
        if 'validation' in dataset:
            print(f"  Validation samples: {len(dataset['validation'])}")
        if 'test' in dataset:
            print(f"  Test samples: {len(dataset['test'])}")

        return True
    except Exception as e:
        print(f"✗ Error downloading RAVDESS: {e}")
        print("  Please check the dataset name and your HuggingFace credentials")
        return False


def download_msp_improv(output_dir):
    """Download MSP-IMPROV Full dataset from HuggingFace."""
    print("\nDownloading MSP-IMPROV Full dataset from HuggingFace...")
    print("Dataset: MSP-IMPROV (Multimodal Signal Processing Improvisation)")

    output_path = Path(output_dir) / "HF" / "MSP_Improv_Full"
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Download from HuggingFace
        # Note: Replace with the actual HuggingFace dataset identifier
        dataset = load_dataset("narad/msp-improv", cache_dir=str(output_path))

        print(f"✓ MSP-IMPROV dataset downloaded successfully to {output_path}")
        print(f"  Train samples: {len(dataset.get('train', []))}")
        if 'validation' in dataset:
            print(f"  Validation samples: {len(dataset['validation'])}")
        if 'test' in dataset:
            print(f"  Test samples: {len(dataset['test'])}")

        return True
    except Exception as e:
        print(f"✗ Error downloading MSP-IMPROV: {e}")
        print("  Please check the dataset name and your HuggingFace credentials")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download external datasets from HuggingFace for PROJECT EMBER'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='external-datasets',
        help='Directory to store downloaded datasets (default: external-datasets)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['ravdess', 'msp-improv', 'all'],
        default='all',
        help='Which dataset to download (default: all)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PROJECT EMBER - External Dataset Download")
    print("=" * 70)
    print(f"Output directory: {output_dir.absolute()}")
    print()

    success = True

    if args.dataset in ['ravdess', 'all']:
        if not download_ravdess(output_dir):
            success = False

    if args.dataset in ['msp-improv', 'all']:
        if not download_msp_improv(output_dir):
            success = False

    print("\n" + "=" * 70)
    if success:
        print("✓ All datasets downloaded successfully!")
    else:
        print("✗ Some datasets failed to download. Check the errors above.")
    print("=" * 70)

    if not success:
        print("\nTroubleshooting:")
        print("1. Install required packages: pip install datasets")
        print("2. Verify the HuggingFace dataset names are correct")
        print("3. You may need to authenticate with HuggingFace:")
        print("   huggingface-cli login")
        print("4. Some datasets require accepting terms on HuggingFace website")


if __name__ == "__main__":
    main()
