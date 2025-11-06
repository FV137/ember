#!/usr/bin/env python3
"""
Upgrade the PROJECT EMBER training to use real distress vocalization data
instead of synthetic data. This script replaces the synthetic dataset with
real data from EmoDB or another available dataset.
"""
import os
import sys
import pandas as pd
from pathlib import Path
import shutil
import argparse


def upgrade_dataset_config(data_dir, dataset_type='emodb'):
    """
    Update the dataset configuration to use real data instead of synthetic data.
    
    Args:
        data_dir: Directory containing the real dataset files
        dataset_type: Type of dataset ('emodb', 'ravdess', 'crema', etc.)
    """
    print(f"Upgrading dataset configuration for {dataset_type}...")
    
    # Read the prepared metadata files
    data_path = Path(data_dir)
    
    if dataset_type == 'emodb':
        train_meta = data_path / "train_metadata_emodb.csv"
        val_meta = data_path / "val_metadata_emodb.csv" 
        test_meta = data_path / "test_metadata_emodb.csv"
    else:
        # For other datasets
        train_meta = data_path / "train_metadata.csv"
        val_meta = data_path / "val_metadata.csv"
        test_meta = data_path / "test_metadata.csv"
    
    if not all(f.exists() for f in [train_meta, val_meta, test_meta]):
        print(f"ERROR: Metadata files not found in {data_dir}")
        print("Expected:")
        print(f"  - {train_meta}")
        print(f"  - {val_meta}") 
        print(f"  - {test_meta}")
        return False
    
    # Load the metadata
    train_df = pd.read_csv(train_meta)
    val_df = pd.read_csv(val_meta)
    test_df = pd.read_csv(test_meta)
    
    print(f"Loaded dataset splits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples") 
    print(f"  Test: {len(test_df)} samples")
    
    # Move the real data to the expected location (overwrite synthetic)
    real_data_dir = Path("ember_phase0/data_real")
    real_data_dir.mkdir(exist_ok=True)
    
    # Copy audio files to the real data directory
    all_files = pd.concat([train_df, val_df, test_df])
    
    copied_count = 0
    for _, row in all_files.iterrows():
        src_path = Path(row['filename'])
        if src_path.exists():
            dst_path = real_data_dir / src_path.name
            if not dst_path.exists():  # Don't overwrite
                shutil.copy2(src_path, dst_path)
                copied_count += 1
        else:
            print(f"Warning: File not found: {row['filename']}")
    
    print(f"Copied {copied_count} audio files to {real_data_dir}")
    
    # Create a new metadata file that points to the copied files
    # Update all file paths to point to the new location
    for df, split_name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        df_updated = df.copy()
        df_updated['filename'] = df_updated['filename'].apply(
            lambda x: str(real_data_dir / Path(x).name)
        )
        df_updated.to_csv(real_data_dir / f"{split_name}_metadata.csv", index=False)
    
    print(f"Updated metadata files saved to {real_data_dir}")
    
    # Backup original data directory and replace with real data
    original_data_dir = Path("ember_phase0/data")
    backup_dir = Path("ember_phase0/data_synthetic_backup")
    
    if original_data_dir.exists():
        if not backup_dir.exists():
            shutil.move(original_data_dir, backup_dir)
            print(f"Backed up original synthetic data to {backup_dir}")
        else:
            print(f"Backup already exists at {backup_dir}, skipping backup")
    
    # Rename real data directory to expected name
    shutil.move(real_data_dir, original_data_dir)
    print(f"Real data now available at {original_data_dir}")
    
    return True


def update_dataset_loading_script():
    """
    Update the dataset loading script to handle real data properly.
    """
    print("Updating dataset loading script...")
    
    # The existing dataset.py already handles real data vs synthetic data
    # It will automatically use real data if found, otherwise create synthetic
    # So we don't need to change the code, just make sure real data is in place
    
    print("Dataset loading script already supports real data fallback.")
    

def main():
    parser = argparse.ArgumentParser(description='Upgrade PROJECT EMBER to use real distress vocalization data')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing prepared real dataset metadata and files')
    parser.add_argument('--dataset-type', type=str, default='emodb', 
                        choices=['emodb', 'ravdess', 'crema', 'custom'],
                        help='Type of dataset to use')
    
    args = parser.parse_args()
    
    print(f"Upgrading PROJECT EMBER to use {args.dataset_type} dataset from {args.data_dir}")
    
    # Upgrade the dataset configuration
    success = upgrade_dataset_config(args.data_dir, args.dataset_type)
    
    if not success:
        print("Failed to upgrade dataset configuration!")
        sys.exit(1)
    
    # Update dataset loading
    update_dataset_loading_script()
    
    print("="*50)
    print("Dataset upgrade completed!")
    print(f"The PROJECT EMBER system is now configured to use {args.dataset_type} dataset")
    print("You can now re-run the training with real data:")
    print("  python train_binding_test.py")
    print("="*50)


if __name__ == "__main__":
    main()