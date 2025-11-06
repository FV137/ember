#!/usr/bin/env python3
"""
Script to download and prepare the Berlin Database of Emotional Speech (EmoDB)
which is freely available and can be used for distress vocalization detection.
"""
import os
import sys
import requests
import tarfile
import zipfile
import shutil
from pathlib import Path
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import json


def download_file(url, destination, chunk_size=8192):
    """Download a file with progress bar."""
    print(f"Downloading {url} to {destination}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded_size = 0
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)
                if total_size > 0:
                    percent = (downloaded_size / total_size) * 100
                    print(f"\rProgress: {percent:.1f}% ({downloaded_size}/{total_size} bytes)", end='', flush=True)
    
    print(f"\nDownload completed: {destination}")


def extract_archive(archive_path, extract_to):
    """Extract tar.gz or zip archive."""
    print(f"Extracting {archive_path} to {extract_to}")
    os.makedirs(extract_to, exist_ok=True)
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    
    print("Extraction completed")


def prepare_emodb_dataset(data_dir):
    """
    Prepare the Berlin Database of Emotional Speech (EmoDB).
    This is freely available: http://emodb.bilderbar.info/download/
    Emotions: W (anger), L (disgust), E (fear), A (joy), F (sadness), T (boredom), N (neutrality)
    We'll consider anger (W), disgust (L), fear (E), and sadness (F) as distress vocalizations.
    """
    print("Preparing EmoDB dataset...")
    data_path = Path(data_dir)
    
    # EmoDB files follow the naming convention: 
    # XXYYZZ.wav where:
    # - XX is speaker ID (03-20)
    # - YY is emotion code: W=anger, L=disgust, E=fear, A=joy, F=sadness, T=boredom, N=neutral
    # - ZZ is sentence number
    
    audio_files = list(data_path.rglob("*.wav")) + list(data_path.rglob("*.WAV"))
    
    if not audio_files:
        print("No audio files found in EmoDB dataset!")
        return None
    
    print(f"Found {len(audio_files)} audio files in EmoDB dataset")
    
    # Create metadata for distress vs non-distress
    metadata = {
        'filename': [],
        'label': [],  # 0 for non-distress, 1 for distress
        'emotion': [],  # emotion code
        'speaker_id': [],  # speaker ID
        'sentence': [],  # sentence number
        'duration': []  # audio duration
    }
    
    for file_path in audio_files:
        filename = file_path.name.replace('.wav', '').replace('.WAV', '')
        
        if len(filename) >= 6:  # XXYYZZ format
            speaker_id = filename[:2]
            emotion_code = filename[2:4]  # middle two characters are emotion
            sentence_num = filename[4:6]  # last two characters are sentence number
            
            # Map emotions to distressed (1) vs non-distressed (0)
            # Consider anger (W), disgust (L), fear (E), and sadness (F) as distress
            if emotion_code in ['W', 'L', 'E', 'F']:  # anger, disgust, fear, sadness
                label = 1  # distress
            elif emotion_code in ['A', 'T', 'N']:  # joy, boredom, neutral
                label = 0  # non-distress
            else:
                label = 0  # default to non-distress
            
            try:
                duration = librosa.get_duration(path=str(file_path))
            except:
                duration = 0.0  # fallback if librosa fails
            
            metadata['filename'].append(str(file_path))
            metadata['label'].append(label)
            metadata['emotion'].append(emotion_code)
            metadata['speaker_id'].append(speaker_id)
            metadata['sentence'].append(sentence_num)
            metadata['duration'].append(duration)
        else:
            # If file doesn't match EmoDB naming convention, treat as unknown
            try:
                duration = librosa.get_duration(path=str(file_path))
            except:
                duration = 0.0
            metadata['filename'].append(str(file_path))
            metadata['label'].append(0)  # default to non-distress
            metadata['emotion'].append('unknown')
            metadata['speaker_id'].append('unknown')
            metadata['sentence'].append('unknown')
            metadata['duration'].append(duration)
    
    df = pd.DataFrame(metadata)
    
    # Filter out very short files that might not be meaningful for distress detection
    df = df[df['duration'] >= 0.5]  # At least 0.5 seconds
    
    print(f"Prepared metadata for {len(df)} files (after filtering very short files)")
    print(f"Distress vocalizations: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    
    return df


def split_and_save_dataset(df, output_dir, test_size=0.2, val_size=0.1):
    """Split dataset into train, validation, and test sets."""
    print("Splitting dataset into train/val/test...")
    
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # First split: separate test set
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=42, 
        stratify=df['label']
    )
    
    # Then split train into train and validation
    final_test_size = val_size / (1 - test_size)  # Adjust for remaining data
    train_df, val_df = train_test_split(
        train_df,
        test_size=final_test_size,
        random_state=42,
        stratify=train_df['label']
    )
    
    # Save the splits
    train_df.to_csv(output_path / "train_metadata_emodb.csv", index=False)
    val_df.to_csv(output_path / "val_metadata_emodb.csv", index=False)
    test_df.to_csv(output_path / "test_metadata_emodb.csv", index=False)
    
    print(f"Dataset split completed:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Also save a summary JSON
    summary = {
        'total_samples': len(df),
        'distress_samples': int(df['label'].sum()),
        'non_distress_samples': int(len(df) - df['label'].sum()),
        'distress_percentage': float(df['label'].mean() * 100),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'split_ratios': {
            'train': len(train_df)/len(df),
            'val': len(val_df)/len(df),
            'test': len(test_df)/len(df)
        }
    }
    
    with open(output_path / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to {output_path / 'dataset_summary.json'}")
    
    return train_df, val_df, test_df


def download_emodb(data_dir):
    """Download EmoDB dataset - this is freely available"""
    print("Downloading EmoDB dataset. This is freely available at:")
    print("http://emodb.bilderbar.info/download/")
    print("However, the download requires navigating their website and agreeing to terms.")
    print("Please download the wav files manually from the link above.")
    print("Then extract the files to:", data_dir)
    
    # Note: EmoDB isn't directly downloadable via script due to the way it's distributed
    # Users need to navigate the website, agree to terms, and download manually
    return False


def main():
    parser = argparse.ArgumentParser(description='Download and prepare EmoDB dataset for distress vocalization training')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing the EmoDB dataset files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save prepared metadata (defaults to data-dir)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.data_dir
    
    print(f"Preparing EmoDB dataset from {args.data_dir}")
    
    # Prepare EmoDB dataset
    emodb_df = prepare_emodb_dataset(args.data_dir)
    
    if emodb_df is None or len(emodb_df) == 0:
        print("ERROR: No valid EmoDB files found!")
        print("Make sure you have downloaded and extracted the EmoDB wav files to the specified directory.")
        print("Visit http://emodb.bilderbar.info/download/ to download the dataset.")
        sys.exit(1)
    
    print(f"Found {len(emodb_df)} valid audio files")
    print("Sample emotions distribution:", emodb_df['emotion'].value_counts().to_dict())
    
    # Split and save the dataset
    splits = split_and_save_dataset(emodb_df, args.output_dir)
    
    print("="*50)
    print("EmoDB dataset preparation completed!")
    print("Metadata files saved to:", args.output_dir)
    
    # Display summary
    print("\nDataset Summary:")
    print(f"  Total samples: {len(emodb_df)}")
    print(f"  Distress samples: {emodb_df['label'].sum()}")
    print(f"  Non-distress samples: {len(emodb_df) - emodb_df['label'].sum()}")
    print(f"  Distress percentage: {emodb_df['label'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()