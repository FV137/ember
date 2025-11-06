#!/usr/bin/env python3
"""
Script to download and prepare real distress vocalization datasets
for PROJECT EMBER training.
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


def prepare_ravdess_dataset(data_dir):
    """
    Prepare the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).
    This is a commonly used emotional speech dataset that contains acted emotional expressions.
    """
    print("Preparing RAVDESS dataset...")
    data_path = Path(data_dir)
    
    # RAVDESS files follow a naming convention: 
    # Each filename contains information about emotion, intensity, etc.
    # Format: ActorID-SentenceType-Emotion-Intensity-Pitch-Replicate-Misc.wav
    # Emotion codes: 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    # We'll map fearful (06), angry (05), and sad (04) as potential distress vocalizations
    
    audio_files = list(data_path.rglob("*.wav"))
    
    if not audio_files:
        print("No audio files found in RAVDESS dataset!")
        return None
    
    print(f"Found {len(audio_files)} audio files in RAVDESS dataset")
    
    # Create a simple mapping for distress vs non-distress
    metadata = {
        'filename': [],
        'label': [],  # 0 for non-distress, 1 for distress
        'emotion': [],  # emotion code
        'intensity': [],  # intensity code
        'actor_id': [],  # actor ID
        'duration': []  # audio duration
    }
    
    for file_path in audio_files:
        filename = file_path.name
        parts = filename.split('-')
        
        if len(parts) >= 7:
            emotion_code = parts[2]  # 3rd element is emotion
            intensity_code = parts[3]  # 4th element is intensity
            actor_id = parts[0][-2:]  # Last 2 digits of actor ID
            
            # Map emotions to distressed (1) vs non-distressed (0)
            # Consider fearful (06), angry (05), sad (04), surprised (08) as distressed
            if emotion_code in ['04', '05', '06', '08']:  # sad, angry, fearful, surprised
                label = 1  # distress
            elif emotion_code in ['01', '02', '03', '07']:  # neutral, calm, happy, disgust
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
            metadata['intensity'].append(intensity_code)
            metadata['actor_id'].append(actor_id)
            metadata['duration'].append(duration)
        else:
            # If file doesn't match RAVDESS naming convention, treat as unknown
            try:
                duration = librosa.get_duration(path=str(file_path))
            except:
                duration = 0.0
            metadata['filename'].append(str(file_path))
            metadata['label'].append(0)  # default to non-distress
            metadata['emotion'].append('unknown')
            metadata['intensity'].append('unknown')
            metadata['actor_id'].append('unknown')
            metadata['duration'].append(duration)
    
    df = pd.DataFrame(metadata)
    print(f"Prepared metadata for {len(df)} files")
    print(f"Distress vocalizations: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    
    return df


def prepare_crema_dataset(data_dir):
    """
    Prepare the Crowd-Sourced Emotional Mutimodal Actors Dataset (CREMA-D).
    This dataset contains acted emotional expressions with multiple actors.
    """
    print("Preparing CREMA-D dataset...")
    data_path = Path(data_dir)
    
    audio_files = list(data_path.rglob("*.wav"))
    
    if not audio_files:
        print("No audio files found in CREMA-D dataset!")
        return None
    
    print(f"Found {len(audio_files)} audio files in CREMA-D dataset")
    
    # CREMA-D files follow a naming convention:
    # ActorID_EmotionalExpression_VocalIntensity_ActorGender_HispanicEthnicity
    # Emotions: 'ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD'
    # Intensities: 'X', 'L', 'M', 'H' (eXtra, Low, Medium, High)
    
    metadata = {
        'filename': [],
        'label': [],  # 0 for non-distress, 1 for distress
        'emotion': [],  # emotion code
        'intensity': [],  # intensity code
        'actor_id': [],  # actor ID
        'gender': [],  # gender
        'duration': []  # audio duration
    }
    
    for file_path in audio_files:
        filename = file_path.name.replace('.wav', '')
        parts = filename.split('_')
        
        if len(parts) >= 4:
            actor_id = parts[0]
            emotion_code = parts[1]  # ANG, DIS, FEA, HAP, NEU, SAD
            intensity_code = parts[2]  # X, L, M, H
            gender = parts[3]  # F, M
            
            # Map emotions to distressed (1) vs non-distressed (0)
            # Consider Angry (ANG), Disgust (DIS), Fear (FEA), Sad (SAD) as distressed
            if emotion_code in ['ANG', 'DIS', 'FEA', 'SAD']:
                label = 1  # distress
            elif emotion_code in ['HAP', 'NEU']:  # Happy, Neutral
                label = 0  # non-distress
            else:
                label = 0  # default
            
            try:
                duration = librosa.get_duration(path=str(file_path))
            except:
                duration = 0.0  # fallback if librosa fails
            
            metadata['filename'].append(str(file_path))
            metadata['label'].append(label)
            metadata['emotion'].append(emotion_code)
            metadata['intensity'].append(intensity_code)
            metadata['actor_id'].append(actor_id)
            metadata['gender'].append(gender)
            metadata['duration'].append(duration)
        else:
            # If file doesn't match CREMA-D naming convention, treat as unknown
            try:
                duration = librosa.get_duration(path=str(file_path))
            except:
                duration = 0.0
            metadata['filename'].append(str(file_path))
            metadata['label'].append(0)  # default to non-distress
            metadata['emotion'].append('unknown')
            metadata['intensity'].append('unknown')
            metadata['actor_id'].append('unknown')
            metadata['gender'].append('unknown')
            metadata['duration'].append(duration)
    
    df = pd.DataFrame(metadata)
    print(f"Prepared metadata for {len(df)} files")
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
    train_df.to_csv(output_path / "train_metadata.csv", index=False)
    val_df.to_csv(output_path / "val_metadata.csv", index=False)
    test_df.to_csv(output_path / "test_metadata.csv", index=False)
    
    print(f"Dataset split completed:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def download_ravdess(data_dir):
    """Download RAVDESS dataset (this is a placeholder - actual download requires consent)"""
    print("RAVDESS dataset needs to be manually downloaded due to licensing.")
    print("Please visit: https://smartlaboratory.org/ravdess/")
    print("Download the Audio-only files and extract them to:", data_dir)
    return False


def download_crema(data_dir):
    """Download CREMA-D dataset (this is a placeholder - actual download requires consent)"""
    print("CREMA-D dataset needs to be manually downloaded due to licensing.")
    print("Please visit: https://github.com/CheyneyComputerScience/CREMA")
    print("Follow the instructions and extract audio files to:", data_dir)
    return False


def main():
    parser = argparse.ArgumentParser(description='Download and prepare real distress vocalization datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=['ravdess', 'crema', 'all'],
                        help='Dataset to prepare: ravdess, crema, or all')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory to store the dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save prepared metadata (defaults to data-dir)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.data_dir
    
    print(f"Preparing {args.dataset} dataset from {args.data_dir}")
    
    if args.dataset in ['ravdess', 'all']:
        print("="*50)
        print("Preparing RAVDESS dataset...")
        ravdess_df = prepare_ravdess_dataset(args.data_dir)
        if ravdess_df is not None:
            ravdess_split = split_and_save_dataset(ravdess_df, 
                                                  os.path.join(args.output_dir, 'ravdess'))
    
    if args.dataset in ['crema', 'all']:
        print("="*50)
        print("Preparing CREMA-D dataset...")
        crema_df = prepare_crema_dataset(args.data_dir)
        if crema_df is not None:
            crema_split = split_and_save_dataset(crema_df, 
                                                os.path.join(args.output_dir, 'crema'))
    
    print("="*50)
    print("Dataset preparation completed!")
    print("Metadata files saved to:", args.output_dir)


if __name__ == "__main__":
    main()