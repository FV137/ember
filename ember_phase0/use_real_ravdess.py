#!/usr/bin/env python3
"""
Script to integrate real RAVDESS dataset from HuggingFace into PROJECT EMBER.
This upgrades from synthetic data to real emotional speech data.
"""
import os
import sys
import pandas as pd
import librosa
import numpy as np
from datasets import load_dataset
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split


def download_ravdess_dataset():
    """Download RAVDESS dataset from HuggingFace Hub."""
    print("Loading RAVDESS dataset from HuggingFace...")
    try:
        # Load the dataset
        dataset = load_dataset("xbgoose/ravdess")
        print(f"Dataset loaded successfully!")
        print(f"Dataset info: {dataset}")
        
        # Show dataset structure
        for split_name, split_dataset in dataset.items():
            print(f"\nSplit: {split_name}")
            print(f"Number of samples: {len(split_dataset)}")
            if len(split_dataset) > 0:
                print(f"Columns: {split_dataset.column_names}")
                print(f"Sample data: {split_dataset[0]}")
        
        return dataset
    except Exception as e:
        print(f"Error loading RAVDESS dataset: {e}")
        return None


def prepare_ravdess_for_ember(dataset):
    """
    Prepare RAVDESS dataset for PROJECT EMBER training.
    RAVDESS naming convention: ActorID-SentenceType-Emotion-Intensity-Pitch-Replicate-Misc.wav
    Emotions: 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    """
    print("\nPreparing RAVDESS dataset for EMBER...")
    
    # We'll work with the train split (usually the only split in this dataset)
    train_split = dataset.get('train')
    if train_split is None:
        train_split = next(iter(dataset.values()))  # Get first available split
    
    # Create directory structure for the data
    data_dir = Path("ember_phase0/data_real_ravdess")
    data_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for organized storage
    audio_dir = data_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    # Prepare metadata dataframe
    filenames = []
    labels = []  # 0 for non-distress, 1 for distress
    emotions = []
    intensities = []
    speakers = []
    durations = []
    
    print(f"Processing {len(train_split)} samples...")
    
    for idx, sample in enumerate(train_split):
        try:
            # Extract filename from the dataset if available
            if 'path' in sample:
                original_filename = Path(sample['path']).name
            elif 'audio' in sample and 'path' in sample['audio']:
                original_filename = Path(sample['audio']['path']).name
            else:
                original_filename = f"audio_{idx:05d}.wav"
            
            # Construct the expected filename to match RAVDESS convention
            # RAVDESS files typically follow: SERxxx.wav where xxx is numeric
            # But actual format might vary
            filename_parts = original_filename.replace('.wav', '').split('-')
            
            # Handle RAVDESS naming convention: ActorID-SentenceType-Emotion-Intensity-Pitch-Replicate-Misc
            # If it follows the RAVDESS pattern: 03-01-08-01-01-01-03-01.wav
            if len(filename_parts) >= 4:
                # Get emotion code (3rd part in typical RAVDESS format)
                emotion_code = filename_parts[2] if len(filename_parts) >= 3 else "01"
            else:
                # Fallback: try to get emotion from metadata if available
                emotion_code = str(sample.get('emotion', '01')).zfill(2)
            
            # Map RAVDESS emotion codes to distressed (1) vs non-distressed (0)
            # Consider fear (06), anger (05), sad (04), surprise (08) as distressed
            if emotion_code in ['04', '05', '06', '08']:  # sad, angry, fearful, surprised
                label = 1  # distress
            elif emotion_code in ['01', '02', '03', '07']:  # neutral, calm, happy, disgust
                label = 0  # non-distress
            else:
                label = 0  # default to non-distress
            
            # Get intensity if available (typically 4th part in RAVDESS format)
            intensity_code = filename_parts[3] if len(filename_parts) >= 4 else "01"
            
            # Get speaker ID (first part in RAVDESS format)
            speaker_id = filename_parts[0] if len(filename_parts) >= 1 else "unknown"
            
            # Handle audio loading
            audio_data = None
            sample_rate = 16000  # Target sample rate
            
            if 'audio' in sample:
                audio_info = sample['audio']
                if 'array' in audio_info:
                    audio_data = audio_info['array']
                    # Resample if necessary
                    actual_sr = audio_info.get('sampling_rate', sample_rate)
                    if actual_sr != sample_rate:
                        # Resample audio to target sample rate
                        import torch
                        audio_tensor = torch.tensor(audio_data).unsqueeze(0)
                        resampler = torchaudio.transforms.Resample(orig_freq=actual_sr, new_freq=sample_rate)
                        audio_data = resampler(audio_tensor).squeeze(0).numpy()
                elif 'path' in audio_info:
                    # Load from file path
                    audio_path = audio_info['path']
                    audio_data, actual_sr = librosa.load(audio_path, sr=sample_rate)
            
            # Save audio file
            output_filename = f"ravdess_{idx:05d}_{speaker_id}_emo{emotion_code}_int{intensity_code}_" + \
                             ("distress" if label == 1 else "neutral") + ".wav"
            output_path = audio_dir / output_filename
            
            if audio_data is not None:
                import soundfile as sf
                sf.write(str(output_path), audio_data, sample_rate)
                duration = len(audio_data) / sample_rate
            else:
                # Create a placeholder (this shouldn't happen with proper dataset)
                print(f"Warning: No audio data for sample {idx}")
                continue
            
            # Add to metadata
            filenames.append(str(output_path))
            labels.append(label)
            emotions.append(emotion_code)
            intensities.append(intensity_code)
            speakers.append(speaker_id)
            durations.append(duration)
            
            if idx % 100 == 0:
                print(f"Processed {idx+1}/{len(train_split)} samples...")
        
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Create metadata dataframe
    df = pd.DataFrame({
        'filename': filenames,
        'label': labels,
        'emotion': emotions,
        'intensity': intensities,
        'speaker': speakers,
        'duration': durations
    })
    
    print(f"\nPrepared {len(df)} samples for EMBER training")
    print(f"Distress samples: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"Non-distress samples: {len(df) - df['label'].sum()}")
    
    return df, data_dir


def split_and_save_metadata(df, data_dir):
    """Split the dataset and save metadata files."""
    print("\nSplitting dataset into train/val/test...")
    
    # Split the data (80/10/10)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    # Save metadata files
    train_df.to_csv(data_dir / "train_metadata.csv", index=False)
    val_df.to_csv(data_dir / "val_metadata.csv", index=False)
    test_df.to_csv(data_dir / "test_metadata.csv", index=False)
    
    print(f"Dataset splits created:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def create_summary(df, data_dir):
    """Create a summary of the prepared dataset."""
    summary = {
        'total_samples': len(df),
        'distress_samples': int(df['label'].sum()),
        'non_distress_samples': int(len(df) - df['label'].sum()),
        'distress_percentage': float(df['label'].mean() * 100),
        'unique_emotions': df['emotion'].nunique(),
        'emotions_distribution': df['emotion'].value_counts().to_dict(),
        'unique_speakers': df['speaker'].nunique(),
        'avg_duration': float(df['duration'].mean()),
        'min_duration': float(df['duration'].min()),
        'max_duration': float(df['duration'].max())
    }
    
    # Save summary
    import json
    with open(data_dir / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDataset Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def main():
    print("Upgrading PROJECT EMBER to use real RAVDESS dataset...")
    
    # Download and load RAVDESS dataset
    dataset = download_ravdess_dataset()
    if dataset is None:
        print("Failed to load RAVDESS dataset!")
        return 1
    
    # Prepare dataset for EMBER
    df, data_dir = prepare_ravdess_for_ember(dataset)
    if df is None or len(df) == 0:
        print("Failed to prepare RAVDESS dataset!")
        return 1
    
    # Split and save metadata
    train_df, val_df, test_df = split_and_save_metadata(df, data_dir)
    
    # Create summary
    create_summary(df, data_dir)
    
    print(f"\n" + "="*60)
    print("RAVDESS dataset preparation completed!")
    print(f"Data saved to: {data_dir}")
    print(f"You can now use this data directory in your EMBER training:")
    print(f"python train_binding_test.py --data-dir {data_dir}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())