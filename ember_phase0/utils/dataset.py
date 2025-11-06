import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import os
from sklearn.model_selection import train_test_split
import requests
from pathlib import Path

# Import our synthetic audio generation
from .synthetic_audio import create_synthetic_dataset


class DistressVocalizationDataset(Dataset):
    """
    Dataset for distinguishing genuine vs acted distress vocalizations.
    This is the core dataset for the binding test.
    """
    def __init__(self, 
                 data_dir: str,
                 max_duration: float = 2.0,  # seconds
                 sample_rate: int = 16000,
                 mode: str = 'train',
                 test_size: float = 0.2,
                 val_size: float = 0.1):
        """
        Initialize the dataset
        
        Args:
            data_dir: Directory containing audio files
            max_duration: Maximum duration of audio clips in seconds
            sample_rate: Sample rate for audio processing
            mode: 'train', 'val', or 'test'
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
        """
        self.data_dir = Path(data_dir)
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        
        # Load metadata and split data
        self.metadata = self._load_metadata()
        self.train_idx, self.val_idx, self.test_idx = self._split_data(
            len(self.metadata), test_size, val_size
        )
        
        # Select indices based on mode
        if mode == 'train':
            self.indices = self.train_idx
        elif mode == 'val':
            self.indices = self.val_idx
        elif mode == 'test':
            self.indices = self.test_idx
        else:
            raise ValueError(f"Mode must be 'train', 'val', or 'test', got {mode}")
    
    def _load_metadata(self) -> pd.DataFrame:
        """
        Load metadata about audio files.
        In a real implementation, this would load from a CSV or similar.
        For now, we'll create synthetic metadata to simulate the dataset.
        """
        # Simulate loading metadata
        # In a real implementation, you'd load from actual files
        metadata = {
            'filename': [],
            'label': [],  # 0 for acted, 1 for genuine
            'source': [],  # dataset source
            'duration': []
        }
        
        # Look for actual audio files in the directory
        audio_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.aac')
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(list(self.data_dir.rglob(f'*{ext}')))
            audio_files.extend(list(self.data_dir.rglob(f'*{ext.upper()}')))
        
        if audio_files:
            # Load actual files if found
            for file_path in audio_files:
                try:
                    # Load audio to get duration
                    duration = librosa.get_duration(path=str(file_path))
                    
                    # Determine label based on filename or directory structure
                    filename_lower = file_path.name.lower()
                    
                    # Check for RAVDESS naming convention (e.g., 03-01-08-01-01-01-03-01.wav)
                    parts = file_path.name.lower().replace('.wav', '').replace('.mp3', '').replace('.flac', '').replace('.m4a', '').replace('.aac', '').split('-')
                    if len(parts) >= 3 and all(part.isdigit() for part in parts[:3]):
                        # RAVDESS format: ActorID-SentenceType-Emotion-Intensity-Pitch-Replicate
                        # 3rd part is emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
                        emotion_code = parts[2].zfill(2)  # Ensure 2-digit format
                        # Map RAVDESS emotion codes: Consider fear (06), anger (05), sad (04), disgust (07), surprise (08) as distressed
                        if emotion_code in ['04', '05', '06', '07', '08']:
                            label = 1  # distressed
                        else:
                            label = 0  # non-distressed
                    elif 'genuine' in filename_lower or 'actual' in filename_lower or 'real' in filename_lower:
                        label = 1  # genuine/distressed
                    elif 'acted' in filename_lower or 'scripted' in filename_lower or 'fake' in filename_lower or 'neutral' in filename_lower:
                        label = 0  # acted/non-distressed
                    else:
                        # Default to 50/50 split if not specified - alternate between 0 and 1
                        label = len(metadata['filename']) % 2  # alternate to ensure balance
                    
                    metadata['filename'].append(str(file_path))
                    metadata['label'].append(label)
                    metadata['source'].append(file_path.parent.name)
                    metadata['duration'].append(duration)
                    
                    metadata['filename'].append(str(file_path))
                    metadata['label'].append(label)
                    metadata['source'].append(file_path.parent.name)
                    metadata['duration'].append(duration)
                    
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
        else:
            # Create synthetic data for testing purposes
            print("No audio files found, creating synthetic dataset for testing...")
            
            # Create synthetic distress vocalization dataset
            create_synthetic_dataset(self.data_dir, num_samples=200)
            
            # Reload metadata after creating synthetic files
            audio_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.aac')
            audio_files = []
            
            for ext in audio_extensions:
                audio_files.extend(list(self.data_dir.rglob(f'*{ext}')))
                audio_files.extend(list(self.data_dir.rglob(f'*{ext.upper()}')))
            
            # Load the synthetic files
            for file_path in audio_files:
                if 'metadata.csv' in str(file_path):  # Skip metadata file
                    continue
                    
                try:
                    # Load audio to get duration
                    duration = librosa.get_duration(path=str(file_path))
                    
                    # Determine label based on filename
                    filename_lower = file_path.name.lower()
                    if 'genuine' in filename_lower:
                        label = 1  # genuine
                    elif 'acted' in filename_lower:
                        label = 0  # acted
                    else:
                        # Default to 50/50 split based on index if not specified
                        label = 0  # arbitrary default
                    
                    metadata['filename'].append(str(file_path))
                    metadata['label'].append(label)
                    metadata['source'].append('synthetic')
                    metadata['duration'].append(duration)
                    
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
        
        return pd.DataFrame(metadata)
    
    def _split_data(self, n_samples: int, test_size: float, val_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets
        """
        indices = np.arange(n_samples)
        
        # First split: separate test set
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=self.metadata['label']
        )
        
        # Second split: separate validation from train
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_size/(1-test_size), 
            random_state=42, stratify=self.metadata.iloc[train_val_idx]['label']
        )
        
        return train_idx, val_idx, test_idx
    
    def _load_audio(self, filepath: str) -> np.ndarray:
        """
        Load and preprocess audio file
        """
        try:
            # Load audio
            audio, sr = librosa.load(filepath, sr=self.sample_rate)
            
            # Trim or pad to max duration
            if len(audio) > self.max_samples:
                # Randomly select a window if too long
                start_idx = np.random.randint(0, len(audio) - self.max_samples + 1)
                audio = audio[start_idx:start_idx + self.max_samples]
            elif len(audio) < self.max_samples:
                # Pad with zeros if too short
                audio = np.pad(audio, (0, self.max_samples - len(audio)), mode='constant')
            
            return audio
        
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Return silence if failed
            return np.zeros(self.max_samples)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset
        
        Returns:
            audio_tensor: (time_samples,) - raw audio waveform
            label: 0 or 1 (acted or genuine distress)
        """
        # Get actual index
        actual_idx = self.indices[idx]
        
        # Get metadata
        row = self.metadata.iloc[actual_idx]
        filepath = row['filename']
        label = int(row['label'])
        
        # Load audio
        audio = self._load_audio(filepath)
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        
        return audio_tensor, label


class TextBaselineDataset(Dataset):
    """
    Dataset for the text-only baseline (control condition).
    Converts audio to text descriptions for comparison.
    """
    def __init__(self, 
                 data_dir: str,
                 max_duration: float = 2.0,
                 sample_rate: int = 16000,
                 mode: str = 'train',
                 test_size: float = 0.2,
                 val_size: float = 0.1):
        """
        Initialize the text baseline dataset
        """
        # Use the same underlying data as the audio dataset
        self.audio_dataset = DistressVocalizationDataset(
            data_dir, max_duration, sample_rate, mode, test_size, val_size
        )
        
        # Placeholder for text embeddings (in real implementation, these would come from Whisper)
        self.text_embeddings = self._generate_placeholder_text_embeddings()
    
    def _generate_placeholder_text_embeddings(self):
        """
        Generate placeholder text embeddings.
        In a real implementation, these would come from Whisper transcription + embedding.
        """
        embeddings = []
        for i in range(len(self.audio_dataset)):
            # Create placeholder text embedding (128-dim as specified in config)
            # In real implementation, this would be from actual text processing
            embedding = torch.randn(128)  # Placeholder random embedding
            embeddings.append(embedding)
        return embeddings
    
    def __len__(self):
        return len(self.audio_dataset)
    
    def __getitem__(self, idx):
        """
        Get a sample (returns audio and label for the control model)
        The control model will internally process the audio
        """
        # Get audio and label from underlying dataset
        audio, label = self.audio_dataset[idx]
        
        return audio, label


def create_dataloaders(data_dir: str, 
                      batch_size: int = 16,
                      num_workers: int = 4,
                      sample_rate: int = 16000) -> dict:
    """
    Create data loaders for train, validation, and test sets
    
    Returns:
        Dictionary with 'train', 'val', 'test' data loaders
    """
    # Create datasets
    train_dataset = DistressVocalizationDataset(
        data_dir, sample_rate=sample_rate, mode='train'
    )
    val_dataset = DistressVocalizationDataset(
        data_dir, sample_rate=sample_rate, mode='val'
    )
    test_dataset = DistressVocalizationDataset(
        data_dir, sample_rate=sample_rate, mode='test'
    )
    
    # Create data loaders
    dataloaders = {
        'train': DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    print(f"Dataset sizes: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    return dataloaders


def create_text_dataloaders(data_dir: str, 
                           batch_size: int = 16,
                           num_workers: int = 4,
                           sample_rate: int = 16000) -> dict:
    """
    Create data loaders for text-only baseline
    """
    # Create text datasets
    train_dataset = TextBaselineDataset(
        data_dir, sample_rate=sample_rate, mode='train'
    )
    val_dataset = TextBaselineDataset(
        data_dir, sample_rate=sample_rate, mode='val'
    )
    test_dataset = TextBaselineDataset(
        data_dir, sample_rate=sample_rate, mode='test'
    )
    
    # Create data loaders
    dataloaders = {
        'train': DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    print(f"Text dataset sizes: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    return dataloaders


# Utility function to download common datasets
def download_example_datasets(data_dir: str):
    """
    Download example datasets that could be used for this task
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    print("Example datasets for distress vocalization detection:")
    print("- RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song")
    print("- MSP-IMPROV: Multimodal Stylistic Speech Improvisation Corpus")
    print("- CREMA-D: Crowd-Sourced Emotional Mutimodal Actors Dataset")
    print("\nYou can download these manually or use existing audio datasets.")
    
    # In a real implementation, you could add code to download these datasets
    # but for now we'll just inform the user where to find them


if __name__ == "__main__":
    # Test the dataset
    import os
    os.makedirs("ember_phase0/data", exist_ok=True)
    
    # Create a simple test
    dataset = DistressVocalizationDataset("ember_phase0/data", mode='train')
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        audio, label = dataset[0]
        print(f"Audio shape: {audio.shape}")
        print(f"Label: {label}")
        print(f"Expected audio length: {dataset.max_samples} samples")
    
    # Create dataloaders
    dataloaders = create_dataloaders("ember_phase0/data", batch_size=4)
    batch_audio, batch_labels = next(iter(dataloaders['train']))
    print(f"Batch audio shape: {batch_audio.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")