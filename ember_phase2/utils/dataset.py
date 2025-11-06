import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import os
from pathlib import Path
import random


class AudioDataset(Dataset):
    """
    Dataset for audio files to train the L-Module
    """
    def __init__(self,
                 data_dir: str,
                 sample_rate: int = 16000,
                 audio_length: float = 2.0,  # in seconds
                 transform=None,
                 file_extensions: List[str] = None):
        """
        Args:
            data_dir: Directory containing audio files
            sample_rate: Target sample rate for audio
            audio_length: Length of audio clips in seconds
            transform: Optional transforms to apply
            file_extensions: List of valid audio file extensions
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.transform = transform
        self.audio_length_samples = int(sample_rate * audio_length)
        
        if file_extensions is None:
            file_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
        
        # Find all audio files
        self.audio_files = []
        for ext in file_extensions:
            self.audio_files.extend(list(self.data_dir.rglob(f'*{ext}')))
            self.audio_files.extend(list(self.data_dir.rglob(f'*{ext.upper()}')))
        
        print(f"Found {len(self.audio_files)} audio files in {data_dir}")
        
        if len(self.audio_files) == 0:
            print(f"Warning: No audio files found in {data_dir}")
            # Create synthetic data for testing
            self.audio_files = [None] * 100  # Placeholder for synthetic data
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def _load_audio(self, file_path: Path) -> np.ndarray:
        """
        Load and preprocess audio file
        """
        if file_path is None:  # Synthetic data case
            # Return synthetic audio
            duration_samples = self.audio_length_samples
            # Create synthetic audio with some structure (not pure noise)
            t = np.linspace(0, self.audio_length, duration_samples)
            # Mix of multiple frequencies with some temporal structure
            audio = (0.3 * np.sin(2 * np.pi * 200 * t) + 
                    0.2 * np.sin(2 * np.pi * 400 * t) + 
                    0.1 * np.sin(2 * np.pi * 800 * t) +
                    0.4 * np.random.randn(duration_samples))
            return audio
        
        # Load real audio file
        try:
            audio, sr = librosa.load(str(file_path), sr=self.sample_rate)
            
            # Ensure proper length
            if len(audio) >= self.audio_length_samples:
                # Random crop for training augmentation
                start_idx = random.randint(0, len(audio) - self.audio_length_samples)
                audio = audio[start_idx:start_idx + self.audio_length_samples]
            else:
                # Pad with zeros if too short
                pad_length = self.audio_length_samples - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
            
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zeros if there's an error
            return np.zeros(self.audio_length_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """
        Returns audio tensor
        """
        audio_path = self.audio_files[idx]
        audio = self._load_audio(audio_path)
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        
        if self.transform:
            audio_tensor = self.transform(audio_tensor)
        
        return audio_tensor,


class AudioAugmentation:
    """
    Simple audio augmentations for self-supervised training
    """
    def __init__(self, 
                 time_stretch_range: Tuple[float, float] = (0.9, 1.1),
                 pitch_shift_range: Tuple[int, int] = (-2, 2),
                 noise_factor: float = 0.001):
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
        self.noise_factor = noise_factor
    
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        # Add small amount of noise
        noise = torch.randn_like(audio) * self.noise_factor
        audio = audio + noise
        
        # Clip values to prevent overflow
        audio = torch.clamp(audio, -1.0, 1.0)
        
        return audio


def create_synthetic_dataset(data_dir: str, num_samples: int = 1000, sample_rate: int = 16000):
    """
    Create a synthetic audio dataset for testing
    """
    import soundfile as sf
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_samples} synthetic audio samples...")
    
    for i in range(num_samples):
        # Create synthetic audio with different characteristics
        duration = 2.0  # seconds
        length = int(sample_rate * duration)
        
        # Create audio with multiple components (frequency sweep, harmonics, etc.)
        t = np.linspace(0, duration, length)
        
        # Base signal with varying frequency
        freq_base = 100 + 300 * (i % 10)  # Vary base frequency
        signal = np.sin(2 * np.pi * freq_base * t)
        
        # Add harmonics
        for harmonic in range(2, 5):
            signal += 0.3 * np.sin(2 * np.pi * freq_base * harmonic * t)
        
        # Add some temporal variation
        envelope = np.exp(-5 * (t - duration/2)**2)  # Gaussian envelope
        signal *= envelope
        
        # Add small amount of noise
        signal += 0.01 * np.random.randn(len(signal))
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.8  # Leave headroom
        
        # Save audio file
        filename = data_path / f"synthetic_{i:04d}.wav"
        sf.write(str(filename), signal, sample_rate)
        
        if i % 100 == 0:
            print(f"Generated {i}/{num_samples} samples...")
    
    print(f"Synthetic dataset created in {data_dir}")


def get_dataloaders(data_dir: str, 
                   batch_size: int = 8,
                   sample_rate: int = 16000,
                   audio_length: float = 2.0,
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1,
                   num_workers: int = 2):
    """
    Create train, validation, and test dataloaders
    """
    # Create dataset
    dataset = AudioDataset(
        data_dir=data_dir,
        sample_rate=sample_rate,
        audio_length=audio_length
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Dataset split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    import tempfile
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing dataset in {temp_dir}")
        
        # Create synthetic data
        create_synthetic_dataset(temp_dir, num_samples=20)
        
        # Create dataset
        dataset = AudioDataset(temp_dir)
        
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            audio_sample, = dataset[0]
            print(f"Audio sample shape: {audio_sample.shape}")
            print(f"Expected length: {16000 * 2} samples (2 seconds at 16kHz)")
        
        # Test dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(temp_dir, batch_size=4)
        
        batch, = next(iter(train_loader))
        print(f"Batch shape: {batch.shape}")
        
        print("Dataset testing completed successfully!")