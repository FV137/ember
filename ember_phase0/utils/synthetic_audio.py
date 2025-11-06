import torch
import numpy as np
from scipy import signal
import librosa


def generate_distress_vocalization(duration=2.0, sample_rate=16000, genuine=True):
    """
    Generate synthetic distress vocalization audio
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate (default 16000Hz)
        genuine: True for genuine distress, False for acted/distorted
    
    Returns:
        audio: Generated audio signal
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    if genuine:
        # Genuine distress: more structured vocal patterns with natural variability
        # Base frequency sweep (emotional distress often has frequency variations)
        base_freq_min = 100  # Hz
        base_freq_max = 500  # Hz
        
        # Create frequency sweep
        freq_sweep = base_freq_min + (base_freq_max - base_freq_min) * (t / duration)
        
        # Add natural micro-variations in pitch
        freq_sweep += 10 * np.sin(2 * np.pi * 2 * t)  # Add 2Hz pitch modulation
        
        # Create a more complex harmonic structure
        signal_gen = np.zeros_like(t)
        
        # Fundamental and harmonics
        for i in range(1, 6):  # First 5 harmonics
            harmonic_freq = freq_sweep * i
            amplitude = 1.0 / i  # Decreasing amplitude for higher harmonics
            phase_noise = 0.1 * np.random.random()  # Slight phase randomness
            signal_gen += amplitude * np.sin(2 * np.pi * np.cumsum(harmonic_freq) / sample_rate + phase_noise)
        
        # Add breathing-like amplitude modulation
        amp_env = 1 + 0.3 * np.sin(2 * np.pi * 0.5 * t)  # 0.5Hz breathing rate
        
        # Add emotional tremor
        tremor = 1 + 0.2 * np.sin(2 * np.pi * 7 * t)  # 7Hz emotional tremor
        
        # Apply modulations
        signal_gen = signal_gen * amp_env * tremor
        
        # Add some natural noise
        noise_level = 0.1
        signal_gen += noise_level * np.random.normal(0, 1, len(signal_gen))
        
    else:
        # Acted/distorted distress: more irregular patterns
        # More erratic frequency changes
        base_freq_min = 80  # Hz
        base_freq_max = 600  # Hz
        
        # More erratic frequency path
        freq_path = np.random.uniform(base_freq_min, base_freq_max, size=int(duration * 10))
        freq_path = np.interp(t, np.linspace(0, duration, len(freq_path)), freq_path)
        
        # Create signal with more distortion
        signal_gen = np.zeros_like(t)
        
        # Multiple overlapping sources (more chaotic)
        for i in range(1, 4):  # Fewer harmonics, more chaotic
            harmonic_freq = freq_path * i
            amplitude = np.random.uniform(0.5, 1.5) / i
            phase_noise = np.random.random() * 2 * np.pi
            signal_gen += amplitude * np.sin(2 * np.pi * np.cumsum(harmonic_freq) / sample_rate + phase_noise)
        
        # More erratic amplitude changes
        amp_env = 1 + 0.5 * np.sin(2 * np.pi * np.random.uniform(0.3, 2.0) * t)
        
        # More chaotic tremor
        tremor_freq = np.random.uniform(5, 12)  # More variable tremor
        tremor = 1 + 0.3 * np.sin(2 * np.pi * tremor_freq * t)
        
        # Apply modulations
        signal_gen = signal_gen * amp_env * tremor
        
        # Add more noise to simulate acted nature
        noise_level = 0.2
        signal_gen += noise_level * np.random.normal(0, 1, len(signal_gen))
    
    # Normalize to prevent clipping
    signal_gen = signal_gen / np.max(np.abs(signal_gen)) * 0.8  # Leave headroom
    
    # Apply low-pass filter to make it more realistic
    b, a = signal.butter(6, 4000 / (sample_rate / 2), btype='low')
    signal_gen = signal.filtfilt(b, a, signal_gen)
    
    return signal_gen


def create_synthetic_dataset(data_dir, num_samples=200, sample_rate=16000):
    """
    Create a synthetic distress vocalization dataset
    
    Args:
        data_dir: Directory to save audio files
        num_samples: Number of samples to create
        sample_rate: Sample rate for the audio
    """
    import os
    from pathlib import Path
    
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # Create metadata list
    metadata_lines = ["filename,label,duration\n"]  # CSV header
    
    for i in range(num_samples):
        # Alternate between genuine and acted
        genuine = i % 2 == 0
        
        # Generate audio
        audio = generate_distress_vocalization(genuine=genuine)
        
        # Create filename
        label_str = "genuine" if genuine else "acted"
        filename = f"distress_{i:03d}_{label_str}.wav"
        filepath = data_path / filename
        
        # Save audio file using soundfile since librosa.output.write_wav is deprecated
        import soundfile as sf
        sf.write(str(filepath), audio, sample_rate)
        
        # Add to metadata
        duration = len(audio) / sample_rate
        metadata_lines.append(f"{filename},{int(genuine)},{duration}\n")
    
    # Save metadata
    metadata_path = data_path / "metadata.csv"
    with open(metadata_path, 'w') as f:
        f.writelines(metadata_lines)
    
    print(f"Created {num_samples} synthetic distress vocalization samples in {data_dir}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    # Test the synthetic audio generation
    import matplotlib.pyplot as plt
    
    # Generate genuine distress vocalization
    genuine_audio = generate_distress_vocalization(genuine=True)
    
    # Generate acted/distorted distress vocalization
    acted_audio = generate_distress_vocalization(genuine=False)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(genuine_audio[:2000])  # First 2000 samples
    plt.title("Genuine Distress Vocalization")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    
    plt.subplot(2, 1, 2)
    plt.plot(acted_audio[:2000])  # First 2000 samples
    plt.title("Acted Distress Vocalization")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    
    plt.tight_layout()
    plt.savefig("distress_comparison.png")
    plt.show()
    
    print("Comparison plot saved as 'distress_comparison.png'")
    print(f"Genuine audio length: {len(genuine_audio)} samples")
    print(f"Acted audio length: {len(acted_audio)} samples")