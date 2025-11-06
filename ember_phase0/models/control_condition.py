import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
from dataclasses import dataclass


@dataclass
class ControlConfig:
    """Configuration for the text-only control condition"""
    text_embed_dim: int = 128  # As specified in the architecture
    num_classes: int = 2
    max_text_length: int = 50  # Max tokens for text processing


class AudioToTextConverter:
    """
    Converts audio to text using a pre-trained model like Whisper.
    This creates the semantic bottleneck for the control condition.
    """
    def __init__(self, model_name: str = "openai/whisper-tiny.en"):
        """
        Initialize the audio-to-text converter
        
        Args:
            model_name: HuggingFace model name for the ASR system
        """
        # For the control condition, we'll use a lightweight ASR model
        # Whisper-tiny is small enough to not dominate our compute budget
        try:
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.model.eval()
        except:
            # Fallback: use a simple placeholder if Whisper model can't be loaded
            self.processor = None
            self.model = None
            print(f"Warning: Could not load {model_name}, using placeholder converter")
    
    def audio_to_text(self, audio: torch.Tensor) -> str:
        """
        Convert audio tensor to text transcript
        
        Args:
            audio: (batch, time_samples) - raw audio waveform
            
        Returns:
            text: Transcribed text (or placeholder if model failed)
        """
        if self.model is None:
            # Placeholder: return a basic description
            return f"Audio clip of distress vocalization, {audio.shape[1]} samples"
        
        # Process audio through Whisper
        # Audio should be at 16kHz for Whisper
        input_features = self.processor(
            audio.squeeze(0).cpu().numpy() if audio.dim() == 2 else audio.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription


class TextEmbeddingExtractor(nn.Module):
    """
    Extract embeddings from text using a pre-trained model.
    This represents the semantic bottleneck in the control condition.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize text embedding extractor
        
        Args:
            model_name: HuggingFace model name for text embeddings
        """
        super().__init__()
        
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_model = AutoModel.from_pretrained(model_name)
            
            # Projection layer to match our desired embedding dimension
            self.projection = nn.Linear(
                self.text_model.config.hidden_size, 
                128  # Match our specified text embedding dimension
            )
        except:
            # Fallback: simple embedding layer if model fails
            self.tokenizer = None
            self.text_model = None
            self.projection = nn.Linear(300, 128)  # Random dimension for placeholder
            print(f"Warning: Could not load {model_name}, using placeholder text embedder")
    
    def forward(self, text: str) -> torch.Tensor:
        """
        Convert text to embedding
        
        Args:
            text: Input text string
            
        Returns:
            embedding: (text_embed_dim,) - text embedding
        """
        if self.text_model is None:
            # Placeholder: return random embedding
            return torch.randn(128)
        
        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=50
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            # Use mean pooling of last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Project to desired dimension
        projected = self.projection(embeddings)
        
        return projected.squeeze(0)  # Remove batch dimension


class TextOnlyBaseline(nn.Module):
    """
    Text-only baseline model for the control condition.
    This represents the semantic bottleneck approach.
    """
    def __init__(self, config: ControlConfig = None):
        """
        Initialize the text-only baseline
        
        Args:
            config: ControlConfig with model parameters
        """
        super().__init__()
        
        if config is None:
            config = ControlConfig()
        
        self.config = config
        
        # Audio to text converter (semantic bottleneck)
        self.audio_converter = AudioToTextConverter()
        
        # Text embedding extractor
        self.text_embedder = TextEmbeddingExtractor()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.text_embed_dim, config.text_embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.text_embed_dim // 2, config.num_classes)
        )
    
    def forward(self, audio_input: torch.Tensor) -> torch.Tensor:
        """
        Process audio through text bottleneck to classification
        
        Args:
            audio_input: (batch, time_samples) - raw audio waveform
            
        Returns:
            logits: (batch, num_classes) - classification logits
        """
        # Convert audio to text (semantic bottleneck)
        if audio_input.dim() == 2:
            # Process each audio sample in the batch
            text_embeddings = []
            for i in range(audio_input.size(0)):
                text = self.audio_converter.audio_to_text(audio_input[i:i+1])
                text_embed = self.text_embedder(text)
                text_embeddings.append(text_embed)
            text_embeddings = torch.stack(text_embeddings, dim=0)
        else:
            # Single audio input
            text = self.audio_converter.audio_to_text(audio_input)
            text_embeddings = self.text_embedder(text).unsqueeze(0)
        
        # Classify based on text embedding
        logits = self.classifier(text_embeddings)
        
        return logits


class SemanticBottleneckBaseline(nn.Module):
    """
    Alternative implementation focusing on semantic features only.
    This removes temporal/dynamic information by converting to static text descriptions.
    """
    def __init__(self, embed_dim: int = 128, num_classes: int = 2):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Simpler text embedding approach using placeholder features
        # In real implementation, this would use actual text processing
        self.text_feature_extractor = nn.Sequential(
            nn.Linear(50, 256),  # Map from text features to hidden
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def generate_text_features(self, audio_input: torch.Tensor) -> torch.Tensor:
        """
        Generate placeholder text-based features that lose temporal information
        """
        batch_size = audio_input.size(0)
        
        # In a real implementation, this would perform actual text processing
        # For now, we'll create features that don't preserve temporal structure
        # This simulates the "semantic bottleneck" where temporal features are lost
        
        # Extract some basic audio statistics (mean, std, etc.) as "text features"
        # These lose temporal/dynamic information
        features = torch.stack([
            audio_input.mean(dim=1),      # Mean amplitude
            audio_input.std(dim=1),       # Std amplitude  
            torch.abs(audio_input).mean(dim=1),  # Mean absolute value
            (audio_input > 0).float().mean(dim=1),  # Proportion of positive values
        ], dim=1)  # (batch, 4)
        
        # Expand to 50-dim (placeholder for text feature space)
        expanded_features = F.pad(features, (0, 46), "constant", 0)
        
        return expanded_features
    
    def forward(self, audio_input: torch.Tensor) -> torch.Tensor:
        """
        Process audio through semantic bottleneck
        
        Args:
            audio_input: (batch, time_samples) - raw audio waveform
            
        Returns:
            logits: (batch, num_classes) - classification logits
        """
        # Extract semantic features (lose temporal information)
        text_features = self.generate_text_features(audio_input)
        
        # Extract embedding preserving semantic but not temporal info
        text_embed = self.text_feature_extractor(text_features)
        
        # Classify
        logits = self.classifier(text_embed)
        
        return logits


def create_control_model(use_whisper: bool = False) -> nn.Module:
    """
    Factory function to create the appropriate control model
    
    Args:
        use_whisper: Whether to use actual Whisper model (requires download)
        
    Returns:
        Control model instance
    """
    if use_whisper:
        return TextOnlyBaseline()
    else:
        # Use the simpler semantic bottleneck model for initial testing
        return SemanticBottleneckBaseline()


# Example usage and testing
if __name__ == "__main__":
    # Test the control condition model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create control model
    control_model = SemanticBottleneckBaseline(embed_dim=128, num_classes=2)
    control_model.to(device)
    
    # Generate dummy audio input
    dummy_audio = torch.randn(2, 32000, device=device)  # 2 sec at 16kHz
    
    # Forward pass
    logits = control_model(dummy_audio)
    print(f"Control model output shape: {logits.shape}")
    
    # Test with TextOnlyBaseline (will use placeholders if models not available)
    text_control = TextOnlyBaseline()
    text_logits = text_control(dummy_audio)
    print(f"Text-only baseline output shape: {text_logits.shape}")
    
    print("Control condition models created successfully!")