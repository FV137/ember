import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


def compute_temporal_coherence(embeddings: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics for temporal coherence in embeddings
    """
    # Convert to numpy for some metrics if needed
    if isinstance(embeddings, torch.Tensor):
        emb_np = embeddings.detach().cpu().numpy()
    else:
        emb_np = embeddings
    
    metrics = {}
    
    # 1. Temporal continuity - how similar consecutive time steps are
    if len(emb_np.shape) > 2:  # batch, time, features
        # Compute similarity between consecutive time steps
        sims = []
        for b in range(emb_np.shape[0]):
            for t in range(1, emb_np.shape[1]):
                sim = cosine_similarity(
                    emb_np[b, t-1:t].reshape(1, -1), 
                    emb_np[b, t:t+1].reshape(1, -1)
                )[0, 0]
                sims.append(sim)
        metrics['temporal_continuity'] = np.mean(sims)
    else:
        metrics['temporal_continuity'] = 0.0
    
    # 2. Embedding variance over time - should not be too constant
    if len(emb_np.shape) > 2:
        time_variance = np.mean(np.var(emb_np, axis=1))  # Variance across time
        metrics['temporal_variance'] = float(time_variance)
    else:
        metrics['temporal_variance'] = float(np.var(emb_np))
    
    # 3. Gradient smoothness in time
    if len(emb_np.shape) > 2:
        time_grads = np.mean(np.abs(np.diff(emb_np, axis=1)))
        metrics['temporal_gradients'] = float(time_grads)
    else:
        metrics['temporal_gradients'] = 0.0
    
    return metrics


def compute_energy_efficiency(spikes: torch.Tensor) -> Dict[str, float]:
    """
    Compute energy efficiency metrics (for spiking networks)
    """
    if isinstance(spikes, torch.Tensor):
        spike_np = spikes.detach().cpu().numpy()
    else:
        spike_np = spikes
    
    metrics = {}
    
    # 1. Spike rate - energy efficiency is higher with lower spike rates
    spike_rate = np.mean(spike_np)
    metrics['spike_rate'] = float(spike_rate)
    
    # 2. Sparsity - proportion of zero activations
    sparsity = np.mean(spike_np == 0)
    metrics['sparsity'] = float(sparsity)
    
    # 3. Coefficient of variation for spikes
    if spike_np.size > 1:
        mean_spikes = np.mean(spike_np)
        std_spikes = np.std(spike_np)
        if mean_spikes > 0:
            cv = std_spikes / mean_spikes
        else:
            cv = 0.0
        metrics['spike_cv'] = float(cv)
    
    return metrics


def compute_representation_quality(original_audio: torch.Tensor, 
                                 reconstructed_audio: torch.Tensor) -> Dict[str, float]:
    """
    Compute quality metrics for reconstructed audio from embeddings
    For now, this is a placeholder - actual reconstruction would require decoder
    """
    metrics = {}
    
    # In practice, this would compare original to reconstructed audio
    # For now, we'll compute simple statistics
    
    # Audio statistics
    orig_mean = torch.mean(original_audio).item()
    orig_std = torch.std(original_audio).item()
    
    metrics['original_audio_mean'] = orig_mean
    metrics['original_audio_std'] = orig_std
    
    return metrics


def compute_phase_preservation_metrics(embeddings: torch.Tensor, 
                                     original_signal: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics related to phase preservation in embeddings
    """
    metrics = {}
    
    # This is a simplified version - in practice you'd need more sophisticated
    # methods to evaluate phase relationships
    
    # 1. Frequency content preservation (simplified)
    if isinstance(embeddings, torch.Tensor):
        emb_np = embeddings.detach().cpu().numpy()
    else:
        emb_np = embeddings
    
    if isinstance(original_signal, torch.Tensor):
        signal_np = original_signal.detach().cpu().numpy()
    else:
        signal_np = original_signal
    
    # Embedding energy
    emb_energy = np.mean(np.sum(emb_np ** 2, axis=-1))
    metrics['embedding_energy'] = float(emb_energy)
    
    # 2. Temporal structure correlation
    if len(emb_np.shape) > 2:  # batch, time, features
        # Compute temporal variance for each feature
        temp_var = np.mean(np.var(emb_np, axis=1))  # (batch, features)
        metrics['temporal_structure_variability'] = float(np.mean(temp_var))
    
    # 3. Feature diversity
    if len(emb_np.shape) == 3:  # batch, time, features
        # Average across batch and time to get feature statistics
        feature_means = np.mean(emb_np, axis=(0, 1))
        feature_stds = np.std(emb_np, axis=(0, 1))
        metrics['feature_diversity_mean'] = float(np.std(feature_means))
        metrics['feature_diversity_std'] = float(np.mean(feature_stds))
    elif len(emb_np.shape) == 2:  # batch, features
        feature_means = np.mean(emb_np, axis=0)
        feature_stds = np.std(emb_np, axis=0)
        metrics['feature_diversity_mean'] = float(np.std(feature_means))
        metrics['feature_diversity_std'] = float(np.mean(feature_stds))
    
    return metrics


def compute_self_supervised_metrics(embedding: torch.Tensor, 
                                  prediction: torch.Tensor, 
                                  target: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics specifically for self-supervised learning
    """
    metrics = {}
    
    # Prediction accuracy
    pred_mse = F.mse_loss(prediction, target).item()
    metrics['prediction_mse'] = pred_mse
    
    pred_cos_sim = F.cosine_similarity(prediction, target, dim=-1).mean().item()
    metrics['prediction_cosine_similarity'] = pred_cos_sim
    
    # Embedding statistics
    emb_mean = embedding.mean().item()
    emb_std = embedding.std().item()
    metrics['embedding_mean'] = emb_mean
    metrics['embedding_std'] = emb_std
    
    # Prediction vs target statistics consistency
    pred_mean = prediction.mean().item()
    pred_std = prediction.std().item()
    target_mean = target.mean().item()
    target_std = target.std().item()
    
    metrics['prediction_mean'] = pred_mean
    metrics['prediction_std'] = pred_std
    metrics['target_mean'] = target_mean
    metrics['target_std'] = target_std
    
    # Consistency metrics
    metrics['mean_consistency'] = abs(pred_mean - target_mean)
    metrics['std_consistency'] = abs(pred_std - target_std)
    
    return metrics


def compute_all_validation_metrics(batch_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                 audio_input: torch.Tensor,
                                 spikes: torch.Tensor = None) -> Dict[str, float]:
    """
    Compute all validation metrics for the L-Module
    """
    embedding, prediction, target = batch_output
    
    all_metrics = {}
    
    # Self-supervised metrics
    ss_metrics = compute_self_supervised_metrics(embedding, prediction, target)
    all_metrics.update(ss_metrics)
    
    # Temporal coherence metrics
    if len(embedding.shape) > 2:  # If we have temporal dimension
        temporal_metrics = compute_temporal_coherence(embedding)
        all_metrics.update(temporal_metrics)
    
    # Phase preservation metrics
    phase_metrics = compute_phase_preservation_metrics(embedding, audio_input)
    all_metrics.update(phase_metrics)
    
    # Energy efficiency (if spikes are provided)
    if spikes is not None:
        energy_metrics = compute_energy_efficiency(spikes)
        all_metrics.update(energy_metrics)
    
    # Audio quality metrics
    audio_metrics = compute_representation_quality(audio_input, audio_input)  # Placeholder
    all_metrics.update(audio_metrics)
    
    return all_metrics


def aggregate_metrics_across_batches(batch_metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple batches for final evaluation
    """
    if not batch_metrics_list:
        return {}
    
    aggregated = {}
    
    # Get all metric keys
    all_keys = set()
    for metrics in batch_metrics_list:
        all_keys.update(metrics.keys())
    
    # Aggregate each metric
    for key in all_keys:
        values = [batch[key] for batch in batch_metrics_list if key in batch]
        if values:
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
            aggregated[f'{key}_min'] = float(np.min(values))
            aggregated[f'{key}_max'] = float(np.max(values))
    
    return aggregated


# Example usage for testing
if __name__ == "__main__":
    # Test the metrics functions
    print("Testing validation metrics...")
    
    # Create sample data
    batch_size, seq_len, feat_dim = 2, 50, 1024
    embedding = torch.randn(batch_size, seq_len, feat_dim)
    prediction = torch.randn(batch_size, feat_dim)
    target = torch.randn(batch_size, feat_dim)
    audio_input = torch.randn(batch_size, 32000)  # 2 seconds at 16kHz
    spikes = torch.bernoulli(torch.ones(batch_size, seq_len, feat_dim) * 0.1)  # 10% spike rate
    
    batch_output = (embedding, prediction, target)
    
    # Compute all metrics
    all_metrics = compute_all_validation_metrics(batch_output, audio_input, spikes)
    
    print(f"Computed {len(all_metrics)} validation metrics")
    print("Sample metrics:")
    for i, (key, value) in enumerate(all_metrics.items()):
        if i < 10:  # Print first 10
            print(f"  {key}: {value:.4f}")
        else:
            break
    
    # Test batch aggregation
    batch_metrics_list = [all_metrics, all_metrics]  # Use same metrics twice for test
    aggregated = aggregate_metrics_across_batches(batch_metrics_list)
    
    print(f"\nAggregated metrics: {len(aggregated)}")
    print("Sample aggregated metrics:")
    for i, (key, value) in enumerate(aggregated.items()):
        if i < 5:  # Print first 5
            print(f"  {key}: {value:.4f}")
        else:
            break
    
    print("\nValidation metrics working correctly!")