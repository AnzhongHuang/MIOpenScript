import json
import torch
import numpy as np
from scipy.stats import wasserstein_distance

def summarize_conv_output(tensor, include_histogram=False, bins=10):
    """Extract per-channel statistics from convolution output tensor"""
    # Determine channel dimension based on tensor shape
    channel_dim = 0 if tensor.dim() == 4 and tensor.size(0) < 16 else 1
    print("summarize_conv_output")
    print(tensor.shape)
    # Move channel dimension to front and flatten
    tensor = tensor.movedim(channel_dim, 0)

    flat_tensor = tensor.reshape(1, -1)
    # print(flat_tensor.shape)
    stats = []
    for i in range(1):
        channel_data = flat_tensor[i].to(torch.float32).detach().cpu().numpy()
        channel_stats = {
            "min": float(channel_data.min()),
            "max": float(channel_data.max()),
            "mean": float(channel_data.mean()),
            "std": float(channel_data.std()),
            "median": float(np.median(channel_data))
        }
        
        if include_histogram:
            hist, bin_edges = np.histogram(channel_data, bins=bins, density=True)
            channel_stats["histogram"] = {
                "values": hist.tolist(),
                "edges": bin_edges.tolist()
            }
        stats.append(channel_stats)
    return stats


def save_golden_stats(stats, file_path):
    """Save statistics to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(stats, f, indent=2)

def load_golden_stats(file_path):
    """Load statistics from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Golden stats file {file_path} not found.")
        return None

def compare_stats(golden_stats, test_stats, tolerance=0.05):
    """Compare two sets of statistics with tolerance threshold"""
    if len(golden_stats) != len(test_stats):
        raise ValueError("Channel count mismatch")
    
    channel_errors = []
    for ch, (g, t) in enumerate(zip(golden_stats, test_stats)):
        errors = []
        for metric in ['min', 'max', 'mean', 'std', 'median']:
            if metric in g and metric in t:
                diff = abs(g[metric] - t[metric])
                scale = max(abs(g[metric]), 1e-6)  # Avoid division by zero
                errors.append(diff / scale)
        
        # Compare histograms if available
        if 'histogram' in g and 'histogram' in t:
            hist_dist = wasserstein_distance(
                g['histogram']['values'],
                t['histogram']['values']
            )
            errors.append(hist_dist)
        
        channel_errors.append(max(errors))
    
    max_error = max(channel_errors)
    return max_error <= tolerance, max_error, channel_errors