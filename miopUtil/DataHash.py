import json
import torch
import numpy as np
import hashlib
from scipy.stats import wasserstein_distance

def generate_shape_key(shape_info):
    """Generate a unique key for shape configuration"""
    if not shape_info:
        return None
    result = {}
    shape_key = {}
     
    shape_key["forw"] = shape_info.forw
    shape_key["batchsize"] = shape_info.batchsize
    shape_key["in_channels"] = shape_info.in_channels
    shape_key["in_h"] = shape_info.in_h
    shape_key["in_w"] = shape_info.in_w
    shape_key["in_d"] = shape_info.in_d
    shape_key["out_channels"] = shape_info.out_channels
    shape_key["fil_h"] = shape_info.fil_h
    shape_key["fil_w"] = shape_info.fil_w
    shape_key["fil_d"] = shape_info.fil_d
    shape_key["pad_h"] = shape_info.pad_h
    shape_key["pad_w"] = shape_info.pad_w
    shape_key["pad_d"] = shape_info.pad_d
    shape_key["stride_h"] = shape_info.conv_stride_h
    shape_key["stride_w"] = shape_info.conv_stride_w
    shape_key["stride_d"] = shape_info.conv_stride_d
    shape_key["dilation_h"] = shape_info.dilation_h
    shape_key["dilation_w"] = shape_info.dilation_w
    shape_key["dilation_d"] = shape_info.dilation_d
    shape_key["group_count"] = shape_info.group_count
    shape_key["spatial_dim"] = shape_info.spatial_dim
    
    # Create key string from non-None values
    key_string = "|".join([
        f"{k}:{v}" for k, v in shape_key.items() if v is not None
    ])
    
    key_string_hash = hashlib.sha256(key_string.encode('utf-8')).hexdigest()
    result = {key_string_hash: {"shape": shape_key}}
    
    return result
    

def summarize_conv_output(tensor, include_histogram=False, bins=10):
    """Extract per-channel statistics from convolution output tensor"""
    # Determine channel dimension based on tensor shape
    channel_dim = 0 if tensor.dim() == 4 and tensor.size(0) < 16 else 1
    # print("summarize_conv_output")
    # print(tensor.shape)
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

def check_shape_exists(target_shape_info, all_stats):
    """Check if a specific shape configuration exists in the stats file"""
    try:
        if not target_shape_info:
            return False, None
        
        if isinstance(all_stats, dict):
            if target_shape_info in all_stats:
                return True, all_stats[target_shape_info]
        
        return False, None
    except Exception as e:
        print(f"Error checking shape existence: {e}")
        return False, None

def save_golden_stats(stats, shape_dict, file_path):
    """Save statistics to JSON file"""
    try:
        if stats is None:
            return
        
        try:
            with open(file_path, 'r') as f:
                all_data = json.load(f)
        except FileNotFoundError:
            all_data = {}
        except json.JSONDecodeError:
            print("Warning: Invalid JSON file, creating new one")
            all_data = {}
        
        # Get the first (and should be only) key from shape_dict
        hash_key = list(shape_dict.keys())[0]
        shape_info = shape_dict[hash_key]["shape"]
        
        # Initialize the first level if it doesn't exist
        if hash_key not in all_data:
            all_data[hash_key] = {}
            
        all_data[hash_key]["shape"] = shape_info
        all_data[hash_key]["stats"] = stats
        
        # Save updated data
        with open(file_path, 'w') as f:
            json.dump(all_data, f, indent=2)
            
        print(f"Successfully saved stats to {file_path}")

    except Exception as e:
        print(f"Error saving golden stats: {e}")
        import traceback
        traceback.print_exc()

def load_golden_stats(shape_info, file_path):
    """Load statistics from JSON file"""
    try:
        with open(file_path, 'r') as f:
            all_stats = json.load(f)
        
        if shape_info is None:
            return None
        
        shape_hash_key = list(shape_info.keys())[0]
        
         # Check if the specific shape exists
        if isinstance(all_stats, dict):
            if shape_hash_key in all_stats:
                return True, all_stats[shape_hash_key]["stats"]
        return False, None
        
    except FileNotFoundError:
        print(f"Golden stats shape not found.")
        return False, None

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