import json
import torch
import numpy as np
import hashlib
import os
import time
import threading
from scipy.stats import wasserstein_distance

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import args_2_shape

database_lock = threading.Lock()
# Global thread-safe file handler instances
_file_handlers = {}
_file_handlers_lock = threading.Lock()

def get_thread_safe_handler(file_path):
    """Get or create thread-safe file handler for given path"""
    with _file_handlers_lock:
        if file_path not in _file_handlers:
            _file_handlers[file_path] = ThreadSafeJSONFile(file_path)
        return _file_handlers[file_path]

class ThreadSafeJSONFile:
    """Thread-safe JSON file handler"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.lock_path = f"{file_path}.lock"
        self._local_lock = threading.Lock()
    
    def acquire_file_lock(self, timeout=30):
        """Acquire exclusive file lock"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(lock_fd, f"{os.getpid()}:{threading.get_ident()}".encode())
                return lock_fd
            except OSError:
                time.sleep(0.01)
        raise TimeoutError(f"Could not acquire file lock for {self.file_path}")
    
    def release_file_lock(self, lock_fd):
        """Release file lock"""
        try:
            os.close(lock_fd)
            os.remove(self.lock_path)
        except OSError:
            pass
    
    def read_json(self):
        """Thread-safe JSON read"""
        with self._local_lock:
            if not os.path.exists(self.file_path):
                return {}
            
            lock_fd = self.acquire_file_lock()
            try:
                # Check file size before reading
                file_size = os.path.getsize(self.file_path)
                if file_size > 100 * 1024 * 1024:  # 100MB
                    print(f"Warning: {self.file_path} is very large ({file_size} bytes)")
                
                with open(self.file_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        # Validate JSON structure
                        data = json.loads(content)
                        if not isinstance(data, dict):
                            print(f"Warning: Root element is not a dictionary in {self.file_path}")
                            return {}
                        return data
                    return {}
            except json.JSONDecodeError as e:
                print(f"JSON decode error in {self.file_path}: {e}")
                self._backup_corrupted_file()
                return {}
            except Exception as e:
                print(f"Error reading {self.file_path}: {e}")
                return {}
            finally:
                self.release_file_lock(lock_fd)
    
    def write_json(self, data):
        """Thread-safe JSON write with atomic operation"""
        with self._local_lock:
            lock_fd = self.acquire_file_lock()
            try:
                import tempfile
                # Create temp file in same directory for atomic rename
                temp_fd, temp_path = tempfile.mkstemp(
                    suffix='.tmp', 
                    prefix=f"{os.path.basename(self.file_path)}_",
                    dir=os.path.dirname(self.file_path) or '.'
                )
                
                try:
                    with os.fdopen(temp_fd, 'w') as temp_file:
                        json.dump(data, temp_file, indent=2, ensure_ascii=False)
                        temp_file.flush()
                        os.fsync(temp_file.fileno())
                    
                    # Atomic rename - this is the key to thread safety
                    os.rename(temp_path, self.file_path)
                    
                except Exception as e:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                    raise e
                    
            finally:
                self.release_file_lock(lock_fd)
    
    def _backup_corrupted_file(self):
        """Backup corrupted file"""
        if os.path.exists(self.file_path):
            backup_path = f"{self.file_path}.corrupted.{int(time.time())}"
            try:
                os.rename(self.file_path, backup_path)
                print(f"Corrupted file backed up to: {backup_path}")
            except OSError:
                pass

def generate_shape_key(shape_info, is_solver=False):
    """Generate a unique key for shape configuration"""
    if not shape_info:
        return None
    
    result = args_2_shape.Solve(None, is_solver, shape_info)
    # shape_key = {}
    # shape_key["forw"] = shape_info.forw
    # shape_key["batchsize"] = shape_info.batchsize
    # shape_key["in_channels"] = shape_info.in_channels
    # shape_key["in_h"] = shape_info.in_h
    # shape_key["in_w"] = shape_info.in_w
    # shape_key["in_d"] = shape_info.in_d
    # shape_key["out_channels"] = shape_info.out_channels
    # shape_key["fil_h"] = shape_info.fil_h
    # shape_key["fil_w"] = shape_info.fil_w
    # shape_key["fil_d"] = shape_info.fil_d
    # shape_key["pad_h"] = shape_info.pad_h
    # shape_key["pad_w"] = shape_info.pad_w
    # shape_key["pad_d"] = shape_info.pad_d
    # shape_key["stride_h"] = shape_info.conv_stride_h
    # shape_key["stride_w"] = shape_info.conv_stride_w
    # shape_key["stride_d"] = shape_info.conv_stride_d
    # shape_key["dilation_h"] = shape_info.dilation_h
    # shape_key["dilation_w"] = shape_info.dilation_w
    # shape_key["dilation_d"] = shape_info.dilation_d
    # shape_key["group_count"] = shape_info.group_count
    # shape_key["spatial_dim"] = shape_info.spatial_dim
    
    # # Create key string from non-None values
    # key_string = "|".join([
    #     f"{k}:{v}" for k, v in shape_key.items() if v is not None
    # ])
    
    # key_string_hash = hashlib.sha256(key_string.encode('utf-8')).hexdigest()[:16]
    # result = {key_string_hash: {"shape": shape_key}}
    
    return result

def summarize_conv_output(tensor, include_histogram=False, bins=10):
    """Extract per-channel statistics from convolution output tensor"""
    channel_dim = 0 if tensor.dim() == 4 and tensor.size(0) < 16 else 1
    tensor = tensor.movedim(channel_dim, 0)
    flat_tensor = tensor.reshape(1, -1)
    
    stats = []
    for i in range(1):
        channel_data = flat_tensor[i].to(torch.float32).detach().cpu().numpy()
        
        # Handle infinite and NaN values
        if np.any(np.isinf(channel_data)) or np.any(np.isnan(channel_data)):
            print(f"Warning: Found infinite or NaN values in channel {i}")
            channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=1e6, neginf=-1e6)
        
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

def save_golden_stats_to_file(stats, file_path):
    """Thread-safe save statistics to JSON file"""
    try:
        if stats is None:
            print("Warning: stats is None, skipping save")
            return
        
        # Get thread-safe file handler
        handler = get_thread_safe_handler(file_path)
        
        # Write data atomically
        handler.write_json(stats)
        # print(f"Successfully saved stats to {file_path} (thread: {threading.get_ident()})")
        
    except Exception as e:
        print(f"Error saving golden stats: {e}")
        import traceback
        traceback.print_exc()

def load_golden_stats_from_file(file_path):
    """Thread-safe load statistics from JSON file"""
    try:
        # Get thread-safe file handler
        handler = get_thread_safe_handler(file_path)
        
        # Read data
        all_stats = handler.read_json()
        
        if not all_stats:
            return {}
        
        # print(f"Shape not found (thread: {threading.get_ident()})")
        return all_stats
        
    except Exception as e:
        print(f"Error loading golden stats: {e}")
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
                # Handle infinite and NaN values
                g_val = g[metric]
                t_val = t[metric]
                
                if np.isnan(g_val) or np.isnan(t_val) or np.isinf(g_val) or np.isinf(t_val):
                    print(f"Warning: Invalid values in comparison - golden: {g_val}, test: {t_val}")
                    errors.append(1.0)  # Max error for invalid values
                    continue
                
                diff = abs(g_val - t_val)
                scale = max(abs(g_val), 1e-6)
                errors.append(diff / scale)
        
        if 'histogram' in g and 'histogram' in t:
            try:
                hist_dist = wasserstein_distance(
                    g['histogram']['values'],
                    t['histogram']['values']
                )
                errors.append(hist_dist)
            except Exception as e:
                print(f"Error comparing histograms: {e}")
        
        channel_errors.append(max(errors) if errors else 0.0)
    
    max_error = max(channel_errors) if channel_errors else 0.0
    return max_error <= tolerance, max_error, channel_errors

def load_golden_stats_from_memory(shape_key, database, need_lock = 0):
    """Thread-safe load from in-memory database"""
    try:
        if shape_key is None:
            return False, None
        
        # Check if shape_dict is actually a dictionary
        if not isinstance(database, dict):
            print(f"Error: database is not a dictionary, got {type(database)}")
            return False, None
        
        if need_lock:
            with database_lock:
                if shape_key in database:
                    entry = database[shape_key]
                    if "stats" in entry:
                        return True, entry["stats"]
        else:
            if shape_key in database:
                entry = database[shape_key]
                if "stats" in entry:
                    return True, entry["stats"]
        
        return False, None
        
    except Exception as e:
        print(f"Error loading from memory database: {e}")
        return False, None

def save_golden_stats_to_memory(stats, shape_key, database):
    """Thread-safe save to in-memory database"""
    try:
        if stats is None or not shape_key or not isinstance(database, dict):
            print(f"Error: database is not a dictionary, got {type(database)}")
            return
        
        with database_lock:
            if shape_key not in database:
                database[shape_key] = {}
            
            database[shape_key]["stats"] = stats
        
    except Exception as e:
        print(f"Error saving to memory database: {e}")

