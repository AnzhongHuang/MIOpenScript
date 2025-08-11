import miopUtil.DataHash as DataHash
import os
import torch
import numpy as np

dtype = torch.float16
file_path = "dump_bwd_din_gpu.bin"

N = 128

with open(file_path, 'rb') as f:
    data = np.fromfile(f, dtype=np.float16)
    if data.size == 0:
        raise ValueError("File is empty or not found")
    
    # Reshape the data to match the expected tensor shape
    shape = (N, -1)  # Assuming the first dimension is N and the rest is inferred
    tensor = torch.tensor(data.reshape(shape), dtype=dtype)

tensor = tensor.reshape(tensor.size(0), -1)  # Flatten the tensor to 2D
print(f"Tensor shape: {tensor.shape}")

stats = DataHash.summarize_conv_output(tensor, include_histogram=True, bins=10)
DataHash.save_golden_stats(stats, "test_stats.json")
# print(f"Statistics: {stats}")
golden_stats = DataHash.load_golden_stats("golden_stats.json")
#print(f"Golden stats: {golden_stats}")
if golden_stats is not None:
    res, max_error, channel_errors = DataHash.compare_stats(golden_stats, stats, tolerance=0.05)
    print(f"Max error: {max_error}, channel error: {channel_errors}")
    print(f"Res: {res}")
    if res:
        print("Statistics match within tolerance.")
    else:
        print("Statistics do not match within tolerance.")