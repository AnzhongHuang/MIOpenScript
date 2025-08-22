import torch
import threading
from typing import Dict, Tuple, Optional
import math
from torch.profiler import profile, record_function
_DType = torch.dtype
_Shape = Tuple[int, ...]
 
class TensorRec:
    __slots__ = ("tensor", "nelems")   # nelems is the capacity in elements
    def __init__(self, tensor: torch.Tensor, nelems: int):
        self.tensor = tensor
        self.nelems = nelems           # capacity
 
class PoolMgr:
    """
    Per-GPU cache that keeps **one random tensor per shape**,
    generated once with torch.randn.
    """
    _known_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    _max_gpus = 8
 
    def __init__(self):
        # (device_id, dtype) -> {shape -> TensorRec}
        self._caches: Dict[Tuple[int, _DType, bool], Dict[_Shape, TensorRec]] = {
            (gpu, dt, readonly): {} for gpu in range(self._max_gpus) for dt in self._known_dtypes for readonly in (True, False)
        }
        self._locks = [threading.Lock()] * self._max_gpus
 
    # key = (device, dtype, nelems)
    def _key(self, nelems: int, dtype: _DType, device: int) -> Tuple[int, _DType, int]:
        return (device, dtype, nelems)

    # ---------- helpers -------------------------------------------------
    def _parse_device(self, device):
        if device is None:
            return torch.cuda.current_device()
        if isinstance(device, torch.device):
            if device.type != "cuda":
                raise ValueError("Only CUDA devices supported")
            return device.index or torch.cuda.current_device()
        if isinstance(device, int):
            if not 0 <= device < self._max_gpus:
                raise ValueError(f"device {device} out of range 0-{self._max_gpus-1}")
            return device
        raise ValueError("Unsupported device type")
 
    # ---------- public API ----------------------------------------------
    def get(self,
            shape: _Shape,
            dtype: _DType,
            readonly: bool,
            device: torch.device,
            gpu_id: int) -> torch.Tensor:
            if dtype not in self._known_dtypes:
                raise ValueError(f"Unsupported dtype {dtype}")
    
            needed = math.prod(shape)
    
            with self._locks[gpu_id]:
                bucket = self._caches[(gpu_id, dtype, readonly)]
    
                # look for the smallest cached tensor that is >= needed
                best_key = None
                best_nelems = None
                for k in bucket.keys():
                    nelems = k[-1]
                    if nelems >= needed:
                        if best_nelems is None or nelems > best_nelems:
                            best_nelems = nelems
                            best_key = k
    
                if best_key is not None:
                    rec = bucket.pop(best_key)   # LRU: move to end
                    tensor = rec.tensor
                    orig_size = tensor.numel()
                    flat_needed = tensor.view(-1)
                    flat_needed = flat_needed[:needed]
                    tensor = flat_needed.reshape(shape)

                    bucket[best_key] = rec  # put it back in the cache
                    # return a view of the first N elements
                    #print(f"needed:{needed}, exists tensor size:{orig_size}")
                    return tensor
    
                # no suitable tensor → create one exactly of size 'needed'
                # tensor = torch.randn(needed, dtype=dtype, device=torch.device("cuda", gpu_id))
                alloc_size = max(915936000, needed)
                tensor = (torch.randn((alloc_size,), dtype=dtype, device=device, requires_grad=True) - 0.5) * 2
                rec = TensorRec(tensor, alloc_size)
                bucket[(gpu_id, dtype, alloc_size)] = rec
                # print(f"gpu:{gpu_id}, new tensor size:{tensor.numel()}")
                flat_needed = tensor.view(-1)
                flat_needed = flat_needed[:needed]
                tensor = flat_needed.reshape(shape)
                return tensor.view(shape)
 
 # ------------------------------------------------------------------ #
    def ShowBucket(self,
                   device: int | torch.device | None = None,
                   dtype: torch.dtype | None = None) -> None:
        """
        Pretty-print the current contents of every bucket that matches
        (device, dtype).  If either argument is None, show all matching
        buckets.
 
        Example:
            mgr.ShowBucket(device=2, dtype=torch.float16)
        """
        gpu_id = self._parse_device(device) if device is not None else None
 
        def _fmt(k: Tuple[int, torch.dtype, int]) -> str:
            g, dt, nelems = k
            return f"GPU{g}-{dt} {nelems} elements"
 
        with self._lock:
            for k, bucket in self._caches.items():
                g, dt = k
                if gpu_id is not None and g != gpu_id:
                    continue
                if dtype is not None and dt != dtype:
                    continue
                if not bucket:
                    continue
 
                print("─" * 60)
                for bkey, tensorrec in bucket.items():
                    print(f"{_fmt(bkey)}")

    # ---------- misc -----------------------------------------------------
    def clear(self, gpu_id: Optional[int] = None):
        with self._lock:
            if gpu_id is None:
                for bucket in self._caches.values():
                    bucket.clear()
            else:
                gpu_id = self._parse_device(gpu_id)
                for dt in self._known_dtypes:
                    self._caches[(gpu_id, dt)].clear()
 
    def stats(self, gpu_id: Optional[int] = None):
        with self._lock:
            out = {}
            targets = range(self._max_gpus) if gpu_id is None else [self._parse_device(gpu_id)]
            for g in targets:
                for dt in self._known_dtypes:
                    bucket = self._caches[(g, dt)]
                    total = len(bucket)
                    hits = sum(r.hits for r in bucket.values())
                    out[(g, dt)] = {"tensors": total, "hits": hits}
            return out