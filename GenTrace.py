import torch
import torch.profiler as profiler
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any
from collections import defaultdict
# https://github.com/ROCm/pyrsmi
from pyrsmi import rocml
import time
import threading
import random
from itertools import cycle
import os

@dataclass
class ThreadBusySpan:
    start: float
    stop: float | None
    gpu_time: float | None
    metadata: Dict[str, Any]
 
class ROCmPerfettoMonitor:
    def __init__(self):
        # Initialize ROCm SMI
        rocml.smi_initialize()
        self.gpu_count = rocml.smi_get_device_count()
        print(f"Number of GPUs: {self.gpu_count}")
        # CPU monitoring setup
        self.cpu_thread_count = 8
        # Recording parameters
        self.call_interval = 0.1
        self.duration = 60 * 60
        self.recording = False
        # Data storage
        self.timestamps = []
        self.cpu_thread_events : Dict[int : List[ThreadBusySpan]] = defaultdict(list)

        self.gpu_data = defaultdict(lambda: {
            'utilization': [],
            'mem_usage': [],
            'timestamps' : []
        })
        self.gpu_total_usage = [0] * self.gpu_count

        self.start_time = 1751410836.000000000

        self.color_diff = [
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N"
        ]
        
    def start_thread_event(self, thread_id: int, ts: float, meta: Dict[str, Any]) -> None:
        if self.recording == False:
            self.start_record()
        elapsed = ts - self.start_time
        span = ThreadBusySpan(start=elapsed, stop=None, gpu_time=None, metadata=meta)
        self.cpu_thread_events[thread_id].append(span)
 
    def stop_thread_event(self, thread_id: int, ts: float, execution_time: float) -> None:
        elapsed = ts - self.start_time
        spans = self.cpu_thread_events[thread_id]
        if spans and spans[-1].stop is None:
            spans[-1].stop = elapsed
            spans[-1].gpu_time = execution_time
 
    def _record_gpu(self):
        """Internal recording thread"""
        while self.recording:
            current_time = time.time() - self.start_time
            # Record GPU data
            for device_id in range(self.gpu_count):
                util = rocml.smi_get_device_utilization(device_id)
                mem_usage = rocml.smi_get_device_memory_used(device_id)
                self.gpu_data[device_id]['utilization'].append(util)
                self.gpu_data[device_id]['mem_usage'].append(mem_usage)
                self.gpu_data[device_id]['timestamps'].append(current_time)
                self.gpu_total_usage[device_id] += util

            time.sleep(self.call_interval)
 
    def start_record(self):
        self.recording = True
        self.record_thread = threading.Thread(target=self._record_gpu)
        self.record_thread.start()
 
    def stop_record(self):
        self.recording = False
        if hasattr(self, 'record_thread'):
            self.record_thread.join()

    def convert_to_perfetto_format(self):
        """Convert collected data to Perfetto-compatible format"""
        trace_data = {"traceEvents": []}
        # Add CPU thread events
        color_iter = cycle(self.color_diff)
        pid_os = os.getpid()

        for thread_id, spans in self.cpu_thread_events.items():
            for span in spans:
                if span.stop is None:
                    continue

                c_name = next(color_iter)
                test_idx = span.metadata['test_idx']
                gpu_idx  = span.metadata['gpu_idx']

                trace_data["traceEvents"].append({
                    "ph":  "X",
                    "cat": f"GPU_{gpu_idx}",
                    "name": f"test:{test_idx}-{c_name}-GPU:{gpu_idx}",
                    "ts":  span.start * 1e6,
                    "dur": (span.stop - span.start) * 1e6,
                    "pid": pid_os,             # CPU process
                    "tid": thread_id,       # logical CPU thread
                    "args": {
                        "device": gpu_idx,
                        "cid":    gpu_idx,    # keep both for convenience
                    }
                })

        # Add GPU events
        gpu_trace_event = []
        for device_id in range(self.gpu_count):
            util_list       = self.gpu_data[device_id]['utilization']
            mem_usage_list  = self.gpu_data[device_id]['mem_usage']
            ts_list         = self.gpu_data[device_id]['timestamps']

            total_mem_usage = rocml.smi_get_device_memory_total(device_id)
            count = len(util_list)
            avg_util = self.gpu_total_usage[device_id] / count if count else 0

            for util, mem_usage, timestamp in zip(util_list, mem_usage_list, ts_list):
                gpu_trace_event.append({
                    "ph": "C",
                    "cat": "gc",
                    "name": f"GPU_{device_id} usage: {avg_util:.1f}%",
                    "ts": timestamp * 1e6,
                    "pid": 101 + device_id,
                    "tid": 0,
                    "args": {"value": util, "units": "%"}
                })
                gpu_trace_event.append({
                    "ph": "C",
                    "cat": "gc",
                    "name": f"GPU_{device_id} mem size(GB) {total_mem_usage/1024/1024/1024:.2f}",
                    "ts": timestamp * 1e6,
                    "pid": 101 + device_id,
                    "tid": 1,
                    "args": {"value": mem_usage/total_mem_usage * 100, "units": "%"}
                })

        # Add GPU counter events
        trace_data["traceEvents"].extend(gpu_trace_event)
        return trace_data
 
    def save_perfetto_trace(self, filename):
        """Save trace in Chrome Tracing format (compatible with Perfetto)"""
        self.stop_record()
        trace_data = self.convert_to_perfetto_format()
        with open(filename, 'w') as f:
            json.dump(trace_data, f)
        print(f"Perfetto trace saved to {filename}")

    def close(self):
        rocml.smi_shutdown()

    def merge_traces_with_relationships(pytorch_trace_path, rocm_trace_path, output_path):
        """
        Merge PyTorch profiler trace with ROCm monitor trace, preserving relationships
        """
        # Load traces
        with open(pytorch_trace_path, 'r') as f:
            pytorch_data = json.load(f)
        with open(rocm_trace_path, 'r') as f:
            rocm_data = json.load(f)
        # Merge events
        merged_events = pytorch_data.get('traceEvents', []) + rocm_data.get('traceEvents', [])
        # Sort events by timestamp
        # merged_events.sort(key=lambda x: x['ts'])

        # Create metadata to document the merge
        merged_trace = {
            'traceEvents': merged_events,
            'displayTimeUnit': 'ms',
            'systemTraceEvents': 'System',
            'otherData': {
                'version': 'Merged PyTorch-ROCm Trace with Relationships',
                'createdAt': time.strftime('%Y-%m-%d %H:%M:%S'),
                'sourceTraces': [pytorch_trace_path, rocm_trace_path]
            }
        }
        # Save merged trace
        with open(output_path, 'w') as f:
            json.dump(merged_trace, f, indent=2)
        print(f"Merged trace with relationships saved to {output_path}")