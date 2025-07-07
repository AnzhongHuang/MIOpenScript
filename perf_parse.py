import json
import argparse
from collections import defaultdict

def parse_pytorch_trace(trace_file):
    """Parse PyTorch trace file and extract kernel performance metrics"""
    with open(trace_file, 'r') as f:
        trace_data = json.load(f)
    
    # Extract kernel events
    kernel_events = []
    for event in trace_data['traceEvents']:
        if event.get('cat') == 'cuda_runtime' and event.get('name') == 'hipLaunchKernel':
            kernel_events.append(event)
            #print(event)
    
    # Aggregate kernel performance data
    kernel_stats = defaultdict(lambda: {'count': 0, 'total_time': 0})
    
    for event in kernel_events:
        if 'args' in event and 'kernel' in event['args']:
            kernel_name = event['args']['kernel']
            duration = event.get('dur', 0)  # Duration in microseconds
            #print(kernel_name)
            kernel_stats[kernel_name]['count'] += 1
            kernel_stats[kernel_name]['total_time'] += duration
    
    # Convert to sorted list (by total_time descending)
    sorted_stats = sorted(
        [(name, data['count'], data['total_time']) 
         for name, data in kernel_stats.items()],
        key=lambda x: x[2], 
        reverse=True
    )
    
    return sorted_stats

def main():
    parser = argparse.ArgumentParser(description='Parse PyTorch trace files for kernel performance')
    parser.add_argument('--trace', required=True, help='Path to PyTorch trace file')
    parser.add_argument('--output', default='', help='Output CSV file path (optional)')
    args = parser.parse_args()
    
    # Parse trace file
    kernel_stats = parse_pytorch_trace(args.trace)
    
    # Print results
    print("\nPyTorch Kernel Performance Report")
    print("=" * 120)
    print(f"{'Kernel Name':<60} {'Calls':>8} {'Total Time (ms)':>15} {'Avg Time (ms)':>15}")
    print("-" * 80)
    
    # Prepare output data
    output_lines = []
    for name, count, total_time_us in kernel_stats:
        total_time_ms = total_time_us / 1000
        avg_time_ms = total_time_ms / count if count > 0 else 0
        
        line = f"{name[:78]:<80} {count:>8} {total_time_ms:>15.3f} {avg_time_ms:>15.5f}"
        print(line)
        output_lines.append(f"{name},{count},{total_time_ms:.3f},{avg_time_ms:.5f}\n")
    
    print("=" * 120)
    print(f"Total kernels: {len(kernel_stats)}")
    print(f"Total kernel execution time: {sum(t for _,_,t in kernel_stats)/1000:.3f} ms")
    
    # Save to CSV if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write("kernel_name,call_count,total_time_ms,avg_time_ms\n")
            f.writelines(output_lines)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
