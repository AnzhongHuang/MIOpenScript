from html import parser
import torch
import time
import torch.nn.functional as F
import argparse
import sys
import os
import numpy as np
import shlex
import concurrent.futures
import itertools
from torch.profiler import profile, record_function, ProfilerActivity
from enum import Enum
import miopUtil.shapeConvert as shapeConvert
from miopUtil.MIArgs import MiopenDataType

def ParseParam(args_list):
    command_name = args_list[0] if len(sys.argv) > 1 else "conv"
    if "fp16" in command_name.lower():
        in_data_type = MiopenDataType.miopenHalf
    elif "bfp16" in command_name.lower():
        in_data_type = MiopenDataType.miopenBFloat16
    elif "int8" in command_name.lower():
        in_data_type = MiopenDataType.miopenInt8
    elif "fp32" in command_name.lower():
        in_data_type = MiopenDataType.miopenFloat
    elif "fp64" in command_name.lower():
        in_data_type = MiopenDataType.miopenDouble

    parser = argparse.ArgumentParser(description='PyTorch MIOpenDriver Simulator',
                                add_help=False)
    # Operation type (F flag)
    parser.add_argument('-F', '--forw', type=int, required=True, 
                        choices=[0, 1, 2, 3, 4, 5, 6], 
                        help='Operation type: 0=all, 1=FWD, 2=BWD data, 4=BWD weight, 3=FWD+BWD, 5=FWD+WRW, 6=BWD+WRW')
    
    # Input tensor parameters
    parser.add_argument('-n', '--batchsize', type=int, required=True, help='Mini-batch size')
    parser.add_argument('-c', '--in_channels', type=int, required=True, help='Input channels')
    parser.add_argument('-H', '--in_h', type=int, required=True, help='Input height')
    parser.add_argument('-W', '--in_w', type=int, required=True, help='Input width')
    parser.add_argument('-!', '--in_d', type=int, default=1, help='Input depth (3D)')
    
    # Output channels
    parser.add_argument('-k', '--out_channels', type=int, required=True, help='Output channels')
    
    # Kernel parameters
    parser.add_argument('-y', '--fil_h', type=int, required=True, help='Filter height')
    parser.add_argument('-x', '--fil_w', type=int, required=True, help='Filter width')
    parser.add_argument('-@', '--fil_d', type=int, default=1, help='Filter depth (3D)')
    
    # Padding parameters
    parser.add_argument('-p', '--pad_h', type=int, default=0, help='Vertical padding')
    parser.add_argument('-q', '--pad_w', type=int, default=0, help='Horizontal padding')
    parser.add_argument('-$', '--pad_d', type=int, default=0, help='Depth padding (3D)')
    
    # Stride parameters
    parser.add_argument('-u', '--conv_stride_h', type=int, default=1, help='Vertical stride')
    parser.add_argument('-v', '--conv_stride_w', type=int, default=1, help='Horizontal stride')
    parser.add_argument('-#', '--conv_stride_d', type=int, default=0, help='Depth stride (3D)')
    
    # Dilation parameters
    parser.add_argument('-l', '--dilation_h', type=int, default=1, help='Vertical dilation')
    parser.add_argument('-j', '--dilation_w', type=int, default=1, help='Horizontal dilation')
    parser.add_argument('-^', '--dilation_d', type=int, default=0, help='Depth dilation (3D)')
    
    # Groups
    parser.add_argument('-g', '--group_count', type=int, default=1, help='Number of groups')
    
    # Spatial dimension
    parser.add_argument('-_', '--spatial_dim', type=int, default=2, choices=[2, 3], 
                        help='Convolution spatial dimension (2=2D, 3=3D)')
    
    # Solver and data type
    parser.add_argument('-S', '--solution', type=int, default=-1, help='Solution ID')
    parser.add_argument('-t', '--time', type=int, default=0,
                        help='Print time in milliseconds')
    parser.add_argument('-V', '--verify', type=int, default=1, 
                        help='Verification mode (0=no, 1=yes)')
    parser.add_argument('-m', '--mode', default='conv', 
                        choices=['conv', 'trans'], help='Convolution mode')
    
    # Data loading
    parser.add_argument('-d', '--in_data', type=str, default='', help='Input data filename')
    parser.add_argument('-e', '--weights', type=str, default='', help='Input weights filename')
    parser.add_argument('-D', '--dout_data', type=str, default='', help='dy data filename for BWD weight')
    
    # Bias
    parser.add_argument('-b', '--bias', type=int, default=0, help='Use bias')
    parser.add_argument('-a', '--in_bias', type=str, default='', help='Input bias filename')

    # Additional MIOpenDriver parameters
    parser.add_argument('-i', '--iter', type=int, default=10, help='Number of iterations')
    parser.add_argument('-z', '--pad_mode', type=str, default='default', 
                        choices=['default', 'same', 'valid'], help='Padding mode')
    parser.add_argument('-f', '--fil_layout', type=str, default='', help='Filter layout')
    parser.add_argument('-I', '--in_layout', type=str, default='', help='Input layout')
    parser.add_argument('-O', '--out_layout', type=str, default='', help='Output layout')
    parser.add_argument('-~', '--gpubuffer_check', type=int, default=0, 
                        help='GPU buffer sanitation check')
    parser.add_argument('-w', '--wall', type=int, default=0, 
                        help='Wall-clock time measurement')
    parser.add_argument('-P', '--printconv', type=int, default=1, 
                        help='Print convolution dimensions')
    parser.add_argument('-r', '--pad_val', type=int, default=0, help='Padding value')
    parser.add_argument('-o', '--dump_output', type=int, default=0, 
                        help='Dump output buffers')
    parser.add_argument('-s', '--search', type=int, default=0, 
                        help='Search kernel config')
    parser.add_argument('-C', '--verification_cache', type=str, default='', 
                        help='Verification cache directory')

    # Trace option
    parser.add_argument('--trace', type=str, default='',
                        help='Path to save PyTorch execution trace log')
    parser.add_argument('--event', type=str, default='',
                        help='Path to save PyTorch execution trace events')
    parser.add_argument('--warmup', type=int, default=5,
                        help='warmup count')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index')

    parser.add_argument('--dbshape', type=int, default=0,
                        help='verify db shape')
    # Parse known args (ignore extra)
    args, _ = parser.parse_known_args(args_list)

    return args, in_data_type

def RunConv(device, args, in_data_type, gpu_idx):

    if args.dbshape:
        problem = shapeConvert.ProblemDescription(
            in_channels=args.in_channels if args.forw == 1 else args.out_channels,
            spatial_dims=args.spatial_dim,
            in_depth=args.in_d,
            in_height=args.in_h,
            in_width=args.in_w,
            weights_depth=args.fil_d,
            weights_height=args.fil_h,
            weights_width=args.fil_w,
            out_channels=args.out_channels if args.forw == 1 else args.in_channels,
            out_depth=0,  # Will be calculated later
            out_height=0,  # Will be calculated later
            out_width=0,  # Will be calculated later
            in_batch_size=args.batchsize,
            pad_d=args.pad_d,
            pad_h=args.pad_h,
            pad_w=args.pad_w,
            kernel_stride_d=args.conv_stride_d,
            kernel_stride_h=args.conv_stride_h,
            kernel_stride_w=args.conv_stride_w,
            dilation_d=args.dilation_d,
            dilation_h=args.dilation_h,
            dilation_w=args.dilation_w,
            bias=int(args.bias),
            in_layout='NCHW' if args.spatial_dim == 2 else 'NCDHW',
            weights_layout='NCHW' if args.spatial_dim == 2 else 'NCDHW',
            out_layout='NCHW' if args.spatial_dim == 2 else 'NCDHW',
            in_data_type=in_data_type,
            weights_data_type=in_data_type,
            out_data_type=in_data_type,
            direction_str=shapeConvert.get_direction_str(args.forw),
            group_count=args.group_count
        )
        problem.InitDef()
        problem.test()

    # which device to use
    torch.cuda.set_device(device)

    # soluion works only without user db
    if args.solution >= 0:
        os.environ["MIOPEN_DEBUG_FIND_ONLY_SOLVER"] = str(args.solution)

    # Determine data type
    if in_data_type == MiopenDataType.miopenHalf:
        dtype = torch.float16
        type_str = "fp16"
    elif in_data_type == MiopenDataType.miopenBFloat16:
        dtype = torch.bfloat16
        type_str = "bf16"
    elif in_data_type == MiopenDataType.miopenInt8:
        dtype = torch.int8
        type_str = "int8"
    else:
        dtype = torch.float32
        type_str = "fp32"

    # Determine spatial dimension
    is_3d = args.spatial_dim == 3
    conv_fn = F.conv3d if is_3d else F.conv2d
    tensor_dim = 5 if is_3d else 4
    
    # Create tensors with requires_grad=True for gradient computation
    def create_tensor(shape, filename=""):
        if filename and os.path.exists(filename):
            # Load from file (assuming raw binary format)
            data = np.fromfile(filename, dtype=np.float32)
            return torch.tensor(data.reshape(shape), dtype=dtype, device=device, requires_grad=True)
        else:
            # Create random tensor
            return torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    
    # Calculate output dimensions
    def calc_output_size(in_size, pad, kernel, stride, dilation=1):
        return (in_size + 2*pad - dilation*(kernel - 1) - 1) // stride + 1
    
    if is_3d:
        out_depth = calc_output_size(args.in_d, args.pad_d, args.fil_d, args.conv_stride_d, args.dilation_d)
    out_height = calc_output_size(args.in_h, args.pad_h, args.fil_h, args.conv_stride_h, args.dilation_h)
    out_width = calc_output_size(args.in_w, args.pad_w, args.fil_w, args.conv_stride_w, args.dilation_w)
    
    # Create input tensor
    if is_3d:
        input_shape = (args.batchsize, args.in_channels, args.in_d, args.in_h, args.in_w)
    else:
        input_shape = (args.batchsize, args.in_channels, args.in_h, args.in_w)
    
    input_tensor = create_tensor(input_shape, args.in_data)
    
    # Create weight tensor
    in_channels_per_group = args.in_channels // args.group_count
    if is_3d:
        weight_shape = (args.out_channels, in_channels_per_group, args.fil_d, args.fil_h, args.fil_w)
    else:
        weight_shape = (args.out_channels, in_channels_per_group, args.fil_h, args.fil_w)
    
    weight = create_tensor(weight_shape, args.weights)
    
    # Create bias tensor if needed
    bias = None
    if args.bias:
        bias_shape = (args.out_channels,)
        bias = create_tensor(bias_shape, args.in_bias)
    
    # Create output gradient tensor for backward operations
    if is_3d:
        grad_output_shape = (args.batchsize, args.out_channels, out_depth, out_height, out_width)
    else:
        grad_output_shape = (args.batchsize, args.out_channels, out_height, out_width)
    
    grad_output = create_tensor(grad_output_shape, args.dout_data)

    trace_name = f"conv{type_str}-F{args.forw}-n{args.batchsize}-c{args.in_channels}"
    # Generate equivalent MIOpenDriver command
    cmd = f"MIOpenDriver conv{type_str} -F {args.forw} -n {args.batchsize} -c {args.in_channels} "
    
    if is_3d:
        cmd += f"-d {args.in_d} "
    cmd += f"-H {args.in_h} -W {args.in_w} -k {args.out_channels} "
    trace_name += f"-H{args.in_h}-W{args.in_w}-k{args.out_channels}"

    if is_3d:
        cmd += f"-@ {args.fil_d} "
    cmd += f"-y {args.fil_h} -x {args.fil_w} "
    trace_name += f"-y{args.fil_h}-x{args.fil_w}"

    if is_3d:
        cmd += f"-$ {args.pad_d} "
    cmd += f"-p {args.pad_h} -q {args.pad_w} "
    trace_name += f"-p{args.pad_h}-q{args.pad_w}"

    if is_3d:
        cmd += f"-# {args.conv_stride_d} "
    cmd += f"-u {args.conv_stride_h} -v {args.conv_stride_w} "
    trace_name += f"-u{args.conv_stride_h}-v{args.conv_stride_w}"

    if is_3d:
        cmd += f"-^ {args.dilation_d} "
    cmd += f"-l {args.dilation_h} -j {args.dilation_w} "
    trace_name += f"-l{args.dilation_h}-j{args.dilation_w}"
    
    cmd += f"-g {args.group_count} -m {args.mode} -_ {args.spatial_dim} "
    trace_name += f"-g{args.group_count}"

    cmd += f"-t {args.time} -S {args.solution} -V {args.verify}"
    
    # Add file parameters if specified
    if args.in_data:
        cmd += f" -d {args.in_data}"
    if args.weights:
        cmd += f" -e {args.weights}"
    if args.dout_data:
        cmd += f" -D {args.dout_data}"
    if args.in_bias:
        cmd += f" -a {args.in_bias}"

    # Wrapper function for convolution operations
    def run_convolution(operation):
        with record_function(f"convolution_{operation}"):
            # Common convolution parameters
            conv_args = {
                'stride': (args.conv_stride_d, args.conv_stride_h, args.conv_stride_w) if is_3d else 
                         (args.conv_stride_h, args.conv_stride_w),
                'padding': (args.pad_d, args.pad_h, args.pad_w) if is_3d else 
                          (args.pad_h, args.pad_w),
                'dilation': (args.dilation_d, args.dilation_h, args.dilation_w) if is_3d else 
                           (args.dilation_h, args.dilation_w),
                'groups': args.group_count
            }
            
            if operation == 1:  # Forward convolution
                return conv_fn(input_tensor, weight, bias, **conv_args)
                
            elif operation == 2:  # Backward data
                output_mask = [True, False, False]
                grad_input, grad_weight, _ = torch.ops.aten.convolution_backward(
                    grad_output=grad_output,
                    input=input_tensor,
                    weight=weight,
                    bias_sizes=[0],          # [0] indicates no bias in forward pass
                    stride=conv_args['stride'],
                    padding=conv_args['padding'],
                    dilation=conv_args['dilation'],
                    transposed=False,
                    output_padding=(0, 0),
                    groups=conv_args['groups'],
                    output_mask=output_mask
                )

                return grad_input
                
            elif operation == 4:  # Backward weight
                output_mask = [False, True, False]
                grad_input, grad_weight, _ = torch.ops.aten.convolution_backward(
                    grad_output=grad_output,
                    input=input_tensor,
                    weight=weight,
                    bias_sizes=[0],          # [0] indicates no bias in forward pass
                    stride=conv_args['stride'],
                    padding=conv_args['padding'],
                    dilation=conv_args['dilation'],
                    transposed=False,
                    output_padding=(0, 0),
                    groups=conv_args['groups'],
                    output_mask=output_mask
                )

                return grad_weight
    
    forw = args.forw

    if args.search:
        torch.backends.cudnn.benchmark=True

    # Warm-up run
    if (args.warmup > 0):
        for _ in range(args.warmup):
            result = run_convolution(forw)
        torch.cuda.synchronize()

    # Configure profiler if trace is requested
    if args.trace or args.event:
        prefix = args.trace.split(".")[0]
        trace_path = f"{prefix}_{trace_name}.json"
        trace_path = os.path.abspath(trace_path)
        trace_dir = os.path.dirname(trace_path)
        if trace_dir and not os.path.exists(trace_dir):
            os.makedirs(trace_dir)

        prefix = args.event.split(".")[0]
        event_path = f"{prefix}_{trace_name}_event.json"
        event_path = os.path.abspath(event_path)

        print(f"\nStarting PyTorch trace capture (saving to {trace_path})")

        # Actual trace capture
        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            record_shapes=True,
            with_stack=False,
            with_flops=True
        ) as prof:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)

            elapsed_time_ms = 0
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(args.iter):
                result = run_convolution(forw)

            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms += start_event.elapsed_time(end_event)

            # The elapsed time is bigger than kernel execution time
            print(f"execution time: {elapsed_time_ms/args.iter:.3f} ms")
        
        # Save trace
        if (args.trace):
            prof.export_chrome_trace(trace_path)
            print(f"PyTorch trace saved to {trace_path}")
        if (args.event):
            events = prof.key_averages(group_by_input_shape=False).table(
                sort_by="self_cuda_time_total", row_limit=120
            )
            with open(event_path, "w") as file:
                file.write(events)
            print(events)
            print(f"Events have been written to {event_path}")
    else:
        start_event = [torch.cuda.Event(enable_timing=True) for _ in range(args.iter)]
        end_event =   [torch.cuda.Event(enable_timing=True) for _ in range(args.iter)]

        elapsed_time_ms = 0
        torch.cuda.synchronize()
        start_event[0].record()
        for _ in range(args.iter):

            result = run_convolution(forw)
        end_event[0].record()

        torch.cuda.synchronize()

        t = start_event[0].elapsed_time(end_event[0])
        elapsed_time_ms += t

        # The elapsed time is bigger than kernel execution time
        print(f"GPU {gpu_idx} - execution time: {elapsed_time_ms/(args.iter):.4f} ms")
    
    # Print results
    op_names = {
        1: "FWD",
        2: "BWD Data",
        4: "BWD Weight"
    }
    operations_str = ", ".join(op_names[forw])

def ParseRunList(file_path):
    convRunList = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        command_line = line.strip()
        if command_line:
            try:
                args_list = shlex.split(command_line)
                args, in_data_type = ParseParam(args_list[1:])
                convRunList.append((args, in_data_type))

            except Exception as e:
                print(f"Error parsing command line: {command_line}\n{e}")
    return convRunList

def Solve():
    # Set device
    num_gpus = torch.cuda.device_count()
    devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]

    fileInput = "--input" in sys.argv[1]
    if fileInput:
        convRunList = ParseRunList(sys.argv[2])

        #for args, in_data_type in (convRunList):
        #    RunConv(devices[args.gpu], args, in_data_type)
        #parser.add_argument('--input', type=str, help='Input file containing command lines')
        gpu_ids = itertools.cycle(range(num_gpus))
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            gpu_idx = next(gpu_ids)
            futures = []
            for args, in_data_type in convRunList:
                # Get the next GPU ID in the cycle
                gpu_idx = next(gpu_ids)
                # Submit the task to the executor
                futures.append(executor.submit(RunConv, devices[gpu_idx], args, in_data_type, gpu_idx))

            # Optionally wait for all futures to complete
            concurrent.futures.wait(futures)

    else:
        # Parse command name to determine data type
        args, in_data_type = ParseParam(sys.argv[1:])

        RunConv(devices[args.gpu], args, in_data_type, args.gpu)

def main():
    start_time = time.time()
    Solve()
    end_time   = time.time()

    elapsed_time = (end_time - start_time) * 1000
    print(f"CPU time: {elapsed_time:.6f} ms" )

if __name__ == "__main__":
    main()
