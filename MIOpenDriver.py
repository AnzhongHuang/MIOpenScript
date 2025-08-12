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
from miopUtil.MIArgs import MIArgs
import MIOpenDriver_Ref
import miopUtil.DataHash as DataHash
import threading
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)

def generate_fixed_seed_input(shape, dtype, device, verify=False):

    if verify:
        torch.manual_seed(12345678)
        # Create separate generator with fixed seed
        input_data = (2 * torch.randn(shape, dtype=dtype, device="cpu", requires_grad=True) - 1)
        if device.type == 'cuda':
            input_data = input_data.to(device)
    else:
        # Use the default random generator
        input_data = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    return input_data

def RunConv(device, args, in_data_type, gpu_idx, test_idx=0):

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
    if device.type == 'cuda':
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
        type_str = "bfp16"
    elif in_data_type == MiopenDataType.miopenInt8:
        dtype = torch.int8
        type_str = "int8"
    else:
        dtype = torch.float32
        type_str = ""

    # Determine spatial dimension
    is_3d = args.spatial_dim == 3
    conv_fn = F.conv3d if is_3d else F.conv2d
    tensor_dim = 5 if is_3d else 4
    print(f"is_3d: {is_3d}")
    
    # Create tensors with requires_grad=True for gradient computation
    def create_tensor(shape, filename=""):
        if filename and os.path.exists(filename):
            # Load from file (assuming raw binary format)
            data = np.fromfile(filename, dtype=np.float32)
            return torch.tensor(data.reshape(shape), dtype=dtype, device=device, requires_grad=True)
        else:
            # Create random tensor
            input_data = generate_fixed_seed_input(shape, dtype, device, verify=args.verify)
            return input_data


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

    print(f"in_channels_per_group: {in_channels_per_group}, in_channels: {args.in_channels}, group_count: {args.group_count}")
    
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
                result = conv_fn(input_tensor, weight, bias, **conv_args)
                return result

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
                    output_padding= (0, 0, 0) if is_3d else (0, 0),
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
                    output_padding=(0, 0, 0) if is_3d else (0, 0),
                    groups=conv_args['groups'],
                    output_mask=output_mask
                )
                return grad_weight

    # gpu golden for convolution operations
    def run_convolution_ref(operation):
        with record_function(f"convolution_ref_{operation}"):
            conv_args = {
                    'stride': (args.conv_stride_d, args.conv_stride_h, args.conv_stride_w) if is_3d else 
                            (args.conv_stride_h, args.conv_stride_w),
                    'padding': (args.pad_d, args.pad_h, args.pad_w) if is_3d else 
                            (args.pad_h, args.pad_w),
                    'dilation': (args.dilation_d, args.dilation_h, args.dilation_w) if is_3d else 
                            (args.dilation_h, args.dilation_w),
                    'groups': args.group_count
                }
            
            print(f"input_shape : {input_shape}")
            reference = MIOpenDriver_Ref.gpu_convolution_reference(
                grad_output=grad_output,
                weight=weight,
                input_shape=input_shape,
                padding=conv_args['padding'][0],
                stride=conv_args['stride'][0],
                dilation=conv_args['dilation'][0],
                group=conv_args['groups'],
                solution_id=86,
                operation=operation
            )
            
            if reference is None:
                print("Error: Reference result is None")
            
            return reference

    forw = args.forw

    print("Log_0")
    if args.search:
        print("Log_1")
        torch.backends.cudnn.benchmark=True
    if device.type == 'cuda':
        print("Log_2")
        stream=torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            if args.warmup > 0:
                for _ in range(args.warmup):
                    result = run_convolution(forw)

    # Configure profiler if trace is requested
    if args.trace or args.event:
        print("Log_3")
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

            for _ in range(args.iter):
                result = run_convolution(forw)

            torch.cuda.synchronize()
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
        result = None
        if device.type == 'cuda':
            with torch.cuda.stream(stream):
                start_event[0].record()
                for _ in range(args.iter):
                    result = run_convolution(forw)
                end_event[0].record()
                stream.synchronize()
                
                # compare gpu result with golden
                golden_result = run_convolution_ref(forw)
                # Only compare if both results are available
                if golden_result is not None and result is not None:
                    try:
                        if result is not None:
                            result = result.float()
                            
                        print(f"gpu type: {type(result)}, cpu type: {type(golden_result)}")
                        
                        is_close = torch.allclose(result, golden_result, rtol=1e-3, atol=1e-3)
                        print(f"Success!!! <Results match with golden reference: {is_close}>")
                        
                        if not is_close:
                            diff = torch.abs(result - golden_result)
                            max_diff = torch.max(diff).item()
                            mean_diff = torch.mean(diff).item()
                            print(f"Max difference: {max_diff:.8f}")
                            print(f"Mean difference: {mean_diff:.8f}")
                    except Exception as e:
                        print(f"Error comparing results: {e}")
                else:
                    print("Skipping golden comparison (reference result not available)")
                
                
            elapsed_time_ms = start_event[0].elapsed_time(end_event[0])

        else:
            # CPU time measurement
            start_time = time.time()
            args.iter = 1
            for _ in range(args.iter):
                result = run_convolution(forw)
            end_time = time.time()
            elapsed_time_ms = (end_time - start_time) * 1000

        if (args.verify):
            status = DataHash.summarize_conv_output(result, include_histogram=True, bins=6)
            # print(f"Convolution result shape: {result.shape}")
            # print(f"Convolution status: {status}")

            golden_stats = DataHash.load_golden_stats("conv_output_stats.json")
            tolerance = 0.05
            if golden_stats:
                res, max_error, channel_errors = DataHash.compare_stats(golden_stats, status, tolerance=tolerance)
                if res:
                    print(f"Conv Verify OK: ({max_error} < {tolerance})")
                else:
                    print(f"Conv Verify FAILED: {max_error} >= {tolerance}")
            DataHash.save_golden_stats(status, "conv_output_stats.json")

        # The elapsed time is bigger than kernel execution time
        safe_print(f"Test {test_idx}, GPU {gpu_idx} - execution time: {elapsed_time_ms/(args.iter):.4f} ms")

    # Print results
    op_names = {
        1: "FWD",
        2: "BWD Data",
        4: "BWD Weight"
    }
    print(f"op_names[forw]:{op_names[forw]}")
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
                args = MIArgs.ParseParam(args_list[1:])
                convRunList.append((args, args.in_data_type))

            except Exception as e:
                print(f"Error parsing command line: {command_line}\n{e}")
    return convRunList

def Solve():
    # Set device
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
    else:
        num_gpus = 20
        devices = [torch.device('cpu')  for i in range(num_gpus)]

    fileInput = "--input" in sys.argv[1]
    if fileInput:
        convRunList = ParseRunList(sys.argv[2])
        print(f"Parsed {len(convRunList)} commands from input file.")
        #for args, in_data_type in (convRunList):
        #    RunConv(devices[args.gpu], args, in_data_type)
        #parser.add_argument('--input', type=str, help='Input file containing command lines')
        gpu_ids = itertools.cycle(range(num_gpus))
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            gpu_idx = next(gpu_ids)
            futures = []
            test_idx = 0
            for args, in_data_type in convRunList:
                # Get the next GPU ID in the cycle
                gpu_idx = next(gpu_ids)
                # Submit the task to the executor
                test_idx += 1
                futures.append(executor.submit(RunConv, devices[gpu_idx], args, in_data_type, gpu_idx, test_idx))

            # Optionally wait for all futures to complete
            concurrent.futures.wait(futures)

    else:
        # Parse command name to determine data type
        args = MIArgs.ParseParam(sys.argv[1:])
        if args.cpu == 1:
            dev = torch.device('cpu')
        else:
            dev = devices[args.gpu] if args.gpu < num_gpus else devices[0]
        RunConv(dev, args, args.in_data_type, args.gpu)

def main():
    start_time = time.time()
    Solve()
    end_time   = time.time()

    elapsed_time = (end_time - start_time) * 1000
    print(f"CPU time: {elapsed_time:.6f} ms" )

if __name__ == "__main__":
    main()
