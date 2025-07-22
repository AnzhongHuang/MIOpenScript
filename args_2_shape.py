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
from miopUtil.shapeConvert import MiopenDataType
from miopUtil.MIArgs import MIArgs

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
    parser.add_argument('--shapeformat', type=str, default='vs',  choices=['vs', 'solver', 'kernel'],
                        help='Shape type to convert: vs=vsdb, solver=solverdb, kernel')

    # Parse known args (ignore extra)
    args, _ = parser.parse_known_args(args_list)

    miargs = MIArgs(
        forw=args.forw,
        batchsize=args.batchsize,
        in_channels=args.in_channels,
        in_h=args.in_h,
        in_w=args.in_w,
        in_d=args.in_d,
        out_channels=args.out_channels,
        fil_h=args.fil_h,
        fil_w=args.fil_w,
        fil_d=args.fil_d,
        pad_h=args.pad_h,
        pad_w=args.pad_w,
        pad_d=args.pad_d,
        conv_stride_h=args.conv_stride_h,
        conv_stride_w=args.conv_stride_w,
        conv_stride_d=args.conv_stride_d,
        dilation_h=args.dilation_h,
        dilation_w=args.dilation_w,
        dilation_d=args.dilation_d,
        group_count=args.group_count,
        spatial_dim=args.spatial_dim,
        solution=args.solution,
        time=args.time,
        verify=args.verify,
        mode=args.mode,
        in_data=args.in_data,
        weights=args.weights,
        dout_data=args.dout_data,
        bias=args.bias,
        in_bias=args.in_bias,
        iter=args.iter,
        pad_mode=args.pad_mode,
        fil_layout=args.fil_layout,
        in_layout=args.in_layout,
        out_layout=args.out_layout,
        gpubuffer_check=args.gpubuffer_check,
        wall=args.wall,
        printconv=args.printconv,
        pad_val=args.pad_val,
        dump_output=args.dump_output,
        search=args.search,
        verification_cache=args.verification_cache,

        # private fields
        shapeformat=args.shapeformat,
        in_data_type=in_data_type
    )

    return miargs

def Solve():
    # Parse command name to determine data type
    args = ParseParam(sys.argv[1:])
    cmds, conv_type = shapeConvert.GetArgument(args)

    if args.shapeformat == 'vs':
        # Print vsdb format
        print(f'"{conv_type}",')
        # Print parameters, e.g. "-a", f'"{in_bias}"', so I can fill them into vscode debugging
        idx = 0
        for idx, param in enumerate(cmds):
            print(f'"{param}"', end=',')
            if idx % 2 == 1:
                print()
            else:
                print(' ', end='')
    elif args.shapeformat == 'solver' or args.shapeformat == 'kernel':
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
            in_data_type=args.in_data_type,
            weights_data_type=args.in_data_type,
            out_data_type=args.in_data_type,
            direction_str=shapeConvert.get_direction_str(args.forw),
            group_count=args.group_count
        )
        problem.InitDef()
        if args.shapeformat == 'solver':
            print(problem.ufdbSerialize())
        elif args.shapeformat == 'kernel':
            print(problem.udbSerialize())

def main():
    Solve()
# python args_2_shape.py convfp16 -F 1 -n 128 -c 192 -H 1 -W 1 -k 8 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -g 1 -m conv --spatial_dim 2 -t 1 --shapeformat solver
# python args_2_shape.py convfp16 -F 1 -n 128 -c 192 -H 1 -W 1 -k 8 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -g 1 -m conv --spatial_dim 2 -t 1 --shapeformat kernel
# python args_2_shape.py convfp16 -F 1 -n 128 -c 192 -H 1 -W 1 -k 8 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -g 1 -m conv --spatial_dim 2 -t 1 --shapeformat vs
if __name__ == "__main__":
    main()
