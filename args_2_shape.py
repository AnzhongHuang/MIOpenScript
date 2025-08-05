from html import parser
import time
import argparse
import sys
import os
import numpy as np
import shlex
import concurrent.futures
import itertools
from enum import Enum
import miopUtil.shapeConvert as shapeConvert
from miopUtil.shapeConvert import MiopenDataType
from miopUtil.MIArgs import MIArgs

def Solve(argvs, is_solver=False):
    seri = ""
    # Parse command name to determine data type
    args = MIArgs.ParseParam(argvs[1:])
    if is_solver:
        args.shapeformat = 'solver'

    if args.shapeformat == 'vs':
        cmds, conv_type = shapeConvert.GetArgument(args)

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
            in_channels=args.in_channels,
            spatial_dims=args.spatial_dim,
            in_depth=args.in_d,
            in_height=args.in_h,
            in_width=args.in_w,
            weights_depth=args.fil_d,
            weights_height=args.fil_h,
            weights_width=args.fil_w,
            out_channels=args.out_channels,
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
            in_layout=args.in_layout if args.in_layout else 'NCHW',
            weights_layout=args.fil_layout if args.fil_layout else 'NCHW',
            out_layout=args.out_layout if args.out_layout else 'NCHW',
            in_data_type=args.in_data_type,
            weights_data_type=args.in_data_type,
            out_data_type=args.in_data_type,
            direction_str=shapeConvert.get_direction_str(args.forw),
            group_count=args.group_count
        )
        problem.InitDef()
        if args.shapeformat == 'solver':
            seri = problem.ufdbSerialize()
        elif args.shapeformat == 'kernel':
            seri = problem.udbSerialize()
    return seri


def Compare(file_name1, file_name2):
    with open(file_name1, 'r') as f1, open(file_name2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
    if len(lines1) != len(lines2):
        print(
            f"Files {file_name1} and {file_name2} have different number of lines.")
        return False
    shape1 = {}
    shape2 = {}

    start_time = time.time()
    for line1 in lines1:
        args_line = line1.strip()
        # print(args_line)
        seri = Solve(shlex.split(args_line), True)
        if seri:
            shape1[seri] = 1
    end_time = time.time()
    print(
        f"Time taken to process {file_name1}: {end_time - start_time:.4f} seconds")
    print(f"Shapes in {file_name1}: {len(shape1)}")

    start_time = time.time()
    for line2 in lines2:
        args_line = line2.strip()
        # print(args_line)
        seri = Solve(shlex.split(args_line), True)
        if seri:
            if seri in shape2:
                print(f"Duplicate: {args_line}: {seri}")
            shape2[seri] = 1
    end_time = time.time()
    print(f"Time taken to process {file_name2}: {end_time - start_time:.4f} seconds")
    print(f"Shapes in {file_name2}: {len(shape2)}")

    # Compare the two dictionaries, ignoring order
    if shape1.keys() != shape2.keys():
        print(f"Shapes in {file_name1} and {file_name2} do not match.")
        return False
    else:
        print(f"Shapes in {file_name1} and {file_name2} match.")
        return True

def Test(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    args_list = []

    # open output file
    output_file = 'test_passlist.txt'
    f = open(output_file, 'w')
    orginal_line=None
    for line in lines:
        if line.strip():
            if line.startswith('#'):
                orginal_line = line[1:].strip()
            else:
                args_line = line.strip()
                # print(args_line)
                seri = Solve(shlex.split(args_line), True)

                if orginal_line is not None:
                    if seri != orginal_line:
                        print(f"Error: {seri} != {orginal_line}")
                    else:
                        print(f"# {orginal_line}", file=f)
                        print(args_line, file=f)
                else:
                    print(f"OK: {seri} == {orginal_line}")

def main():
    seri = Solve(sys.argv[:])
    print(seri)
# python args_2_shape.py convfp16 -F 1 -n 128 -c 192 -H 1 -W 1 -k 8 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -g 1 -m conv --spatial_dim 2 -t 1 --shapeformat solver
# python args_2_shape.py convfp16 -F 1 -n 128 -c 192 -H 1 -W 1 -k 8 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -g 1 -m conv --spatial_dim 2 -t 1 --shapeformat kernel
# python args_2_shape.py convfp16 -F 1 -n 128 -c 192 -H 1 -W 1 -k 8 -y 1 -x 1 -u 1 -v 1 -l 1 -j 1 -g 1 -m conv --spatial_dim 2 -t 1 --shapeformat vs
if __name__ == "__main__":

    if sys.argv[1] == '--compare':
        if len(sys.argv) != 4:
            print("Usage: python args_2_shape.py --compare <file1> <file2>")
            sys.exit(1)
        file_name1 = sys.argv[2]
        file_name2 = sys.argv[3]
        Compare(file_name1, file_name2)
    elif sys.argv[1] == '--test':
        # print(sys.argv)
        Test(sys.argv[2])

    else:
        main()

