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

# Mapping of flags to normalized names (both short and long forms)
FLAG_MAPPING = {
    # Operation type
    '-F': 'forw', '--forw': 'forw',
    # Input tensor
    '-n': 'batchsize', '--batchsize': 'batchsize',
    '-c': 'in_channels', '--in_channels': 'in_channels',
    '-H': 'in_h', '--in_h': 'in_h',
    '-W': 'in_w', '--in_w': 'in_w',
    '-!': 'in_d', '--in_d': 'in_d',
    # Output channels
    '-k': 'out_channels', '--out_channels': 'out_channels',
    # Kernel
    '-y': 'fil_h', '--fil_h': 'fil_h',
    '-x': 'fil_w', '--fil_w': 'fil_w',
    '-@': 'fil_d', '--fil_d': 'fil_d',
    # Padding
    '-p': 'pad_h', '--pad_h': 'pad_h',
    '-q': 'pad_w', '--pad_w': 'pad_w',
    '-$': 'pad_d', '--pad_d': 'pad_d',
    # Stride
    '-u': 'conv_stride_h', '--conv_stride_h': 'conv_stride_h',
    '-v': 'conv_stride_w', '--conv_stride_w': 'conv_stride_w',
    '-#': 'conv_stride_d', '--conv_stride_d': 'conv_stride_d',
    # Dilation
    '-l': 'dilation_h', '--dilation_h': 'dilation_h',
    '-j': 'dilation_w', '--dilation_w': 'dilation_w',
    '-^': 'dilation_d', '--dilation_d': 'dilation_d',
    # Groups
    '-g': 'group_count', '--group_count': 'group_count',
    # Spatial dimension
    '-_': 'spatial_dim', '--spatial_dim': 'spatial_dim',
    # Solver and data type
    '-S': 'solution', '--solution': 'solution',
    '-t': 'time', '--time': 'time',
    '-V': 'verify', '--verify': 'verify',
    '-m': 'mode', '--mode': 'mode',
    # Data loading
    '-d': 'in_data', '--in_data': 'in_data',
    '-e': 'weights', '--weights': 'weights',
    '-D': 'dout_data', '--dout_data': 'dout_data',
    # Bias
    '-b': 'bias', '--bias': 'bias',
    '-a': 'in_bias', '--in_bias': 'in_bias',
    # Additional parameters
    '-i': 'iter', '--iter': 'iter',
    '-z': 'pad_mode', '--pad_mode': 'pad_mode',
    '-f': 'fil_layout', '--fil_layout': 'fil_layout',
    '-I': 'in_layout', '--in_layout': 'in_layout',
    '-O': 'out_layout', '--out_layout': 'out_layout',
    '-~': 'gpubuffer_check', '--gpubuffer_check': 'gpubuffer_check',
    '-w': 'wall', '--wall': 'wall',
    '-P': 'printconv', '--printconv': 'printconv',
    '-r': 'pad_val', '--pad_val': 'pad_val',
    '-o': 'dump_output', '--dump_output': 'dump_output',
    '-s': 'search', '--search': 'search',
    '-C': 'verification_cache', '--verification_cache': 'verification_cache',
    # Shape format
    '--shapeformat': 'shapeformat'
}

CONVERTERS = {
    'forw': int, 'batchsize': int, 'in_channels': int, 'in_h': int, 'in_w': int,
    'in_d': int, 'out_channels': int, 'fil_h': int, 'fil_w': int, 'fil_d': int,
    'pad_h': int, 'pad_w': int, 'pad_d': int, 'conv_stride_h': int, 'conv_stride_w': int,
    'conv_stride_d': int, 'dilation_h': int, 'dilation_w': int, 'dilation_d': int,
    'group_count': int, 'spatial_dim': int, 'solution': int, 'time': int, 'verify': int,
    'bias': int, 'iter': int, 'gpubuffer_check': int, 'wall': int, 'printconv': int,
    'pad_val': int, 'dump_output': int, 'search': int,
    # Strings (keep as-is)
    'mode': str, 'in_data': str, 'weights': str, 'dout_data': str, 'in_bias': str,
    'pad_mode': str, 'fil_layout': str, 'in_layout': str, 'out_layout': str,
    'verification_cache': str, 'shapeformat': str
}
def ParseParam(args_list):
    command_name = args_list[0] if len(args_list) > 1 else "conv"
    if "bfp16" in command_name.lower():
        in_data_type = MiopenDataType.miopenBFloat16
    elif "fp16" in command_name.lower():
        in_data_type = MiopenDataType.miopenHalf
    elif "int8" in command_name.lower():
        in_data_type = MiopenDataType.miopenInt8
    elif "conv" == command_name.lower():
        in_data_type = MiopenDataType.miopenFloat
    elif "fp64" in command_name.lower():
        in_data_type = MiopenDataType.miopenDouble
    else:
        print(f"Unknown {command_name} in command name: {args_list}")

    args={}
    idx = 1

    # default value
    args['in_d']=1
    args['fil_d']=1
    args['pad_h']=0
    args['pad_w'] = 0
    args['pad_d'] = 0
    args['conv_stride_h'] = 1
    args['conv_stride_w'] = 1
    args['conv_stride_d'] = 0
    args['dilation_h'] = 1
    args['dilation_w'] = 1
    args['dilation_d'] = 0
    args['group_count'] = 1
    args['spatial_dim'] = 2
    args['solution'] = -1
    args['time'] = 0
    args['verify'] = 1
    args['mode'] = 'conv'
    args['in_data'] = ''
    args['weights'] = ''
    args['dout_data'] = ''
    args['bias'] = 0
    args['in_bias'] = ''
    args['iter'] = 10
    args['pad_mode'] = 'default'
    args['fil_layout'] = None
    args['in_layout'] = None
    args['out_layout'] = None
    args['gpubuffer_check'] = 0
    args['wall'] = 0
    args['printconv'] = 1
    args['pad_val'] = 0
    args['dump_output'] = 0
    args['search'] = 0
    args['verification_cache'] = ''
    args['shapeformat'] = 'solver'  # default shape format

    while idx < len(args_list):
        if args_list[idx].startswith('-'):
            key = args_list[idx]
            value = args_list[idx + 1]

            norm_name = FLAG_MAPPING.get(key)
            convert = CONVERTERS.get(norm_name, str)
            args[norm_name] = convert(value)
            idx += 1

        idx += 1

    if args['in_layout'] == None:
        if args['spatial_dim'] == 2:
            args['in_layout'] = 'NCHW'
        else:
            args['in_layout'] = 'NCDHW'
    if args['fil_layout'] == None:
        args['fil_layout'] = args['in_layout']
    if args['out_layout'] == None:
        args['out_layout'] = args['in_layout']

    if 'forw' not in args:
        print(args_list)
    # parser = argparse.ArgumentParser(description='PyTorch MIOpenDriver Simulator',
    #                             add_help=False)
    # # Operation type (F flag)
    # parser.add_argument('-F', '--forw', type=int, required=True, 
    #                     choices=[0, 1, 2, 3, 4, 5, 6], 
    #                     help='Operation type: 0=all, 1=FWD, 2=BWD data, 4=BWD weight, 3=FWD+BWD, 5=FWD+WRW, 6=BWD+WRW')
    
    # # Input tensor parameters
    # parser.add_argument('-n', '--batchsize', type=int, required=True, help='Mini-batch size')
    # parser.add_argument('-c', '--in_channels', type=int, required=True, help='Input channels')
    # parser.add_argument('-H', '--in_h', type=int, required=True, help='Input height')
    # parser.add_argument('-W', '--in_w', type=int, required=True, help='Input width')
    # parser.add_argument('-!', '--in_d', type=int, default=1, help='Input depth (3D)')
    
    # # Output channels
    # parser.add_argument('-k', '--out_channels', type=int, required=True, help='Output channels')
    
    # # Kernel parameters
    # parser.add_argument('-y', '--fil_h', type=int, required=True, help='Filter height')
    # parser.add_argument('-x', '--fil_w', type=int, required=True, help='Filter width')
    # parser.add_argument('-@', '--fil_d', type=int, default=1, help='Filter depth (3D)')
    
    # # Padding parameters
    # parser.add_argument('-p', '--pad_h', type=int, default=0, help='Vertical padding')
    # parser.add_argument('-q', '--pad_w', type=int, default=0, help='Horizontal padding')
    # parser.add_argument('-$', '--pad_d', type=int, default=0, help='Depth padding (3D)')
    
    # # Stride parameters
    # parser.add_argument('-u', '--conv_stride_h', type=int, default=1, help='Vertical stride')
    # parser.add_argument('-v', '--conv_stride_w', type=int, default=1, help='Horizontal stride')
    # parser.add_argument('-#', '--conv_stride_d', type=int, default=0, help='Depth stride (3D)')
    
    # # Dilation parameters
    # parser.add_argument('-l', '--dilation_h', type=int, default=1, help='Vertical dilation')
    # parser.add_argument('-j', '--dilation_w', type=int, default=1, help='Horizontal dilation')
    # parser.add_argument('-^', '--dilation_d', type=int, default=0, help='Depth dilation (3D)')
    
    # # Groups
    # parser.add_argument('-g', '--group_count', type=int, default=1, help='Number of groups')
    
    # # Spatial dimension
    # parser.add_argument('-_', '--spatial_dim', type=int, default=2, choices=[2, 3], 
    #                     help='Convolution spatial dimension (2=2D, 3=3D)')
    
    # # Solver and data type
    # parser.add_argument('-S', '--solution', type=int, default=-1, help='Solution ID')
    # parser.add_argument('-t', '--time', type=int, default=0,
    #                     help='Print time in milliseconds')
    # parser.add_argument('-V', '--verify', type=int, default=1, 
    #                     help='Verification mode (0=no, 1=yes)')
    # parser.add_argument('-m', '--mode', default='conv', 
    #                     choices=['conv', 'trans'], help='Convolution mode')
    
    # # Data loading
    # parser.add_argument('-d', '--in_data', type=str, default='', help='Input data filename')
    # parser.add_argument('-e', '--weights', type=str, default='', help='Input weights filename')
    # parser.add_argument('-D', '--dout_data', type=str, default='', help='dy data filename for BWD weight')
    
    # # Bias
    # parser.add_argument('-b', '--bias', type=int, default=0, help='Use bias')
    # parser.add_argument('-a', '--in_bias', type=str, default='', help='Input bias filename')

    # # Additional MIOpenDriver parameters
    # parser.add_argument('-i', '--iter', type=int, default=10, help='Number of iterations')
    # parser.add_argument('-z', '--pad_mode', type=str, default='default', 
    #                     choices=['default', 'same', 'valid'], help='Padding mode')
    # parser.add_argument('-f', '--fil_layout', type=str, default='', help='Filter layout')
    # parser.add_argument('-I', '--in_layout', type=str, default='', help='Input layout')
    # parser.add_argument('-O', '--out_layout', type=str, default='', help='Output layout')
    # parser.add_argument('-~', '--gpubuffer_check', type=int, default=0, 
    #                     help='GPU buffer sanitation check')
    # parser.add_argument('-w', '--wall', type=int, default=0, 
    #                     help='Wall-clock time measurement')
    # parser.add_argument('-P', '--printconv', type=int, default=1, 
    #                     help='Print convolution dimensions')
    # parser.add_argument('-r', '--pad_val', type=int, default=0, help='Padding value')
    # parser.add_argument('-o', '--dump_output', type=int, default=0, 
    #                     help='Dump output buffers')
    # parser.add_argument('-s', '--search', type=int, default=0, 
    #                     help='Search kernel config')
    # parser.add_argument('-C', '--verification_cache', type=str, default='', 
    #                     help='Verification cache directory')

    # # Trace option
    # parser.add_argument('--shapeformat', type=str, default='vs',  choices=['vs', 'solver', 'kernel'],
    #                     help='Shape type to convert: vs=vsdb, solver=solverdb, kernel')

    # # Parse known args (ignore extra)
    # args, _ = parser.parse_known_args(args_list)

    miargs = MIArgs(
        forw=args["forw"],
        batchsize=args["batchsize"],
        in_channels=args["in_channels"],
        in_h=args["in_h"],
        in_w=args["in_w"],
        in_d=args["in_d"],
        out_channels=args["out_channels"],
        fil_h=args["fil_h"],
        fil_w=args["fil_w"],
        fil_d=args["fil_d"],
        pad_h=args["pad_h"],
        pad_w=args["pad_w"],
        pad_d=args["pad_d"],
        conv_stride_h=args["conv_stride_h"],
        conv_stride_w=args["conv_stride_w"],
        conv_stride_d=args["conv_stride_d"],
        dilation_h=args["dilation_h"],
        dilation_w=args["dilation_w"],
        dilation_d=args["dilation_d"],
        group_count=args["group_count"],
        spatial_dim=args["spatial_dim"],
        solution=args["solution"],
        time=args["time"],
        verify=args["verify"],
        mode=args["mode"],
        in_data=args["in_data"],
        weights=args["weights"],
        dout_data=args["dout_data"],
        bias=args["bias"],
        in_bias=args["in_bias"],
        iter=args["iter"],
        pad_mode=args["pad_mode"],
        fil_layout=args["fil_layout"],
        in_layout=args["in_layout"],
        out_layout=args["out_layout"],
        gpubuffer_check=args["gpubuffer_check"],
        wall=args["wall"],
        printconv=args["printconv"],
        pad_val=args["pad_val"],
        dump_output=args["dump_output"],
        search=args["search"],
        verification_cache=args["verification_cache"],

        # private fields
        shapeformat=args["shapeformat"],
        in_data_type=in_data_type
    )

    return miargs

def Solve(argvs, is_solver=False):
    seri = ""
    # Parse command name to determine data type
    args = ParseParam(argvs[1:])
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

