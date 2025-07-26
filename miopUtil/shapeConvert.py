from enum import Enum
from miopUtil.MIArgs import MiopenDataType, get_data_type_name, get_direction_str
from miopUtil.MIArgs import MIArgs

def print_dhw(sep, spatial_dims, depth, height, width):
    def inner(stream):
        components = []
        if spatial_dims > 2:
            components.append(str(depth))
        components.extend([str(height), str(width)])
        stream.append(sep.join(components))
    return inner

def print_hwd(sep, spatial_dims, height, width, depth):
    def inner(stream):
        components = []
        components.extend([str(height), str(width), str(depth)])
        stream.append(sep.join(components))
    return inner

MemLayout = ["NCHW", "CNHW", "NHWC", "CHWN", "HWCH", "HWNC", "NGCHW", "CGNHW", "GCNHW",
             "NCDHW", "CDNHW", "NDHWC", "DCHWN", "DHNWC", "DHNCW", "NCHWD", "CHNWD",
             "NHWCD", "HWNCD", "NCDHW", "CDHNW", "NDHWC", "DCHWN", "DHNWC", "DHNCW"]

def GetArgument(args):
    # Determine data type
    if args.in_data_type == MiopenDataType.miopenHalf:
        type_str = "fp16"
    elif args.in_data_type == MiopenDataType.miopenBFloat16:
        type_str = "bfp16"
    elif args.in_data_type == MiopenDataType.miopenInt8:
        type_str = "int8"
    else:
        type_str = ""  # fp32 is default

    # Determine spatial dimension
    is_3d = args.spatial_dim == 3
    tensor_dim = 5 if is_3d else 4

    cmds = []
    # Generate equivalent MIOpenDriver command
    conv_type = (f"conv{type_str}")
    cmds.append("-F")
    cmds.append(str(args.forw))
    cmds.append("-n")
    cmds.append(str(args.batchsize))
    cmds.append("-c")
    cmds.append(str(args.in_channels))

    if is_3d:
        cmds.append("--in_d")
        cmds.append(str(args.in_d))

    cmds.append("--in_h")
    cmds.append(str(args.in_h))
    cmds.append("--in_w")
    cmds.append(str(args.in_w))
    cmds.append("-k")
    cmds.append(str(args.out_channels))

    if is_3d:
        cmds.append("--fil_d")
        cmds.append(str(args.fil_d))

    cmds.append("-y")
    cmds.append(str(args.fil_h))
    cmds.append("-x")
    cmds.append(str(args.fil_w))

    if is_3d:
        cmds.append("--pad_d")
        cmds.append(str(args.pad_d))

    cmds.append("--pad_h")
    cmds.append(str(args.pad_h))
    cmds.append("--pad_w")
    cmds.append(str(args.pad_w))
    
    if is_3d:
        cmds.append("--conv_stride_d")
        cmds.append(str(args.conv_stride_d))

    cmds.append("-u")
    cmds.append(str(args.conv_stride_h))
    cmds.append("-v")
    cmds.append(str(args.conv_stride_w))

    if is_3d:
        cmds.append("--dilation_d")
        cmds.append(str(args.dilation_d))
    cmds.append("-l")
    cmds.append(str(args.dilation_h))
    cmds.append("-j")
    cmds.append(str(args.dilation_w))
    
    cmds.append("-g")
    cmds.append(str(args.group_count))
    cmds.append("-m")
    cmds.append(args.mode)
    cmds.append("--spatial_dim")
    cmds.append(str(args.spatial_dim))

    cmds.append("-t")
    cmds.append(str(args.time))
    if args.solution != -1:
        cmds.append("-S")
        cmds.append(str(args.solution))
    
    if (args.in_layout != "NCHW" and args.in_layout != "NCDHW"):
        cmds.append("--in_layout")
        cmds.append(args.in_layout)
    if (args.fil_layout != "NCHW" and args.fil_layout != "NCDHW"):
        cmds.append("--fil_layout")
        cmds.append(args.fil_layout)
    if (args.out_layout != "NCHW" and args.out_layout != "NCDHW"):
        cmds.append("--out_layout")
        cmds.append(args.out_layout)
        
    # Add file parameters if specified
    if args.in_data:
        cmds.append("--in_data")
        cmds.append(args.in_data)
    if args.weights:
        cmds.append("-e")
        cmds.append(args.weights)
    if args.dout_data:
        cmds.append("-D")
        cmds.append(args.dout_data)
    if args.in_bias:
        cmds.append("-a")
        cmds.append(args.in_bias)
    return cmds, conv_type

class ProblemDescription:
    def __init__(self, 
                in_channels =0,
                spatial_dims = 2,
                in_depth = 1,
                in_height = 1,
                in_width = 1,
                weights_depth = 1,
                weights_height = 1,
                weights_width = 1,
                out_channels = 0,
                out_depth = 0,
                out_height = 0,
                out_width = 0,
                in_batch_size = 1,
                pad_d = 0,
                pad_h = 0,
                pad_w = 0,
                kernel_stride_d = 1,
                kernel_stride_h = 1,
                kernel_stride_w = 1,
                dilation_d = 1,
                dilation_h = 1,
                dilation_w = 1,
                bias = 0,
                in_layout = "NCHW",
                weights_layout = "NCHW",
                out_layout = "NCHW",
                in_data_type = MiopenDataType.miopenFloat,
                weights_data_type = MiopenDataType.miopenFloat,
                out_data_type = MiopenDataType.miopenFloat,
                direction_str = "F",
                group_count = 1,
                 in_cast_type=None, weights_cast_type=None, out_cast_type=None):
        self.in_channels = in_channels
        self.spatial_dims = spatial_dims
        self.in_depth = in_depth
        self.in_height = in_height
        self.in_width = in_width
        self.weights_depth = weights_depth
        self.weights_height = weights_height
        self.weights_width = weights_width
        self.out_channels = out_channels
        self.out_depth = out_depth
        self.out_height = out_height
        self.out_width = out_width
        self.in_batch_size = in_batch_size
        self.pad_d = pad_d
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.kernel_stride_d = kernel_stride_d
        self.kernel_stride_h = kernel_stride_h
        self.kernel_stride_w = kernel_stride_w
        self.dilation_d = dilation_d
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.bias = bias
        self.in_layout = in_layout
        self.weights_layout = weights_layout
        self.out_layout = out_layout
        if ('D' not in self.in_layout and spatial_dims == 3):
            self.in_layout = "NCDHW"
            self.weights_layout = "NCDHW"
            self.out_layout = "NCDHW"

        self.in_data_type = in_data_type
        self.weights_data_type = weights_data_type
        self.out_data_type = out_data_type
        self.direction_str = direction_str
        self.group_count = group_count
        self.in_cast_type = in_cast_type
        self.weights_cast_type = weights_cast_type
        self.out_cast_type = out_cast_type

    @staticmethod
    def Problem2MIArgs(problem):
        """
        Convert the problem description to MIArgs.
        """
        return MIArgs(
            forw=1 if problem.direction_str == "F" else 2 if problem.direction_str == "B" else 4,
            batchsize=problem.in_batch_size,
            in_channels=problem.in_channels,
            in_h=problem.in_height,
            in_w=problem.in_width,
            in_d=problem.in_depth,
            out_channels=problem.out_channels,
            fil_h=problem.weights_height,
            fil_w=problem.weights_width,
            fil_d=problem.weights_depth,
            pad_h=problem.pad_h,
            pad_w=problem.pad_w,
            pad_d=problem.pad_d,
            conv_stride_h=problem.kernel_stride_h,
            conv_stride_w=problem.kernel_stride_w,
            conv_stride_d=problem.kernel_stride_d,
            dilation_h=problem.dilation_h,
            dilation_w=problem.dilation_w,
            dilation_d=problem.dilation_d,
            group_count=problem.group_count,
            spatial_dim=problem.spatial_dims,
            solution=-1,  # Default value, can be set later
            time=1,  # Default value, can be set later
            verify=1,  # Default value, can be set later
            mode='conv',  # Default mode
            in_data='',  # Default value, can be set later
            weights='',  # Default value, can be set later
            dout_data='',  # Default value, can be set later
            bias=problem.bias,
            in_bias='',  # Default value, can be set later
            iter=10,  # Default iterations
            pad_mode='default',  # Default padding mode
            fil_layout=problem.weights_layout,
            in_layout=problem.in_layout,
            out_layout=problem.out_layout,
            gpubuffer_check=0,  # Default GPU buffer check
            wall=0,  # Default wall time
            printconv=1,  # Print convolution details by default
            pad_val=0,  # Default padding value
            dump_output=0,  # Do not dump output by default
            search=0,  # Not searching by default
            verification_cache='',  # No verification cache by default
            trace='',  # No trace by default
            event='' ,# No event by default
            in_data_type=problem.in_data_type
        )

    @staticmethod
    def MIArgs2Problem(mi_args):
        """
        Convert MIArgs back to ProblemDescription.
        """
        problem = ProblemDescription(
            in_channels=mi_args.in_channels,
            spatial_dims=mi_args.spatial_dim,
            in_depth=mi_args.in_d,
            in_height=mi_args.in_h,
            in_width=mi_args.in_w,
            weights_depth=mi_args.fil_d,
            weights_height=mi_args.fil_h,
            weights_width=mi_args.fil_w,
            out_channels=mi_args.out_channels,
            out_depth=0,  # Not specified in MIArgs
            out_height=0,  # Not specified in MIArgs
            out_width=0,  # Not specified in MIArgs
            in_batch_size=mi_args.batchsize,
            pad_d=mi_args.pad_d,
            pad_h=mi_args.pad_h,
            pad_w=mi_args.pad_w,
            kernel_stride_d=mi_args.conv_stride_d,
            kernel_stride_h=mi_args.conv_stride_h,
            kernel_stride_w=mi_args.conv_stride_w,
            dilation_d=mi_args.dilation_d,
            dilation_h=mi_args.dilation_h,
            dilation_w=mi_args.dilation_w,
            bias=mi_args.bias,
            in_layout=mi_args.in_layout,
            weights_layout=mi_args.fil_layout,
            out_layout=mi_args.out_layout,
            in_data_type=mi_args.in_data_type,
            weights_data_type=mi_args.weights_data_type,
            out_data_type=mi_args.out_data_type,
            direction_str=get_direction_str(mi_args.forw),
            group_count=mi_args.group_count
        )
        problem.InitDef()  # Initialize derived attributes
        return problem

    def InitDef(self):
        def calc_output_size(in_size, pad, kernel, stride, dilation=1):
            return (in_size + 2*pad - dilation*(kernel - 1) - 1) // stride + 1
        if self.out_width == 0:
            self.out_width = calc_output_size(self.in_width, self.pad_w, self.weights_width, self.kernel_stride_w, self.dilation_w)
        self.out_width = max(self.out_width, 1)

        if self.out_height == 0:
            self.out_height = calc_output_size(self.in_height, self.pad_h, self.weights_height, self.kernel_stride_h, self.dilation_h)

        self.out_height = max(self.out_height, 1)
        kernel_stride_d = self.kernel_stride_d if self.kernel_stride_d > 0 else 1
        if self.out_depth == 0:
            self.out_depth = calc_output_size(self.in_depth, self.pad_d, self.weights_depth, kernel_stride_d, self.dilation_d)

        self.out_depth = max(self.out_depth, 0)
        
    def ufdbSerialize(self):
        sep = '-'
        stream = []

        if self.direction_str == "F":
            stream.append(str(self.in_channels))
            print_dhw(sep, self.spatial_dims, self.in_depth, self.in_height, self.in_width)(stream)
        else:
            stream.append(str(self.out_channels))
            print_dhw(sep, self.spatial_dims, self.out_depth, self.out_height, self.out_width)(stream)
  
        print_dhw('x', self.spatial_dims, self.weights_depth, self.weights_height, self.weights_width)(stream)

        if self.direction_str == "F":
            stream.append(str(self.out_channels))
            print_dhw(sep, self.spatial_dims, self.out_depth, self.out_height, self.out_width)(stream)
        else:
            stream.append(str(self.in_channels))
            print_dhw(sep, self.spatial_dims, self.in_depth, self.in_height, self.in_width)(stream)
  
        stream.append(str(self.in_batch_size))
        print_dhw('x', self.spatial_dims, self.pad_d, self.pad_h, self.pad_w)(stream)
        print_dhw('x', self.spatial_dims, self.kernel_stride_d, self.kernel_stride_h, self.kernel_stride_w)(stream)
        print_dhw('x', self.spatial_dims, self.dilation_d, self.dilation_h, self.dilation_w)(stream)
        stream.append(str(self.bias))

        if (self.in_layout in ["NCHW", "NCDHW"] and 
            self.weights_layout == self.in_layout and 
            self.out_layout == self.in_layout):
            stream.append(self.in_layout)
        else:
            stream.extend([self.in_layout, self.weights_layout, self.out_layout])

        if self.in_data_type == self.weights_data_type == self.out_data_type:
            encoded_data_types = get_data_type_name(self.in_data_type)
        else:
            encoded_data_types = f"{get_data_type_name(self.in_data_type)}{get_data_type_name(self.weights_data_type)}{get_data_type_name(self.out_data_type)}"
        stream.append(encoded_data_types)
        stream.append(self.direction_str)

        optional = []

        if self.group_count != 1:
            optional.append(f"_g{self.group_count}")

        if self.in_cast_type is not None:
            optional.append(f"_ci{get_data_type_name(self.in_cast_type)}")
        if self.weights_cast_type is not None:
            optional.append(f"_cw{get_data_type_name(self.weights_cast_type)}")
        if self.out_cast_type is not None:
            optional.append(f"_co{get_data_type_name(self.out_cast_type)}")

        res = '-'.join(stream)
        if optional:
            res += "".join(optional)

        return res

    # kernel shape serialization for UDB format
    # static void Visit(Self&& self, std::function<void(int64_t, std::string)> f)
    def udbSerialize(self):
        sep = 'x'
        stream = []
        stream.append(str(self.spatial_dims))
        stream.append(str(self.in_channels))
        print_hwd('x', self.spatial_dims, self.in_height, self.in_width, self.in_depth)(stream)
        print_hwd('x', self.spatial_dims, self.weights_height, self.weights_width, self.weights_depth)(stream)
        stream.append(str(self.out_channels))
        # print_dhw(sep, self.spatial_dims, self.out_depth, self.out_height, self.out_width)(stream)
        stream.append(str(self.in_batch_size))
        print_hwd('x', self.spatial_dims, self.pad_h, self.pad_w, self.pad_d)(stream)
        print_hwd('x', self.spatial_dims, self.kernel_stride_h, self.kernel_stride_w, self.kernel_stride_d)(stream)
        print_hwd('x', self.spatial_dims, self.dilation_h, self.dilation_w, self.dilation_d)(stream)
        stream.append(str(self.bias))
        stream.append(str(self.group_count))
        stream.append(self.in_layout)

        if self.in_data_type == self.weights_data_type == self.out_data_type:
            encoded_data_types = get_data_type_name(self.in_data_type)
        else:
            encoded_data_types = f"{get_data_type_name(self.in_data_type)}{get_data_type_name(self.weights_data_type)}{get_data_type_name(self.out_data_type)}"
        stream.append(encoded_data_types)
        stream.append(self.direction_str)

        return 'x'.join(stream)


    @staticmethod
    def udbDeserialize(serialized_str):
        """
        Deserialize a UDB format string into a ProblemDescription object.
        """
        parts = serialized_str.split('x')

        idx = 0
        spatial_dims = int(parts[idx]); idx += 1
        in_channels = int(parts[idx]); idx += 1

        in_height = int(parts[idx]); idx += 1
        in_width = int(parts[idx]); idx += 1
        in_depth = int(parts[idx]); idx += 1

        weights_height = int(parts[idx]); idx += 1
        weights_width = int(parts[idx]); idx += 1
        weights_depth = int(parts[idx]); idx += 1

        out_channels = int(parts[idx]); idx += 1
        in_batch_size = int(parts[idx]); idx += 1

        pad_h = int(parts[idx]); idx += 1
        pad_w = int(parts[idx]); idx += 1
        pad_d = int(parts[idx]); idx += 1

        kernel_stride_h = int(parts[idx]); idx += 1
        kernel_stride_w = int(parts[idx]); idx += 1
        kernel_stride_d = int(parts[idx]); idx += 1

        dilation_h = int(parts[idx]); idx += 1
        dilation_w = int(parts[idx]); idx += 1
        dilation_d = int(parts[idx]); idx += 1

        bias = int(parts[idx]); idx += 1
        group_count = int(parts[idx]); idx += 1
        in_layout = parts[idx]; idx += 1
        
        weights_layout = out_layout = in_layout

        # Parse data types
        data_type_str = parts[idx]; idx += 1
        data_types = []

        # Map data type strings back to enum values
        data_type_map = {
            "FP16": MiopenDataType.miopenHalf,
            "FP32": MiopenDataType.miopenFloat,
            "INT32": MiopenDataType.miopenInt32,
            "INT8": MiopenDataType.miopenInt8,
            "BF16": MiopenDataType.miopenBFloat16,
            "FP64": MiopenDataType.miopenDouble,
            "FP8": MiopenDataType.miopenFloat8_fnuz,
            "BF8": MiopenDataType.miopenBFloat8_fnuz,
            "INT64": MiopenDataType.miopenInt64
        }

        beginning_letter = ['F', 'B', 'I']
        # split data type string into parts based on the first letter
        data_type_parts = []
        current_part = ""
        for char in data_type_str:
            if char in beginning_letter and len(current_part) > 0:
                data_type_parts.append(current_part)
                current_part = char
            else:
                current_part += char

        if len(current_part) > 0:
            data_type_parts.append(current_part)

        # Convert each part to its corresponding enum value
        in_data_type = data_type_map.get(data_type_parts[0], MiopenDataType.miopenFloat)
        if len(data_type_parts) > 2:
            weights_data_type = data_type_map.get(data_type_parts[1], in_data_type)
            out_data_type     = data_type_map.get(data_type_parts[2], in_data_type)
        else:
            out_data_type = in_data_type
            weights_data_type = in_data_type

        direction_str = parts[idx]; idx += 1

        # Create and return problem description
        problem = ProblemDescription(
            in_channels=in_channels,
            spatial_dims=spatial_dims,
            in_depth=in_depth,
            in_height=in_height,
            in_width=in_width,
            weights_depth=weights_depth,
            weights_height=weights_height,
            weights_width=weights_width,
            out_channels=out_channels,
            out_depth=0,
            out_height=0,
            out_width=0,
            in_batch_size=in_batch_size,
            pad_d=pad_d,
            pad_h=pad_h,
            pad_w=pad_w,
            kernel_stride_d=kernel_stride_d,
            kernel_stride_h=kernel_stride_h,
            kernel_stride_w=kernel_stride_w,
            dilation_d=dilation_d,
            dilation_h=dilation_h,
            dilation_w=dilation_w,
            bias=bias,
            in_layout=in_layout,
            weights_layout=weights_layout,
            out_layout=out_layout,
            in_data_type=in_data_type,
            weights_data_type=weights_data_type,
            out_data_type=out_data_type,
            direction_str=direction_str,
            group_count=group_count
        )

        problem.InitDef()
        return problem

    @staticmethod
    def ufdbDeserialize(problem):
        """
        Deserialize a UFDB format string into a ProblemDescription object.
        """
        parts = problem.split('-')

        idx = 0
        in_channels = int(parts[idx]); idx += 1

        spatial_dims = 3
        if 'x' in parts[idx+2]:
            spatial_dims = 2
            in_height = int(parts[idx]); idx += 1
            in_width = int(parts[idx]); idx += 1
            in_depth = 1
        else:
            in_depth = int(parts[idx]); idx += 1
            in_height = int(parts[idx]); idx += 1
            in_width = int(parts[idx]); idx += 1

        if spatial_dims == 2:
            weights_depth = 1
            weights_height, weights_width = map(int, parts[idx].split('x')); idx += 1
        else:
            weights_depth, weights_height, weights_width = map(int, parts[idx].split('x')); idx += 1

        out_channels = int(parts[idx]); idx += 1
        if spatial_dims == 2:
            out_depth = 1
            out_height = int(parts[idx]); idx += 1
            out_width = int(parts[idx]); idx += 1
        else:
            out_depth = int(parts[idx]); idx += 1
            out_height = int(parts[idx]); idx += 1
            out_width = int(parts[idx]); idx += 1

        in_batch_size = int(parts[idx]); idx += 1

        if spatial_dims == 2:
            pad_d = 0
            pad_h, pad_w = map(int, parts[idx].split('x')); idx += 1
        else:
            pad_d, pad_h, pad_w = map(int, parts[idx].split('x')); idx += 1
        
        if spatial_dims == 2:
            kernel_stride_d = 1
            kernel_stride_h, kernel_stride_w = map(int, parts[idx].split('x')); idx += 1
        else:
            kernel_stride_d, kernel_stride_h, kernel_stride_w = map(int, parts[idx].split('x')); idx += 1
        if spatial_dims == 2:
            dilation_d = 1
            dilation_h, dilation_w = map(int, parts[idx].split('x')); idx += 1
        else:
            dilation_d, dilation_h, dilation_w = map(int, parts[idx].split('x')); idx += 1

        bias = int(parts[idx]); idx += 1

        layouts = parts[idx:idx+3]
        if layouts[0] in MemLayout:
            in_layout = layouts[0]
            idx += 1
        if len(layouts) > 2 and layouts[1] in MemLayout:
            weights_layout = layouts[1]
            out_layout    = layouts[2]
            idx += 2
        else:
            weights_layout = out_layout = in_layout

        data_types_str = parts[idx]; idx += 1
        data_types_map = {
            "FP16": MiopenDataType.miopenHalf,
            "FP32": MiopenDataType.miopenFloat,
            "INT32": MiopenDataType.miopenInt32,
            "INT8": MiopenDataType.miopenInt8,
            "BF16": MiopenDataType.miopenBFloat16,
            "FP64": MiopenDataType.miopenDouble,
            "FP8": MiopenDataType.miopenFloat8_fnuz,
            "BF8": MiopenDataType.miopenBFloat8_fnuz,
            "INT64": MiopenDataType.miopenInt64
        }

        beginning_letter = ['F', 'B', 'I']
        # split data type string into parts based on the first letter
        data_type_parts = []
        current_part = ""
        for char in data_types_str:
            if char in beginning_letter and len(current_part) > 2:
                data_type_parts.append(current_part)
                current_part = char
            else:
                current_part += char

        if len(current_part) > 2:
            data_type_parts.append(current_part)

        # Convert each part to its corresponding enum value
        in_data_type = data_types_map.get(data_type_parts[0], MiopenDataType.miopenFloat)
        if len(data_type_parts) > 2:
            weights_data_type = data_types_map.get(data_type_parts[1], in_data_type)
            out_data_type     = data_types_map.get(data_type_parts[2], in_data_type)
        else:
            out_data_type = in_data_type
            weights_data_type = in_data_type

        last_part = parts[idx]; idx += 1
        parts = last_part.split('_')
        idx = 0

        direction_str = parts[idx]; idx += 1

        # _g{group_count} optional part
        group_count = 1
        if idx < len(parts) and parts[idx].startswith('g'):
            group_count = int(parts[idx][1:])
            idx += 1
        # _ci{in_cast_type}, _cw{weights_cast_type}, _co{out_cast_type} optional parts
        in_cast_type = weights_cast_type = out_cast_type = None
        if idx < len(parts):
            if parts[idx].startswith('ci'):
                in_cast_type = data_types_map.get(parts[idx][2:], MiopenDataType.miopenFloat)
                idx += 1
            if idx < len(parts) and parts[idx].startswith('cw'):
                weights_cast_type = data_types_map.get(parts[idx][2:], MiopenDataType.miopenFloat)
                idx += 1
            if idx < len(parts) and parts[idx].startswith('co'):
                out_cast_type = data_types_map.get(parts[idx][2:], MiopenDataType.miopenFloat)
                idx += 1

        # Create and return problem description
        problem = ProblemDescription(
            in_channels=in_channels if direction_str == "F" else out_channels,
            spatial_dims=spatial_dims,
            in_depth=in_depth if direction_str == "F" else out_depth,
            in_height=in_height if direction_str == "F" else out_height,
            in_width=in_width if direction_str == "F" else out_width,
            weights_depth=weights_depth,
            weights_height=weights_height,
            weights_width=weights_width,
            out_channels=out_channels if direction_str == "F" else in_channels,
            out_depth=out_depth if direction_str == "F" else in_depth,
            out_height=out_height if direction_str == "F" else in_height,
            out_width=out_width if direction_str == "F" else in_width,
            in_batch_size=in_batch_size,
            pad_d=pad_d,
            pad_h=pad_h,
            pad_w=pad_w,
            kernel_stride_d=kernel_stride_d,
            kernel_stride_h=kernel_stride_h,
            kernel_stride_w=kernel_stride_w,
            dilation_d=dilation_d,
            dilation_h=dilation_h,
            dilation_w=dilation_w,
            bias=bias,
            in_layout=in_layout,
            weights_layout=weights_layout,
            out_layout=out_layout,
            in_data_type=in_data_type,
            weights_data_type=weights_data_type,
            out_data_type=out_data_type,
            direction_str=direction_str,
            group_count=group_count,
            in_cast_type=in_cast_type,
            weights_cast_type=weights_cast_type,
            out_cast_type=out_cast_type
        )
        # problem.InitDef()
        return problem
    
    def test(self):
        """
        Test the ProblemDescription class.
        """
        serialized = self.ufdbSerialize()
        # print(f"Serialized UFDB: {serialized}")
        
        deserialized = ProblemDescription.ufdbDeserialize(serialized)
        # print(f"Deserialized UFDB: {deserialized.ufdbSerialize()}")
        if deserialized.in_data_type != self.in_data_type:
            print(f"ufdb test fail: {deserialized.in_data_type} != {self.in_data_type}")
        else:
            print("ufdb test pass")

        serialized_udb = self.udbSerialize()
        # print(f"Serialized UDB: {serialized_udb}")
        
        deserialized_udb = ProblemDescription.udbDeserialize(serialized_udb)
        # print(f"Deserialized UDB: {deserialized_udb.udbSerialize()}")
        if deserialized_udb.in_data_type != self.in_data_type:
            print(f"udb test fail: {deserialized_udb.in_data_type} != {self.in_data_type}")
        else:
            print("udb test pass")