from enum import Enum

def print_dhw(sep, spatial_dims, depth, height, width):
    def inner(stream):
        components = []
        if spatial_dims > 2:
            components.append(str(depth))
        components.extend([str(height), str(width)])
        stream.append(sep.join(components))
    return inner

class MiopenDataType(Enum):
    miopenHalf = 0  # 16-bit floating point (Fully supported)
    miopenFloat = 1  # 32-bit floating point (Fully supported)
    miopenInt32 = 2  # 32-bit integer (Partially supported)
    miopenInt8 = 3  # 8-bit integer (Partially supported)
    # miopenInt8x4 = 4  # Pack of 4x Int8 in NCHW_VECT_C format (Support discontinued)
    miopenBFloat16 = 5  # 16-bit binary floating point (8-bit exponent, 7-bit fraction) (Partially supported)
    miopenDouble = 6  # 64-bit floating point (Partially supported)
    miopenFloat8_fnuz = 7
    miopenBFloat8_fnuz = 8
    miopenInt64 = 9

def get_data_type_name(data_type):
    return {
        MiopenDataType.miopenHalf:          "FP16",
        MiopenDataType.miopenFloat:         "FP32",
        MiopenDataType.miopenInt32:         "INT32",
        MiopenDataType.miopenInt8:          "INT8",
        MiopenDataType.miopenBFloat16:      "BF16",
        MiopenDataType.miopenDouble:        "FP64",
        MiopenDataType.miopenFloat8_fnuz:   "FP8",
        MiopenDataType.miopenBFloat8_fnuz:  "BF8",
        MiopenDataType.miopenInt64:         "INT64"
    }.get(data_type, f"Unknown({data_type})")

def get_direction_str(forw):
    if forw == 1:
        return "F"
    elif forw == 2:
        return "B"
    elif forw == 4:
        return "W"
    else:
        return "unknown"

class ProblemDescription:
    def __init__(self, in_channels, spatial_dims, in_depth, in_height, in_width,
                 weights_depth, weights_height, weights_width, out_channels,
                 out_depth, out_height, out_width, in_batch_size, pad_d, pad_h, pad_w,
                 kernel_stride_d, kernel_stride_h, kernel_stride_w, dilation_d, dilation_h,
                 dilation_w, bias, in_layout, weights_layout, out_layout, in_data_type,
                 weights_data_type, out_data_type, direction_str, group_count=1,
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
        self.in_data_type = in_data_type
        self.weights_data_type = weights_data_type
        self.out_data_type = out_data_type
        self.direction_str = direction_str
        self.group_count = group_count
        self.in_cast_type = in_cast_type
        self.weights_cast_type = weights_cast_type
        self.out_cast_type = out_cast_type

    def serialize(self):
        sep = '-'
        stream = []

        stream.append(str(self.in_channels))
        print_dhw(sep, self.spatial_dims, self.in_depth, self.in_height, self.in_width)(stream)
        print_dhw('x', self.spatial_dims, self.weights_depth, self.weights_height, self.weights_width)(stream)
        stream.append(str(self.out_channels))
        print_dhw(sep, self.spatial_dims, self.out_depth, self.out_height, self.out_width)(stream)
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
            encoded_data_types = f"{get_data_type_name(self.in_data_type)}-{get_data_type_name(self.weights_data_type)}-{get_data_type_name(self.out_data_type)}"
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

        if optional:
            stream.append(''.join(optional))

        return '-'.join(stream)
