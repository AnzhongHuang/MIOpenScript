from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

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


@dataclass
class MIArgs:
    forw: int
    batchsize: int
    in_channels: int
    in_h: int
    in_w: int
    in_d: int = 1
    out_channels: int = 0
    fil_h: int = 0
    fil_w: int = 0
    fil_d: int = 1
    pad_h: int = 0
    pad_w: int = 0
    pad_d: int = 0
    conv_stride_h: int = 1
    conv_stride_w: int = 1
    conv_stride_d: int = 0
    dilation_h: int = 1
    dilation_w: int = 1
    dilation_d: int = 0
    group_count: int = 1
    spatial_dim: int = 2
    solution: int = -1
    time: int = 0
    verify: int = 1
    mode: str = 'conv'
    in_data: str = ''
    weights: str = ''
    dout_data: str = ''
    bias: int = 0
    in_bias: str = ''
    iter: int = 10
    pad_mode: str = 'default'
    fil_layout: str = ''
    in_layout: str = ''
    out_layout: str = ''
    gpubuffer_check: int = 0
    wall: int = 0
    printconv: int = 1
    pad_val: int = 0
    dump_output: int = 0
    search: int = 0
    verification_cache: str = ''

    # private fields
    trace: str = ''
    event: str = ''
    warmup: int = 5
    gpu: int = 0
    dbshape: int = 0

    in_data_type: MiopenDataType = MiopenDataType.miopenHalf
    shapeformat: str = 'vs'