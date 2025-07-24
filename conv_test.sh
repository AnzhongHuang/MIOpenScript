#!/bin/bash

# Start time measurement
START_TIME=$(date +%s)

MIOpenDriver="MIOpenDriver"
TRACE=""
ALL_TEST=true
G1F2_TEST=false
CONV3D_TEST=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --trace)
            MIOpenDriver="python MIOpenDriver.py"
            TRACE+=" --trace $2"
            shift
            ;;
        --event)
            MIOpenDriver="python MIOpenDriver.py"
            TRACE+=" --event $2"
            shift
            ;;
        --g1f2)
            if [[ "$2" == "1" ]]; then
                ALL_TEST=false
                G1F2_TEST=true
                shift
            fi
            ;;
        --conv3d)
            if [[ "$2" == "1" ]]; then
                ALL_TEST=false
                CONV3D_TEST=true
                shift
            fi
            ;;
        *)
            # Forward other arguments to the driver commands
            break
            ;;
    esac
    shift
done

#export MIOPEN_ENABLE_LOGGING=1
#export MIOPEN_ENABLE_LOGGING_CMD=1
#export MIOPEN_LOG_LEVEL=100
export  HIP_VISIBLE_DEVICES=6
export MIOPEN_DRIVER_USE_GPU_REFERENCE=0
#LAYOUT="--in_layout NHWC --out_layout NHWC --fil_layout NHWC"
LAYOUT=""

if [ "$ALL_TEST" = true ] || [ "$G1F2_TEST" = true ]; then
    echo "Test -g 1 -F 2"
    $MIOpenDriver convfp16 -n 128 -c 64 -H 112 -W 112 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 192 -H 56 -W 56 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 288 -H 56 -W 56 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 48 -H 56 -W 56 -k 288 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 32 -H 112 -W 112 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 80 -H 28 -W 28 -k 480 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 32 -H 112 -W 112 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 480 -H 14 -W 14 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 480 -H 28 -W 28 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 288 -H 28 -W 28 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 224 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 1344 -H 14 -W 14 -k 224 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 224 -H 14 -W 14 -k 1344 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 160 -H 14 -W 14 -k 960 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 1344 -H 7 -W 7 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 384 -H 7 -W 7 -k 2304 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 640 -H 7 -W 7 -k 2560 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 640 -H 7 -W 7 -k 3840 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 20 -H 1 -W 1 -k 480 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 1344 -H 1 -W 1 -k 56 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 40 -H 1 -W 1 -k 960 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 56 -H 1 -W 1 -k 1344 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 288 -H 1 -W 1 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 12 -H 1 -W 1 -k 288 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 8 -H 1 -W 1 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 160 -H 1 -W 1 -k 3840 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 640 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 960 -H 1 -W 1 -k 40 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 3840 -H 7 -W 7 -k 640 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 3840 -H 1 -W 1 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 192 -H 1 -W 1 -k 8 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 480 -H 1 -W 1 -k 20 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 16 -H 1 -W 1 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 64 -H 1 -W 1 -k 16 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 32 -H 1 -W 1 -k 8 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 2304 -H 1 -W 1 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 8 -H 1 -W 1 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 96 -H 1 -W 1 -k 2304 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
    echo "Test -g 1 -F 2 - completed"
fi

if [ "$ALL_TEST" = true ] || [ "$CONV3D_TEST" = true ]; then
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 1 -H 104 -W 152 -k 384 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 1 -H 104 -W 156 -k 384 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 1 -H 108 -W 148 -k 384 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 1 -H 112 -W 140 -k 384 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 1 -H 96 -W 164 -k 384 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 1 -H 96 -W 168 -k 384 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 2 -H 104 -W 152 -k 384 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 2 -H 104 -W 156 -k 384 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 2 -H 108 -W 148 -k 384 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 2 -H 112 -W 140 -k 384 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 2 -H 96 -W 164 -k 384 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 2 -H 96 -W 168 -k 384 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 3 -H 106 -W 154 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 3 -H 106 -W 158 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 3 -H 110 -W 150 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 3 -H 114 -W 142 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 3 -H 194 -W 330 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 3 -H 194 -W 338 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 3 -H 210 -W 306 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 3 -H 210 -W 314 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 3 -H 218 -W 298 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 3 -H 226 -W 282 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 3 -H 98 -W 166 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 3 -H 98 -W 170 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 4 -H 106 -W 154 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 4 -H 106 -W 158 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 4 -H 110 -W 150 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 4 -H 114 -W 142 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 4 -H 98 -W 166 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 4 -H 98 -W 170 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 5 -H 104 -W 152 -k 192 --fil_d 3 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 2 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 5 -H 104 -W 156 -k 192 --fil_d 3 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 2 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 5 -H 108 -W 148 -k 192 --fil_d 3 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 2 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 5 -H 112 -W 140 -k 192 --fil_d 3 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 2 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 5 -H 96 -W 164 -k 192 --fil_d 3 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 2 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 5 -H 96 -W 168 -k 192 --fil_d 3 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 2 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 6 -H 194 -W 330 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 6 -H 194 -W 338 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 6 -H 210 -W 306 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 6 -H 210 -W 314 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 6 -H 218 -W 298 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 192 --in_d 6 -H 226 -W 282 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 3 --in_d 3 -H 386 -W 658 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 3 --in_d 3 -H 386 -W 674 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 3 --in_d 3 -H 418 -W 610 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 3 --in_d 3 -H 418 -W 626 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 3 --in_d 3 -H 434 -W 594 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 3 --in_d 3 -H 450 -W 562 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 3 --in_d 6 -H 386 -W 658 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 3 --in_d 6 -H 386 -W 674 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 3 --in_d 6 -H 418 -W 610 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 3 --in_d 6 -H 418 -W 626 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 3 --in_d 6 -H 434 -W 594 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 3 --in_d 6 -H 450 -W 562 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 32 --in_d 21 -H 48 -W 82 -k 32 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 32 --in_d 21 -H 48 -W 84 -k 32 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 32 --in_d 21 -H 52 -W 76 -k 32 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 32 --in_d 21 -H 52 -W 78 -k 32 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 32 --in_d 21 -H 54 -W 74 -k 32 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 32 --in_d 21 -H 56 -W 70 -k 32 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 106 -W 154 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 106 -W 158 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 110 -W 150 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 114 -W 142 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 48 -W 82 -k 384 --fil_d 3 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 2 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 48 -W 84 -k 384 --fil_d 3 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 2 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 50 -W 84 -k 32 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 50 -W 84 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 50 -W 86 -k 32 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 50 -W 86 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 52 -W 76 -k 384 --fil_d 3 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 2 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 52 -W 78 -k 384 --fil_d 3 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 2 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 54 -W 74 -k 384 --fil_d 3 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 2 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 54 -W 78 -k 32 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 54 -W 78 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 54 -W 80 -k 32 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 54 -W 80 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 56 -W 70 -k 384 --fil_d 3 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 2 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 56 -W 76 -k 32 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 56 -W 76 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 58 -W 72 -k 32 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 58 -W 72 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 98 -W 166 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 3 -H 98 -W 170 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 4 -H 106 -W 154 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 4 -H 106 -W 158 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 4 -H 110 -W 150 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 4 -H 114 -W 142 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 4 -H 98 -W 166 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 384 --in_d 4 -H 98 -W 170 -k 384 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 1 -H 192 -W 328 -k 192 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 1 -H 192 -W 336 -k 192 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 1 -H 208 -W 304 -k 192 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 1 -H 208 -W 312 -k 192 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 1 -H 216 -W 296 -k 192 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 1 -H 224 -W 280 -k 192 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 3 -H 194 -W 330 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 3 -H 194 -W 338 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 3 -H 210 -W 306 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 3 -H 210 -W 314 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 3 -H 218 -W 298 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 3 -H 226 -W 282 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 3 -H 386 -W 658 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 3 -H 386 -W 674 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 3 -H 418 -W 610 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 3 -H 418 -W 626 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 3 -H 434 -W 594 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 3 -H 450 -W 562 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 4 -H 192 -W 328 -k 192 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 4 -H 192 -W 336 -k 192 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 4 -H 208 -W 304 -k 192 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 4 -H 208 -W 312 -k 192 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 4 -H 216 -W 296 -k 192 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 4 -H 224 -W 280 -k 192 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 6 -H 194 -W 330 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 6 -H 194 -W 338 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 6 -H 210 -W 306 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 6 -H 210 -W 314 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 6 -H 218 -W 298 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 6 -H 226 -W 282 -k 192 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 6 -H 386 -W 658 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 6 -H 386 -W 674 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 6 -H 418 -W 610 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 6 -H 418 -W 626 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 6 -H 434 -W 594 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convbfp16 -n 1 -c 96 --in_d 6 -H 450 -W 562 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
fi

if [ "$ALL_TEST" = true ]; then
    $MIOpenDriver convfp16 -n 128 -c 480 -H 28 -W 28 -k 480 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 480 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 960 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 960 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 2304 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 2304 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 1344 -H 14 -W 14 -k 1344 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1344 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 288 -H 56 -W 56 -k 288 -y 5 -x 5 -p 2 -q 2 -u 2 -v 2 -l 1 -j 1 -m conv -g 288 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 1344 -H 14 -W 14 -k 1344 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1344 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 960 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 960 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 480 -H 28 -W 28 -k 480 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 480 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 2304 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 2304 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 1344 -H 14 -W 14 -k 1344 -y 5 -x 5 -p 2 -q 2 -u 2 -v 2 -l 1 -j 1 -m conv -g 1344 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 3840 -H 7 -W 7 -k 3840 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 3840 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 2304 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 2304 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 2304 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 2304 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 960 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 960 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 288 -H 56 -W 56 -k 288 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 288 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 64 -H 112 -W 112 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 64 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 288 -H 56 -W 56 -k 288 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 288 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 64 -H 112 -W 112 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 64 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 1344 -H 14 -W 14 -k 1344 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1344 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 3840 -H 7 -W 7 -k 3840 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 3840 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 960 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 960 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 32 -H 112 -W 112 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 32 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 48 -H 56 -W 56 -k 288 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 32 -H 112 -W 112 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 32 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 480 -H 28 -W 28 -k 480 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 480 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 80 -H 28 -W 28 -k 480 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 2304 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 2304 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 960 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 960 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 960 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 960 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 288 -H 56 -W 56 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 32 -H 112 -W 112 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 1344 -H 14 -W 14 -k 1344 -y 5 -x 5 -p 2 -q 2 -u 2 -v 2 -l 1 -j 1 -m conv -g 1344 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 2304 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 2304 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 3840 -H 7 -W 7 -k 3840 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 3840 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 288 -H 56 -W 56 -k 288 -y 5 -x 5 -p 2 -q 2 -u 2 -v 2 -l 1 -j 1 -m conv -g 288 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 288 -H 56 -W 56 -k 288 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 288 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 64 -H 112 -W 112 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 64 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 480 -H 28 -W 28 -k 480 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 480 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 192 -H 112 -W 112 -k 192 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 192 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 192 -H 56 -W 56 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 32 -H 112 -W 112 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 288 -H 28 -W 28 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 32 -H 112 -W 112 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 32 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 480 -H 28 -W 28 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 1344 -H 14 -W 14 -k 1344 -y 5 -x 5 -p 2 -q 2 -u 2 -v 2 -l 1 -j 1 -m conv -g 1344 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 288 -H 56 -W 56 -k 288 -y 5 -x 5 -p 2 -q 2 -u 2 -v 2 -l 1 -j 1 -m conv -g 288 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 480 -H 28 -W 28 -k 480 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 480 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 480 -H 28 -W 28 -k 480 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 480 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 192 -H 112 -W 112 -k 192 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 192 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 64 -H 112 -W 112 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 192 -H 112 -W 112 -k 192 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 192 -F 2 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 160 -H 14 -W 14 -k 960 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 480 -H 14 -W 14 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 224 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 224 -H 14 -W 14 -k 1344 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 224 -H 14 -W 14 -k 1344 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 480 -H 1 -W 1 -k 20 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 288 -H 28 -W 28 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 160 -H 14 -W 14 -k 960 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 224 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 480 -H 28 -W 28 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 80 -H 28 -W 28 -k 480 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 288 -H 56 -W 56 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 1344 -H 14 -W 14 -k 224 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 384 -H 7 -W 7 -k 2304 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 480 -H 14 -W 14 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 32 -H 112 -W 112 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 64 -H 112 -W 112 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 3 -H 224 -W 224 -k 64 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 288 -H 1 -W 1 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 48 -H 56 -W 56 -k 288 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 192 -H 56 -W 56 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 640 -H 7 -W 7 -k 2560 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 640 -H 7 -W 7 -k 3840 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 640 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 384 -H 7 -W 7 -k 2304 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 3840 -H 7 -W 7 -k 640 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 1344 -H 14 -W 14 -k 224 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 1344 -H 7 -W 7 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 32 -H 112 -W 112 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 1344 -H 1 -W 1 -k 56 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 640 -H 7 -W 7 -k 3840 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 640 -H 7 -W 7 -k 2560 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 640 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 3840 -H 7 -W 7 -k 640 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 192 -H 1 -W 1 -k 8 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 1344 -H 7 -W 7 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 2304 -H 1 -W 1 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 8 -H 1 -W 1 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 3 -H 224 -W 224 -k 64 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 56 -H 1 -W 1 -k 1344 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 288 -H 1 -W 1 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 40 -H 1 -W 1 -k 960 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 40 -H 1 -W 1 -k 960 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 960 -H 1 -W 1 -k 40 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 160 -H 1 -W 1 -k 3840 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 192 -H 1 -W 1 -k 8 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 160 -H 1 -W 1 -k 3840 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 96 -H 1 -W 1 -k 2304 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 3840 -H 1 -W 1 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 8 -H 1 -W 1 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 20 -H 1 -W 1 -k 480 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 8 -H 1 -W 1 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 32 -H 1 -W 1 -k 8 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 20 -H 1 -W 1 -k 480 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 16 -H 1 -W 1 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 16 -H 1 -W 1 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 56 -H 1 -W 1 -k 1344 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 960 -H 1 -W 1 -k 40 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 3840 -H 1 -W 1 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 64 -H 1 -W 1 -k 16 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 12 -H 1 -W 1 -k 288 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 32 -H 1 -W 1 -k 8 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 8 -H 1 -W 1 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 2304 -H 1 -W 1 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 96 -H 1 -W 1 -k 2304 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 1344 -H 1 -W 1 -k 56 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 64 -H 1 -W 1 -k 16 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 480 -H 1 -W 1 -k 20 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
    $MIOpenDriver convfp16 -n 128 -c 12 -H 1 -W 1 -k 288 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
fi

# End time measurement
END_TIME=$(date +%s)

# Calculate the elapsed time
ELAPSED_TIME=$((END_TIME - START_TIME))

# Output the elapsed time in seconds
echo "Elapsed time: $ELAPSED_TIME seconds"
