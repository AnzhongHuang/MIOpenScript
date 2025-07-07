#!/bin/bash

MIOpenDriver="MIOpenDriver"
TRACE=""
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)
            if [[ "$2" == "py" ]]; then
                MIOpenDriver="python mi_conv_torch.py"
                TRACE="--trace trace.json"
            fi
            shift 2
            ;;
        *)
            # Forward other arguments to the driver commands
            break
            ;;
    esac
done

#export MIOPEN_ENABLE_LOGGING=1
#export MIOPEN_ENABLE_LOGGING_CMD=1
#export MIOPEN_LOG_LEVEL=100
export  HIP_VISIBLE_DEVICES=6
export MIOPEN_DRIVER_USE_GPU_REFERENCE=0
#LAYOUT="--in_layout NHWC --out_layout NHWC --fil_layout NHWC"
LAYOUT=""
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
$MIOpenDriver convfp16 -n 128 -c 64 -H 112 -W 112 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 2304 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 2304 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 960 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 960 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 960 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 960 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 288 -H 56 -W 56 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 192 -H 56 -W 56 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 288 -H 56 -W 56 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 32 -H 112 -W 112 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 48 -H 56 -W 56 -k 288 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 1344 -H 14 -W 14 -k 1344 -y 5 -x 5 -p 2 -q 2 -u 2 -v 2 -l 1 -j 1 -m conv -g 1344 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 2304 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 2304 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 3840 -H 7 -W 7 -k 3840 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 3840 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 288 -H 56 -W 56 -k 288 -y 5 -x 5 -p 2 -q 2 -u 2 -v 2 -l 1 -j 1 -m conv -g 288 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 288 -H 56 -W 56 -k 288 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 288 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 64 -H 112 -W 112 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 64 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 480 -H 28 -W 28 -k 480 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 480 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 192 -H 112 -W 112 -k 192 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 192 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 32 -H 112 -W 112 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 192 -H 56 -W 56 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 80 -H 28 -W 28 -k 480 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 32 -H 112 -W 112 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 288 -H 28 -W 28 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 32 -H 112 -W 112 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 32 -H 112 -W 112 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 32 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 480 -H 28 -W 28 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 1344 -H 14 -W 14 -k 1344 -y 5 -x 5 -p 2 -q 2 -u 2 -v 2 -l 1 -j 1 -m conv -g 1344 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 288 -H 56 -W 56 -k 288 -y 5 -x 5 -p 2 -q 2 -u 2 -v 2 -l 1 -j 1 -m conv -g 288 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 480 -H 14 -W 14 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 480 -H 28 -W 28 -k 480 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 480 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 480 -H 28 -W 28 -k 480 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 480 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 192 -H 112 -W 112 -k 192 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 192 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 480 -H 28 -W 28 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 64 -H 112 -W 112 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 192 -H 112 -W 112 -k 192 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 192 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 160 -H 14 -W 14 -k 960 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 480 -H 14 -W 14 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 288 -H 28 -W 28 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 224 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 224 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 1344 -H 14 -W 14 -k 224 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 224 -H 14 -W 14 -k 1344 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 224 -H 14 -W 14 -k 1344 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 224 -H 14 -W 14 -k 1344 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
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
$MIOpenDriver convfp16 -n 128 -c 160 -H 14 -W 14 -k 960 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 640 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 384 -H 7 -W 7 -k 2304 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 3840 -H 7 -W 7 -k 640 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 1344 -H 14 -W 14 -k 224 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 1344 -H 7 -W 7 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 384 -H 7 -W 7 -k 2304 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 640 -H 7 -W 7 -k 2560 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 1344 -H 7 -W 7 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 32 -H 112 -W 112 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 1344 -H 1 -W 1 -k 56 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 640 -H 7 -W 7 -k 3840 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 640 -H 7 -W 7 -k 3840 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 640 -H 7 -W 7 -k 2560 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 640 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 20 -H 1 -W 1 -k 480 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 3840 -H 7 -W 7 -k 640 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 1344 -H 1 -W 1 -k 56 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 192 -H 1 -W 1 -k 8 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 1344 -H 7 -W 7 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 2304 -H 1 -W 1 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 8 -H 1 -W 1 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 3 -H 224 -W 224 -k 64 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 40 -H 1 -W 1 -k 960 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 56 -H 1 -W 1 -k 1344 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 56 -H 1 -W 1 -k 1344 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 288 -H 1 -W 1 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 12 -H 1 -W 1 -k 288 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 8 -H 1 -W 1 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 288 -H 1 -W 1 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 40 -H 1 -W 1 -k 960 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 40 -H 1 -W 1 -k 960 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 960 -H 1 -W 1 -k 40 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 160 -H 1 -W 1 -k 3840 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 160 -H 1 -W 1 -k 3840 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 192 -H 1 -W 1 -k 8 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 160 -H 1 -W 1 -k 3840 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 96 -H 1 -W 1 -k 2304 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 3840 -H 1 -W 1 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 8 -H 1 -W 1 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 640 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 20 -H 1 -W 1 -k 480 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 960 -H 1 -W 1 -k 40 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 3840 -H 7 -W 7 -k 640 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 3840 -H 1 -W 1 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 192 -H 1 -W 1 -k 8 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 8 -H 1 -W 1 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 32 -H 1 -W 1 -k 8 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 480 -H 1 -W 1 -k 20 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 16 -H 1 -W 1 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 64 -H 1 -W 1 -k 16 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 32 -H 1 -W 1 -k 8 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 2304 -H 1 -W 1 -k 96 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 8 -H 1 -W 1 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
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
$MIOpenDriver convfp16 -n 128 -c 96 -H 1 -W 1 -k 2304 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 2 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 96 -H 1 -W 1 -k 2304 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 1344 -H 1 -W 1 -k 56 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 64 -H 1 -W 1 -k 16 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 480 -H 1 -W 1 -k 20 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 ${TRACE} ${LAYOUT}
$MIOpenDriver convfp16 -n 128 -c 12 -H 1 -W 1 -k 288 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 ${TRACE} ${LAYOUT}
