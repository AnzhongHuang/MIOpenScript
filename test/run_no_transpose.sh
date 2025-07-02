#!/bin/bash

# Start time measurement
START_TIME=$(date +%s)

./MIOpenDriver convfp16 -n 128 -c 48 -H 56 -W 56 -k 288 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -V 1 -I NCHW
./MIOpenDriver convfp16 -n 128 -c 80 -H 28 -W 28 -k 480 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -V 1 -I NCHW
./MIOpenDriver convfp16 -n 128 -c 160 -H 14 -W 14 -k 960 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -V 1 -I NCHW
./MIOpenDriver convfp16 -n 128 -c 32 -H 112 -W 112 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -V 1 -I NCHW
./MIOpenDriver convfp16 -n 128 -c 224 -H 14 -W 14 -k 1344 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -V 1 -I NCHW
./MIOpenDriver convfp16 -n 128 -c 288 -H 28 -W 28 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -V 1 -I NCHW
./MIOpenDriver convfp16 -n 128 -c 480 -H 28 -W 28 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -V 1 -I NCHW
./MIOpenDriver convfp16 -n 128 -c 480 -H 14 -W 14 -k 160 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -V 1 -I NCHW
./MIOpenDriver convfp16 -n 128 -c 1344 -H 14 -W 14 -k 224 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -V 1 -I NCHW
./MIOpenDriver convfp16 -n 128 -c 384 -H 7 -W 7 -k 2304 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -V 1 -I NCHW
./MIOpenDriver convfp16 -n 128 -c 32 -H 112 -W 112 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -V 1 -I NCHW
./MIOpenDriver convfp16 -n 128 -c 288 -H 56 -W 56 -k 48 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -V 1 -I NCHW
./MIOpenDriver convfp16 -n 128 -c 640 -H 7 -W 7 -k 2560 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -V 1 -I NCHW
./MIOpenDriver convfp16 -n 128 -c 640 -H 7 -W 7 -k 3840 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -V 1 -I NCHW
./MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 224 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -V 1 -I NCHW
./MIOpenDriver convfp16 -n 128 -c 64 -H 112 -W 112 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -V 1 -I NCHW

# End time measurement
END_TIME=$(date +%s)

# Calculate the elapsed time
ELAPSED_TIME=$((END_TIME - START_TIME))

# Output the elapsed time in seconds
echo "Elapsed time: $ELAPSED_TIME seconds"
