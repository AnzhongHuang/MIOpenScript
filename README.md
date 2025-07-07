# MIOpenScript

## Run with MIOpenDriver
    ./conv_test.sh

## Run via pytorch
    ./conv_test.sh --run py

## Run -g=1 bwd tests
    ./conv_test.sh --g1f2 1

## Parse pytorch trace
    python perf_parse.py --trace trace.json --output trace.log
