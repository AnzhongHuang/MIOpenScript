# MIOpenScript

## Run with MIOpenDriver
    ./conv_test.sh

## Run via pytorch
    ./conv_test.sh --trace trace.json --event event.log

## Run -g=1 bwd tests
    ./conv_test.sh --g1f2 1

## Parse pytorch trace
    python perf_parse.py --trace trace.json --output trace.log

## Diff MIOpen logs
    python miopenlog_diff.py --log1 xxx --log2 xxx
