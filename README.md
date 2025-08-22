# MIOpenScript

## Run with MIOpenDriver
    python MIOpenDriver.py --test_list conv_30.sh --verify 0 --usage 0

## Run with pytorch trace
    python MIOpenDriver.py --test_list conv_30.sh --verify 0 --usage 0 --trace trace.json

## Run with GPU usage:
    python MIOpenDriver.py --test_list conv_30.sh --verify 0 --usage 1

## Use global pool memory
    python MIOpenDriver.py --test_list conv_30.sh --verify 0 --usage 0 --pool 1

## Diff MIOpen logs
    python miopenlog_diff.py --log1 xxx --log2 xxx

## Export testlist from db
    python db_shape_2_args.py --solver_file /opt/rocm/share/miopen/db/gfx942130.HIP.fdb.txt --output gfx942130.txt

## Verify testlist
    python args_2_shape.py --test gfx942130.txt  # passlist generated: test_passlist.txt

## Convert a ufdb case to MIOpenDriver test
    python db_shape_2_args.py --solver 288-8-48-32-3x1x1-288-8-48-32-37-1x0x0-1x1x1-1x1x1-0-NCDHW-BF16-F_g3

## Convert a udb case to MIOpenDriver test
    python db_shape_2_args.py --kernel 2x1344x14x14x1x1x1x1x224x128x0x0x0x1x1x0x1x1x0x0x1xNCHWxFP16xW

## Convert a test to a shape
    python args_2_shape.py convbf16 -F 1 -n 4 -c 96 -H 48 -W 32 -k 96 -y 3 -x 1 --pad_h 2 --pad_w 0 -u 1 -v 1 -l 2 -j 2 -g 1 -m conv --spatial_dim 2 -t 1 --shapeformat solver # ufdb shape
    python args_2_shape.py convbf16 -F 1 -n 4 -c 96 -H 48 -W 32 -k 96 -y 3 -x 1 --pad_h 2 --pad_w 0 -u 1 -v 1 -l 2 -j 2 -g 1 -m conv --spatial_dim 2 -t 1 --shapeformat kernel # udb shape
    python args_2_shape.py convbf16 -F 1 -n 4 -c 96 -H 48 -W 32 -k 96 -y 3 -x 1 --pad_h 2 --pad_w 0 -u 1 -v 1 -l 2 -j 2 -g 1 -m conv --spatial_dim 2 -t 1 --shapeformat vs # vscode debug args