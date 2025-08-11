#!/bin/bash

echo "=== Installing MIOpenDriver_Ref Module ==="

cd /mnt/workspace/ytn/docker_path/MIOpenScript/MIOpenDriver_Ref

python3 setup.py build_ext --inplace

rm ../*.so

cp ./*.so ../