#!/bin/bash

echo "=== Installing MIOpenDriver_Ref Module ==="

cd /mnt/workspace/ytn/docker_path/MIOpenScript/MIOpenDriver_Ref

python3 setup.py build_ext --inplace

rm ../MIOpenDriver_Ref.*.so

mv ./MIOpenDriver_Ref.*.so ../