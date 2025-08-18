#!/bin/bash

echo "=== Installing MIOpenDriver_Ref Module ==="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd "$SCRIPT_DIR" > /dev/null

python3 setup.py build_ext --inplace

rm -f ../MIOpenDriver_Ref.*.so

mv ./MIOpenDriver_Ref.*.so ../

popd > /dev/null
