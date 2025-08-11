#!/bin/bash

echo "=== Installing MIOpenDriver_Ref Module ==="

cd /mnt/workspace/ytn/docker_path/MIOpenScript/MIOpenDriver_Ref

# 安装模块到Python环境
echo "Installing module to Python environment..."
pip3 install -e .

if [ $? -eq 0 ]; then
    echo "✓ Installation successful"
    
    # 验证安装
    echo "Verifying installation..."
    python3 -c "
import MIOpenDriver_Ref
print('✓ Module installed and imported successfully')
print(f'Module location: {MIOpenDriver_Ref.__file__}')
print(f'MIOpen version: {MIOpenDriver_Ref.get_miopen_version()}')
"
else
    echo "✗ Installation failed"
    exit 1
fi