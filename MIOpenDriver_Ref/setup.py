from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension
import os

# Define the extension module
ext_modules = [
    CppExtension(
        name='MIOpenDriver_Ref',
        sources=[
            '/mnt/workspace/ytn/docker_path/MIOpenScript/MIOpenDriver_Ref/MIOpenDriver_Ref.cpp',
        ],
        include_dirs=[
            # PyTorch includes
            *torch.utils.cpp_extension.include_paths(),
            # Pybind11 includes
            pybind11.get_include(),
            # MIOpen includes (adjust path as needed)
            '/opt/rocm/include',
            '/usr/local/include',
        ],
        libraries=['MIOpen'],
        library_dirs=[
            '/opt/rocm/lib',
            '/usr/local/lib',
        ],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3'],
        extra_link_args=['-Wl,-rpath,/opt/rocm/lib'],
    ),
]

setup(
    name='MIOpenDriver_Ref',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)