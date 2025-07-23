# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from pybind11.setup_helpers import Pybind11Extension

class BuildExtension(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')
        original_compile = self.compiler._compile

        def custom_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith('.cu'):
                nvcc = os.environ.get('CUDAHOME', '/usr/local/cuda/bin') + '/nvcc'
                if not os.path.exists(nvcc):
                    nvcc = 'nvcc'  
                filtered_cc_args = [arg for arg in cc_args if arg not in ['-fvisibility=hidden', '-c', '-fPIC']]
                include_args = [arg for arg in filtered_cc_args if arg.startswith('-I')]
                compiler_args = [arg for arg in filtered_cc_args if not arg.startswith('-I')]
                cmd = [
                    nvcc, '-c', src, '-o', obj,
                    '-Xcompiler', ','.join(compiler_args + ['-fPIC']),
                    '-O3', '--use_fast_math'
                ] + include_args
                print("Compiling CUDA:", " ".join(cmd))
                subprocess.check_call(cmd)
            else:
                original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = custom_compile
        build_ext.build_extensions(self)

ext_modules = [
    Pybind11Extension(
        "mobility_gen_path_planner._mobility_gen_path_planner_C",
        ["src/python_bindings.cpp"],
        include_dirs=["src"]
    ),
    Pybind11Extension(
        "pure_pursuit_cuda._path_helper_gpu_op",
        #["src/pure_pursuit_cuda.cpp", "src/pure_pursuit_cuda.cu"],
        ["src/pure_pursuit_cuda.cu"],
        include_dirs=["src"],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="mobility_gen_op",
    version="0.0.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

