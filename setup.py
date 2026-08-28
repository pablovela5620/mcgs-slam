from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='droid_backends',
    ext_modules=[
        CUDAExtension('droid_backends',
            include_dirs=[osp.join(ROOT, 'thirdparty/eigen')],
            sources=[
                'src/droid.cpp', 
                'src/droid_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
            ],
            # Target archs come from TORCH_CUDA_ARCH_LIST (set in pixi.toml);
            # CUDA 13 dropped the old hardcoded sm_60..sm_86 gencodes.
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)

setup(
    name='lietorch',
    version='0.2',
    description='Lie Groups for PyTorch',
    packages=['lietorch'],
    package_dir={'': 'thirdparty/lietorch'},
    ext_modules=[
        CUDAExtension('lietorch_backends', 
            include_dirs=[
                osp.join(ROOT, 'thirdparty/lietorch/lietorch/include'), 
                osp.join(ROOT, 'thirdparty/eigen')],
            sources=[
                'thirdparty/lietorch/lietorch/src/lietorch.cpp', 
                'thirdparty/lietorch/lietorch/src/lietorch_gpu.cu',
                'thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2']
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)
