from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_cuda',
    ext_modules=[
        CUDAExtension('custom_cuda', [
            'custom_cuda.cpp',
            'preprocessing_cuda_kernel.cu',
            'nms_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })