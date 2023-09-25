from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='quip_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        'quip_cuda', ['q_gemm.cpp', 'q_gemm_kernel.cu']
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
