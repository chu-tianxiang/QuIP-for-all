from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='quiptools_cuda',
      ext_modules=[cpp_extension.CUDAExtension(
          'quiptools_cuda',
          ['quiptools_wrapper.cpp', 'e8p_gemv.cu', 'origin_order.cu'],
          extra_compile_args={'cxx': ['-g', '-lineinfo'],
                              'nvcc': ['-O3', '-g', '-Xcompiler', '-rdynamic', '-lineinfo']})],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

