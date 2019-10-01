from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='carafe_layer_cuda',
    ext_modules=[
        CUDAExtension('carafe_layer_cuda', [
            'src/carafe_layer_cuda.cpp',
            'src/carafe_layer_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
