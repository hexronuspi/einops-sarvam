from setuptools import setup, Extension
import numpy as np
import pybind11

eigen_backend = Extension(
    'rearrange.eigen_backend', 
    sources=['eigen_backend.cpp'],
    include_dirs=[
        np.get_include(),
        '/usr/include/eigen3',
        pybind11.get_include(),
    ],
    extra_compile_args=['-std=c++17', '-ftemplate-depth=2048'],
    language='c++'
)

setup(
    name='rearrange',
    version='1.0',
    packages=['rearrange'],  
    package_dir={'rearrange': '.'}, 
    ext_modules=[eigen_backend],
    install_requires=['numpy', 'pybind11', 'numba']
)
