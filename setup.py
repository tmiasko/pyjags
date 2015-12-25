#!usr/bin/env python
from distutils.core import setup, Extension

setup(
    name='pyjags', 
    version='0.1',
    description='Python interface to JAGS', 
    author='Tomasz Miąsko',
    license='BSD',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)'
    ],
    ext_modules=[
        Extension(
            'pyjags.console', 
            include_dirs=['/usr/include/JAGS/', 'pybind11/include/'],
            libraries=['jags'],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
            extra_compile_args=['-std=c++14'],
            sources=['console.cc'])
    ])
