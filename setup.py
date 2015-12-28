# encoding: utf-8

from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc
from numpy.distutils.misc_util import get_numpy_include_dirs
import subprocess


def pkg_config(*args):
    cmd = ['pkg-config']
    cmd.extend(args)
    return subprocess.check_output(cmd).decode().split()


setup(name='pyjags',
      version='0.1',
      description='Python interface to JAGS',
      author=u'Tomasz Miąsko',
      # TODO url
      license='GPL',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)'
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
      ],
      packages=['pyjags'],
      ext_modules=[
          Extension(
              'pyjags.console',
              include_dirs=['pybind11/include/'] + get_numpy_include_dirs(),
              libraries=['jags'],
              define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
              extra_compile_args=['-std=c++14'] + pkg_config('jags', '--cflags'),
              extra_link_args=pkg_config('jags', '--libs'),
              sources=['pyjags/console.cc'])
      ],
      requires=['numpy'])
