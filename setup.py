# encoding: utf-8
#
# Copyright (C) 2015-2016 Tomasz Miasko
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import sys

from setuptools import dist, setup, Extension
import subprocess
import versioneer


def content(path):
    with open(path) as fh:
        return fh.read()


def add_pkg_config(ext, package):
    flags_map = {
        '-I': ['include_dirs'],
        '-L': ['library_dirs', 'runtime_library_dirs'],
        '-l': ['libraries'],
    }

    try:
        args = ['pkg-config', '--libs', '--cflags', package]
        output = subprocess.check_output(args)
        output = output.decode()
        for flag in output.split():
            for attr in flags_map[flag[:2]]:
                getattr(ext, attr).append(flag[2:])

        args = ['pkg-config', '--modversion', package]
        output = subprocess.check_output(args)
        return output.strip()
    except Exception as err:
        print("Error while executing pkg-config: {}".format(err))
        sys.exit(1)


def add_jags(ext):
    version = add_pkg_config(ext, 'jags')
    version = '"{}"'.format(version)
    ext.define_macros.append(('PYJAGS_JAGS_VERSION', version))


def add_numpy(ext):
    try:
        import numpy
    except ImportError:
        sys.exit('Please install numpy first.')
    ext.include_dirs.append(numpy.get_include())


def add_pybind11(ext):
    ext.include_dirs.append('pybind11/include')
    ext.extra_compile_args.append('-std=c++11')


if __name__ == '__main__':
    ext = Extension('pyjags.console',
                    language='c++',
                    sources=['pyjags/console.cc'])
    add_jags(ext)
    add_numpy(ext)
    add_pybind11(ext)

    setup(name='pyjags',
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          description='Python interface to JAGS library for Bayesian data analysis.',
          long_description=content('README.rst'),
          author=u'Tomasz Miąsko',
          author_email='tomasz.miasko@gmail.com',
          url='https://github.com/tmiasko/pyjags',
          license='GPLv2',
          classifiers=[
              'Development Status :: 4 - Beta',
              'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
              'Operating System :: POSIX',
              'Programming Language :: C++',
              'Programming Language :: Python :: 2',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.4',
              'Programming Language :: Python :: 3.5',
              'Programming Language :: Python',
              'Topic :: Scientific/Engineering :: Information Analysis',
              'Topic :: Scientific/Engineering',
          ],
          packages=['pyjags'],
          ext_modules=[ext],
          install_requires=['numpy'],
          test_suite='test')
