#!/bin/bash
set -e -x

if [ "${TRAVIS_OS_NAME}" == osx ]; then
  eval "$(pyenv init -)"
  export PKG_CONFIG_PATH=$(brew --prefix jags)/lib/pkgconfig:$PKG_CONFIG_PATH
fi

env
which python

python setup.py build_ext -i
python setup.py test
