#!/bin/bash
set -e -x

if [ "${TRAVIS_OS_NAME}" == osx ]; then
  brew update > /dev/null
  brew outdated pkg-config || brew upgrade pkg-config
  brew outdated pyenv || brew upgrade pyenv
  brew install jags
  eval "$(pyenv init -)"
  hash -r
  pyenv install --list
  pyenv install -s "${PYENV_VERSION}"
  pip install numpy
fi
