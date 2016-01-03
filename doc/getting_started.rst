Getting Started
===============

Prerequisites
-------------

* JAGS
* C++11 compiler
* Python 2.7, >= 3.2
* NumPy >= 1.7

Installation
------------

Using pip::

  pip install git+https://github.com/tmiasko/pyjags.git

Using setup.py after cloning the repository::

  git clone https://github.com/tmiasko/pyjags.git
  git submodule --update --init
  python setup.py install

The setup.py script uses pkg-config to locate the JAGS library. If JAGS is
installed in some non-standard location, then you may need to configure
pkg-config to pickup correct metadata file. For example, if you JAGS have been
configured with ``--prefix=/opt/``, then before running setup.py, following
environment variable should be exported::

  export PKG_CONFIG_PATH=/opt/lib/pkgconfig/:$PKG_CONFIG_PATH

In this case, before you can import pyjags, the library path should be also
include JAGS library directory::

  export LD_LIBRARY_PATH=/opt/lib:$LD_LIBRARY_PATH

Example
-------

Simple linear regression:

.. include:: example.py
  :code:

.. include:: example.out
  :code: text

.. TODO add plot
