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

Stable release from pip (recommended)::

  pip install pyjags

Development version::

  pip install git+https://github.com/tmiasko/pyjags.git

Using setup.py after cloning the repository::

  git clone --recursive https://github.com/tmiasko/pyjags.git
  python setup.py install

The setup.py script uses pkg-config to locate the JAGS library. If JAGS is
installed in some non-standard location, then you may need to configure
pkg-config to pickup correct metadata file. For example, if JAGS have been
configured with ``--prefix=/opt/``, then JAGS metadata file would be located in
``/opt/lib/pkgconfig/``.  This path can be included in pkg-config search path
as follows::

  export PKG_CONFIG_PATH=/opt/lib/pkgconfig/:$PKG_CONFIG_PATH

Example
-------

Simple linear regression:

.. include:: example.py
  :code:

.. include:: example.out
  :code: text

.. TODO add plot
