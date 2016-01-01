# Copyright (C) 2015-2016 Tomasz Miasko
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

__all__ = ['list_modules', 'load_module', 'unload_module', 'Model']

from .console import *

import collections
import contextlib
import ctypes
import os
import os.path
import tempfile

import numpy as np

# TODO determine package dir using pkg-config
JAGS_MODULE_DIR = '/usr/lib/JAGS/modules-3'
JAGS_MODULE_EXT = '.so' if os.name == 'posix' else '.dll'
JAGS_MODULES = {}
# Special value indicating missing data in JAGS.
JAGS_NA = Console.na()


def load_module(name):
    """Load a module.

    During initialization PyJAGS loads basemod module and bugs module.
    """
    if name not in JAGS_MODULES:
        path = os.path.join(JAGS_MODULE_DIR, name + JAGS_MODULE_EXT)
        lib = ctypes.cdll.LoadLibrary(path)
        JAGS_MODULES[name] = lib
    Console.loadModule(name)


def unload_module(name):
    """Unload a module."""
    return Console.unloadModule(name)


def list_modules():
    """Return a list of loaded modules."""
    return Console.listModules()


# Default modules
load_module('basemod')
load_module('bugs')


def _to_numpy_dictionary(src):
    """Return a dictionary where values are converted into numpy arrays suitable
    for use with JAGS.

     * Returned arrays have at least one dimension.
     * Masked values are replaced by JAGS_NA.
     * Empty arrays are removed from the dictionary.
    """
    dst = {}
    for k, v in src.items():
        if np.ma.is_masked(v):
            v = np.ma.array(data=v, dtype=np.double, ndmin=1, fill_value=JAGS_NA)
            v = np.ma.filled(v)
        else:
            v = np.atleast_1d(v)
        if not np.size(v):
            continue
        dst[k] = v
    return dst


class Model:
    """High level representation of JAGS model.

    Note
    ----
    In JAGS arrays are indexed from 1. On the other hand Python uses 0 based
    indexing. It is important to keep this in mind when providing data to JAGS
    and interpreting resulting samples. For example, what in JAGS would be
    ``x[4,2,7]`` in Python is ``x[3,1,6]``.

    Note
    ----
    The JAGS supports data sets where some of observations have no value.
    In PyJAGS those missing values are described using numpy MaskedArray.
    For example, to create a model with observations ``x[1] = 0.25``,
    ``x[3] = 0.75``, and observation ``x[2]`` missing, we would provide
    following data to Model constructor:

    >>> {'x': np.ma.masked_array(data=[0.25, 0, 0.75], mask=[False, True, False])}
    {'x': masked_array(data = [0.25 -- 0.75],
                 mask = [False  True False],
           fill_value = 1e+20)
    }

    From JAGS version 4.0.0 it is also possible to monitor variables that are
    not completely defined in the description of the model, e.g., if y[i] is
    defined only for y[3], then y[1], and y[2] will have missing values for
    all iterations in all chains. Those missing values are also represented
    using numpy MaskedArray.
    """

    def __init__(self, code=None, data=None, init=None, chains=4, tune=1000, file=None, encoding='utf-8'):
        """
        Create a JAGS model and run adaptation steps.

        Parameters
        ----------
        code : string, optional
            Code of the model to load. Model may be also provided with file
            keyword argument.
        file : string, optional
            Path to the model to load. Model may be also provided with code
            keyword argument.
        init : dict or list of dicts, optional
            Specifies initial values for parameters. It can be either a
            dictionary providing initial values for parameters used as keys,
            or a list of dictionaries providing initial values separately for
            each chain. If omitted, initial values will be generated
            automatically.

            Additionally this option allows to configure random number
            generators using following special keys:

             * '.RNG.name'  string, name of random number generator
             * '.RNG.seed'  int, seed for random number generator
             * '.RNG.state' array, may be specified instead of seed, shape of
               array depends on particular generator used
        data : dict, optional
            Dictionary with observed nodes in the model. Keys are variable
            names and values should be convertible to numpy arrays with shape
            compatible with one used in the model.

            The numpy.ma.MaskedArray can be used to provide data where some of
            observations are missing.
        chains : int, 4 by default
            A positive number specifying number of parallel chains.
        tune : int, 1000 by default
            An integer specifying number of adaptations steps.
        encoding : string, 'utf-8' by default
            When model code is provided as a string, this specifies its encoding.
        """

        # Detect potential type errors.
        chains = int(chains)
        tune = int(tune)

        @contextlib.contextmanager
        def model_path(file=None, code=None):
            """Utility function returning model path, if necessary creates a
            new temporary file with a model code written into it.
            """
            if file:
                yield file
            elif code:
                if isinstance(code, str):
                    code = code.encode(encoding=encoding)
                # TODO use separate delete to support Windows?
                with tempfile.NamedTemporaryFile() as fh:
                    fh.write(code)
                    fh.flush()
                    yield fh.name
            else:
                raise ValueError('Either model name or model text must be provided.')

        self.console = Console()
        with model_path(file, code) as path:
            self.console.checkModel(path)

        data = {} if data is None else _to_numpy_dictionary(data)
        unused = set(data.keys()) - set(self.variables)
        if unused:
            raise ValueError('Unused data for variables: {}'.format(','.join(unused)))
        self.console.compile(data, chains, True)

        init = {} if init is None else init
        if isinstance(init, collections.Mapping):
            init = [init] * chains
        elif not isinstance(init, collections.Sequence):
            raise ValueError('Init should be a sequence or a dictionary.')
        if len(init) != self.num_chains:
            raise ValueError('Length of init sequence should equal the number of chains.')
        for data, chain in zip(init, self.chains):
            data = dict(data)
            rng_name = data.pop('.RNG.name', None)
            if rng_name is not None:
                self.console.setRNGname(rng_name, chain)
            data = _to_numpy_dictionary(data)
            unused = set(data.keys()) - set(self.variables) - {'.RNG.seed', '.RNG.state'}
            if unused:
                raise ValueError('Unused initial values in chain {} for variables: {}'.format(chain, ','.join(unused)))
            self.console.setParameters(data, chain)

        self.console.initialize()
        if tune:
            self.adapt(tune)

    def update(self, iterations):
        """Updates the model for given number of iterations."""
        # TODO progress bar?
        self.console.update(iterations)

    def sample(self, iterations, vars=None, thin=1, monitor_type="trace"):
        """
        Creates monitors for given variables, runs the model for provided
        number of iterations and returns monitored samples.


        Parameters
        ----------
        iterations : int
            A positive integer specifying number of iterations.
        vars : list of variables, optional
            A list of variables to monitor.
        thin : int, optional
            A positive integer specifying thinning interval.
        Returns
        -------
        dict
            Sampled values of monitored variables as a dictionary where keys
            are variable names and values are numpy arrays with shape:
            (dim_1, dim_n, chains, iterations).

            dim_1, ..., dim_n describe the shape of variable in JAGS model.
        """
        if vars is None:
            vars = self.variables
        try:
            for name in vars:
                self.console.setMonitor(name, thin, monitor_type)
            self.update(iterations)
            samples = self.console.dumpMonitors(monitor_type, False)
            # TODO support NA values that could be returned since JAGS 4.0.0
        finally:
            for name in vars:
                self.console.clearMonitor(name, monitor_type)
        return samples

    def adapt(self, iterations):
        """Run adaptation steps to maximize samplers efficiency.

        Returns
        -------
        bool
            True if achieved performance is close to the theoretical optimum.
        """
        if not self.console.isAdapting():
            # Model does not require adaptation
            return True
        self.console.update(iterations)
        return self.console.checkAdaptation()

    @property
    def variables(self):
        """A list of variables in the model."""
        return self.console.variableNames()

    @property
    def num_chains(self):
        """A number of chains in the model."""
        return self.console.nchain()

    @property
    def chains(self):
        """A list of chain identifiers in the model."""
        return list(range(1, self.console.nchain()+1))

    @property
    def state(self):
        """Internal state of the model."""
        return [ self.console.dumpState(DUMP_ALL, chain) for chain in self.chains]


