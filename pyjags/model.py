# Copyright (C) 2015 Tomasz Miasko
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

__all__ = ['Model', 'list_modules', 'load_module', 'unload_module']

from .console import *

import collections
import contextlib
import ctypes
import locale
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
    """Load module."""
    if name not in JAGS_MODULES:
        path = os.path.join(JAGS_MODULE_DIR, name + JAGS_MODULE_EXT)
        lib = ctypes.cdll.LoadLibrary(path)
        JAGS_MODULES[name] = lib
    Console.loadModule(name)


def unload_module(name):
    """Unload module."""
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


@contextlib.contextmanager
def _get_model_path(name, text):
    if name:
        yield name
    elif text:
        if isinstance(text, str):
            text = text.encode()
        with tempfile.NamedTemporaryFile() as fh:
            fh.write(text)
            fh.flush()
            yield fh.name
    else:
        raise ValueError('Either model name or model text must be provided.')


class Model:

    def __init__(self, name=None, text=None, data=None, start=None, chains=1, tune=1000):
        """
        Create a JAGS model and run adaptation steps.

        Parameters
        ----------
        name : string, optional
            File with a model to load.
        text : string, optional
            Content of model to load.
        start : dict or list of dicts, optional
            Dictionary with initial parameter values. To use different starting
            values for each chain provide a list of dictionaries.
            Random number generator can be configured using special '.RNG.name'
            key. And its initial state using '.RNG.state'.
        data : dict, optional
            Dictionary with observed nodes in the model. The numpy.ma.MaskedArray
            can be used to provided data when some of observation are missing.
        chains : int, optional
            Number of parallel chains.
        tune : int, optional
            Number of adaptations steps.
        """

        chains = int(chains)
        tune = int(tune)

        self.console = Console()
        with _get_model_path(name, text) as path:
            self.console.checkModel(path)

        data = {} if data is None else _to_numpy_dictionary(data)
        unused = set(data.keys()) - set(self.variables)
        if unused:
            raise ValueError('Unused data for variables: {}'.format(','.join(unused)))
        self.console.compile(data, chains, True)

        start = {} if start is None else start
        if isinstance(start, collections.Mapping):
            start = [start] * chains
        elif not isinstance(start, collections.Sequence):
            raise ValueError('Start should be a sequence or a dictionary.')
        if len(start) != self.num_chains:
            raise ValueError('Length of start sequence should equal the number of chains.')
        for data, chain in zip(start, self.chains):
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
        Parameters
        ----------
        iterations : int
            Number of iterations.
        vars : list of variables, optional
            List of variables to monitor.
        thin : int, optional
            Thinning interval, i.e., every thin iteration will be recorded.
        Returns
        -------
        """
        if vars is None:
            vars = self.variables
        try:
            for name in vars:
                self.console.setMonitor(name, thin, monitor_type)
            self.update(iterations)
            samples = self.console.dumpMonitors(monitor_type, False)
        finally:
            for name in vars:
                self.console.clearMonitor(name, monitor_type)
        return samples

    def adapt(self, iterations):
        """Run adaptation steps to maximize samplers efficiency.

        Returns
        -------
        adapt : bool
            True if achieved performance is close to the theoretical optimum.
        """
        if not self.console.isAdapting():
            # Model does not require adaptation
            return True
        self.console.update(iterations)
        return self.console.checkAdaptation()

    @property
    def variables(self):
        """List of variables in the model."""
        return self.console.variableNames()

    @property
    def num_chains(self):
        """Number of chains in the model."""
        return self.console.nchain()

    @property
    def chains(self):
        """List of chains."""
        return list(range(1, self.console.nchain()+1))

    @property
    def state(self):
        return [ self.console.dumpState(DUMP_ALL, chain) for chain in self.chains]


