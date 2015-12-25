#!/usr/bin/env python3

import ctypes
import pyjags
import pyjags.console
import numpy as np
import pprint
import tempfile
import contextlib

_loaded_modules = {}
#Special float value indicating missing data in JAGS.
JAGS_NA = pyjags.console.Console.na()

def load_module(name):
    if not name in _loaded_modules:
        path = '/usr/lib/JAGS/modules-3/{}.so'.format(name)
        module = ctypes.cdll.LoadLibrary(path)
        _loaded_modules[name] = module
    pyjags.console.Console.loadModule(name)

def unload_module(name):
    pyjags.console.Console.unloadModule(name)

load_module('basemod')
load_module('bugs')

def _to_numpy_array(src):
    """
    """
    d = {}
    for k, v in src.items():
        # Avoid 0-dimensional arrays. They would require special handling in C code.
        v = np.array(v, ndmin=1)
        # JAGS SArray does not support empty arrays.
        if not np.size(v):
            continue
        # Convert missing data to JAGS format.
        v = np.ma.filled(v, JAGS_NA)
        d[k] = v
    return d

@contextlib.contextmanager
def _get_model_path(name, text):
    if name:
        yield name
    elif text:
        with tempfile.NamedTemporaryFile() as fh:
            fh.write(text)
            fh.flush()
            yield fh.name

class Model:
    """
    TODO
    """

    def __init__(self, name=None, text=None, data={}, start={}, chains=1, tune=1000):
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
        chains : int, optional
            Number of parallel chains.
        tune : int, optional
            Number of adaptations steps.
        """

        if name is None == text is None:
            raise ValueError('Either model name or model text must be provided.')

        self.console = pyjags.console.Console()
        with _get_model_path(name, text) as path:
            self.console.checkModel(path)

        data = _to_numpy_array(data)
        unused = set(data.keys()) - set(self.variables)
        if unused:
            raise ValueError('Unused data for variables: {}'.format(','.join(unused)))
        self.console.compile(data, chains, True)

        rng_name = start.pop('.RNG.name')
        for chain in self.chains:
            self.console.setRNGname(rng_name, chain)

        start = _to_numpy_array(start)
        unused = set(start.keys()) - set(self.variables) - {'.RNG.seed', '.RNG.state'}
        if unused:
            raise ValueError('Unused start values for variables: {}'.format(','.join(unused)))

        for chain in self.chains:
            self.console.setParameters(start, chain)
        self.console.initialize()

        if tune:
            self.adapt(tune)

    def update(self, iterations):
        # TODO progress bar?
        self.console.update(iterations)

    def sample(self, vars, iterations, thin=1, monitor_type="trace"):
        """
        Parameters
        ----------
        vars : list of variables
        iterations : int
        thin : int

        Returns
        -------

        """

        # Enable variable monitoring
        for name in vars:
            self.console.setMonitor(name, thin, monitor_type)
        self.update(iterations)
        samples = self.console.dumpMonitors(monitor_type, False)
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

model = Model(
    name='model.jags',
    data={'x':[[1,0,1],[1,1,1]]},
    start={
        '.RNG.name': 'base::Wichmann-Hill',
        '.RNG.seed': 1
    },
    chains=1)
samples = model.sample(model.variables, 100)
x = samples['x']
