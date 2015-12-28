__all__ = ['Model', 'load_module', 'unload_module']

from .console import Console

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
    """Load JAGS module."""
    if name not in JAGS_MODULES:
        path = os.path.join(JAGS_MODULE_DIR, name + JAGS_MODULE_EXT)
        lib = ctypes.cdll.LoadLibrary(path)
        JAGS_MODULES[name] = lib
    Console.loadModule(name)


def unload_module(name):
    """Unload JAGS module."""
    return Console.unloadModule(name)


def _to_numpy_array(src):
    # TODO
    d = {}
    for k, v in src.items():
        # Avoid 0-dimensional arrays. They would require special handling in C code.
        v = np.atleast_1d(v)
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

        self.console = Console()
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

