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

__all__ = ['Model']

import collections
import contextlib
import sys
import tempfile

import numpy as np

from .console import Console, DUMP_ALL, DUMP_DATA, DUMP_PARAMETERS
from .modules import load_module
from .progressbar import  progress_bar, stepwise

# Special value indicating missing data in JAGS.
JAGS_NA = -sys.float_info.max*(1-1e-15)


def dict_to_jags(src):
    """Convert Python dictionary with array like values to format suitable
    for use with JAGS.

     * Returned arrays have at least one dimension.
     * Empty arrays are removed from the dictionary.
     * Masked values are replaced with JAGS_NA.
    """
    dst = {}
    for k, v in src.items():
        if np.ma.is_masked(v):
            v = np.ma.array(data=v, dtype=np.double, ndmin=1,
                            fill_value=JAGS_NA)
            v = np.ma.filled(v)
        else:
            v = np.atleast_1d(v)
        if not np.size(v):
            continue
        dst[k] = v
    return dst


def dict_from_jags(src):
    """Convert Python dictionary with array like values returned from JAGS to
    format suitable for use with Python.

     * Arrays containing JAGS_NA values are converted to numpy MaskedArray.
    """
    dst = {}
    for k, v in src.items():
        mask = v == JAGS_NA
        # Don't mask if it not necessary
        if np.any(mask):
            v = np.ma.masked_equal(v, JAGS_NA, copy=False)
        dst[k] = v
    return dst


@contextlib.contextmanager
def model_path(file=None, code=None, encoding='utf-8'):
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


class Model:
    """High level representation of JAGS model.

    Attributes
    ----------
    chains : int
        A number of chains in the model.

    Note
    ----
    In JAGS arrays are indexed from 1. On the other hand Python uses 0 based
    indexing. It is important to keep this in mind when providing data to JAGS
    and interpreting resulting samples. For example, what in JAGS would be
    x[4,2,7] in Python is x[3,1,6].

    Note
    ----
    The JAGS supports data sets where some of observations have no value.
    In PyJAGS those missing values are described using numpy MaskedArray.
    For example, to create a model with observations x[1] = 0.25, x[3] = 0.75,
    and observation x[2] missing, we would provide following data to Model
    constructor:

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

    def __init__(self, code=None, data=None, init=None, chains=4, adapt=1000,
                 file=None, encoding='utf-8', generate_data=True,
                 progress_bar=True):
        """
        Create a JAGS model and run adaptation steps.

        Parameters
        ----------
        code : str or bytes, optional
            Code of the model to load. Model may be also provided with file
            keyword argument.
        file : str, optional
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

             * '.RNG.name'  str, name of random number generator
             * '.RNG.seed'  int, seed for random number generator
             * '.RNG.state' array, may be specified instead of seed, shape of
               array depends on particular generator used
        data : dict, optional
            Dictionary with observed nodes in the model. Keys are variable
            names and values should be convertible to numpy arrays with shape
            compatible with one used in the model.

            The numpy.ma.MaskedArray can be used to provide data where some of
            observations are missing.
        generate_data : bool, optional
            If true, data block in the model is used to generate data.
        chains : int, 4 by default
            A positive number specifying number of parallel chains.
        adapt : int, 1000 by default
            An integer specifying number of adaptations steps.
        encoding : str, 'utf-8' by default
            When model code is provided as a string, this specifies its encoding.
        progress_bar : bool, optional
            If true, enables the progress bar.
        """

        # Ensure that default modules are loaded.
        load_module('basemod')
        load_module('bugs')

        self.enable_progress_bar = progress_bar
        self.chains = int(chains)
        adapt = int(adapt)

        # Load the model
        self.console = Console()
        with model_path(file, code, encoding) as path:
            self.console.checkModel(path)

        # Compile the model and extract data
        data = {} if data is None else dict_to_jags(data)
        unused = set(data.keys()) - set(self.variables)
        if unused:
            raise ValueError(
                'Unused data for variables: {}'.format(','.join(unused)))
        self.console.compile(data, self.chains, generate_data)
        for v in self.data.values():
            v.setflags(write=False)

        # Set parameters and configure random number generators.
        init = {} if init is None else init
        if isinstance(init, collections.Mapping):
            init = [init] * self.chains
        elif not isinstance(init, collections.Sequence):
            raise ValueError('Init should be a sequence or a dictionary.')
        if len(init) != self.chains:
            raise ValueError(
                'Length of init sequence should equal the number of chains.')
        for data, chain in zip(init, range(1, self.chains + 1)):
            data = dict(data)
            rng_name = data.pop('.RNG.name', None)
            if rng_name is not None:
                self.console.setRNGname(rng_name, chain)
            data = dict_to_jags(data)
            unused = set(data.keys()) - set(self.variables) - {'.RNG.seed',
                                                               '.RNG.state'}
            if unused:
                raise ValueError(
                    'Unused initial values in chain {} for variables: {}'.format(
                        chain, ','.join(unused)))
            self.console.setParameters(data, chain)

        # Initialize the model and run adaptation steps.
        self.console.initialize()
        if adapt:
            self.adapt(adapt)

    def _update(self, iterations, header):
        progress = progress_bar(self.enable_progress_bar)
        with progress(self.console.update, iterations, header=header) as fun:
            stepwise(fun, iterations, .5)

    def update(self, iterations):
        """Updates the model for given number of iterations."""
        self._update(iterations, 'updating: ')

    def sample(self, iterations, vars=None, thin=1, monitor_type="trace"):
        """
        Creates monitors for given variables, runs the model for provided
        number of iterations and returns monitored samples.


        Parameters
        ----------
        iterations : int
            A positive integer specifying number of iterations.
        vars : list of str, optional
            A list of variables to monitor.
        thin : int, optional
            A positive integer specifying thinning interval.
        Returns
        -------
        dict
            Sampled values of monitored variables as a dictionary where keys
            are variable names and values are numpy arrays with shape:
            (dim_1, dim_n, iterations, chains). dim_1, ..., dim_n describe the
            shape of variable in JAGS model.
        """
        if vars is None:
            vars = self.variables
        monitored = []
        try:
            for name in vars:
                self.console.setMonitor(name, thin, monitor_type)
                monitored.append(name)
            self._update(iterations, 'sampling: ')
            samples = self.console.dumpMonitors(monitor_type, False)
            samples = dict_from_jags(samples)
        finally:
            for name in monitored:
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
        self._update(iterations, 'adapting: ')
        return self.console.checkAdaptation()

    @property
    def variables(self):
        """Variable names used in the model."""
        return self.console.variableNames()

    @property
    def state(self):
        """Values of model parameters and model data for each chain.

        See Also
        --------
        parameters
        data
        """
        return [dict_from_jags(self.console.dumpState(DUMP_ALL, chain))
                for chain in range(1, self.chains + 1)]

    @property
    def parameters(self):
        """Values of model parameters for each chain. Includes name of random
        number generator as '.RNG.name' and its state as '.RNG.state'.
        """
        return [dict_from_jags(self.console.dumpState(DUMP_PARAMETERS, chain))
                for chain in range(1, self.chains + 1)]

    @property
    def data(self):
        """Model data. Includes data provided during model construction and
        data generated as part of data block.
        """
        return dict_from_jags(self.console.dumpState(DUMP_DATA, 1))


