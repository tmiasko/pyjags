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

import os.path
import unittest

import numpy as np

import pyjags


class TestModel(unittest.TestCase):

    def test_model_from_string(self):
        # No exceptions should be thrown.
        pyjags.Model(code='model { y ~ dbern(1) }')
        pyjags.Model(code=b'model { y ~ dbern(1) }')

    def test_model_from_file(self):
        path = os.path.dirname(__file__)
        path = os.path.join(path, 'model.jags')
        pyjags.Model(file=path)

    def test_model_is_required(self):
        with self.assertRaises(ValueError):
            pyjags.Model()

    def test_sample_with_rng_seed_is_deterministic(self):
        model_string = '''
        model {
            x ~ dbern(0.5)
        }
        '''
        init = {
            '.RNG.name': 'base::Wichmann-Hill',
            '.RNG.seed': 1
        }
        n = 100
        s1 = pyjags.Model(model_string, init=init).sample(n)
        s2 = pyjags.Model(model_string, init=init).sample(n)
        np.testing.assert_equal(s1, s2)

    def test_rng_name_set_and_get(self):
        expected_names = [
            'base::Marsaglia-Multicarry',
            'base::Marsaglia-Multicarry',
            'base::Mersenne-Twister',
            'base::Mersenne-Twister',
            'base::Wichmann-Hill',
        ]
        model = 'model { x ~ dbern(0.5) }'
        init = [{'.RNG.name': name, '.RNG.seed': 0} for name in expected_names]
        chains = len(init)
        m = pyjags.Model(model, init=init, chains=chains, tune=10)
        actual_names = [v for chain_state in m.state for k, v in chain_state.items() if k == '.RNG.name']
        self.assertEqual(expected_names, actual_names)

    def test_empty_array(self):
        # This used to throw an exception.
        model = 'model { x ~ dbern(1) }'
        pyjags.Model(model, data={'x': []})

    def test_invalid_length_of_initial_value_list_throws_exception(self):
        model = 'model { x ~ dbern(1) }'
        with self.assertRaises(ValueError):
            pyjags.Model(model, chains=2, init=[{'x': 1}])

    def test_invalid_type_of_init_value_list(self):
        model = 'model { x ~ dbern(1) }'
        with self.assertRaises(ValueError):
            pyjags.Model(model, chains=2, init=1234)

    def test_missing_input_data(self):
        model = '''
        model {
            for (i in 1:length(x)) {
                x[i] ~ dbern(0.5)
            }
        }'''

        data = {'x': np.ma.masked_outside([0, 1, -1], 0, 1)}
        m = pyjags.Model(model, data=data, chains=1)
        n = 100
        s = m.sample(n, vars=['x'])

        x1 = s['x'][0,:,:]
        x2 = s['x'][1,:,:]
        x3 = s['x'][2,:,:]

        # Observed values, samples should be constant
        np.testing.assert_equal([0] * n, x1.flatten())
        np.testing.assert_equal([1] * n, x2.flatten())
        # Missing value, samples should vary between 0 and 1
        self.assertIn(0, x3)
        self.assertIn(1, x3)

    @unittest.skipIf(pyjags.version() < (4,0,0), "Not supported before JAGS 4.0.0")
    def test_missing_sample_data(self):
        model = '''
        model {
            x[1] ~ dnorm(0, 10)
            x[3] ~ dnorm(0, 15)
        }'''

        m = pyjags.Model(model, chains=2)
        s = m.sample(10, vars=['x'])

        x = s['x']
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        # x[2] should be completely masked
        self.assertTrue(np.all(x2.mask))
        # x[1] and x[3] should be plain unmasked numpy arrays
        self.assertFalse(np.ma.is_mask(x1))
        self.assertFalse(np.ma.is_mask(x3))

    def test_unused_variables_throws_exception(self):
        model = 'model { x ~ dbern(0.5) }'

        with self.assertRaises(ValueError):
            pyjags.Model(model, data={'x': 1, 'y': 2})

        with self.assertRaises(ValueError):
            pyjags.Model(model, init={'x': 1, 'y': 2})

if __name__ == '__main__':
    unittest.main()
