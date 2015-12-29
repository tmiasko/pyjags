import unittest
import pyjags
import numpy as np


class TestModel(unittest.TestCase):

    def test_default_modules(self):
        self.assertEqual(
            ['basemod', 'bugs'],
            pyjags.list_modules())

    def test_sample_with_rng_seed_is_deterministic(self):
        model_string = b'''
        model {
            x ~ dbern(0.5)
        }
        '''
        start = {
            '.RNG.name': 'base::Wichmann-Hill',
            '.RNG.seed': 1
        }
        n = 100
        s1 = pyjags.Model(text=model_string, start=start).sample(n)
        s2 = pyjags.Model(text=model_string, start=start).sample(n)
        np.testing.assert_equal(s1, s2)

    def test_rng_name_set_and_get(self):
        expected_names = [
            'base::Marsaglia-Multicarry',
            'base::Marsaglia-Multicarry',
            'base::Mersenne-Twister',
            'base::Mersenne-Twister',
            'base::Wichmann-Hill',
        ]
        model = b'model { x ~ dbern(0.5) }'
        start = [{'.RNG.name': name, '.RNG.seed': 0} for name in expected_names]
        chains = len(start)
        m = pyjags.Model(text=model, start=start, chains=chains, tune=10)
        actual_names = [v for chain_state in m.state for k, v in chain_state.items() if k == '.RNG.name']
        self.assertEqual(expected_names, actual_names)

