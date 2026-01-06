import os

import numpy as np

from tf_pwa.tests.test_full import gen_toy, toy_config_lazy

this_dir = os.path.dirname(os.path.abspath(__file__))


def test_lazycall(toy_config_lazy):
    results = toy_config_lazy.fit(batch=100000)
    assert np.allclose(results.min_nll, -204.9468493307786)
    toy_config_lazy.plot_partial_wave(
        prefix="toy_data/figure_lazy", batch=100000
    )
