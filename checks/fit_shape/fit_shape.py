import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.utils import time_print

config = ConfigLoader("config.yml")

config.set_params("init_params.json")

config_mi = ConfigLoader("config_mi.yml")
config_mi.get_params()

# config_mi.set_params("final_params.json")

config_mi.vm.set_fix("MI_w_0r", unfix=True)
config_mi.vm.set_fix("MI_w_0i", unfix=True)


f1 = config.get_particle_function("BC1")
f2 = config.get_particle_function("BC2")

f = config_mi.get_particle_function("MI")

m = f.mass_linspace(10000)


fast_f = f.cached_call(m)
target_f = f1(m) + f2(m)


plot_count = 1


def f_loss():
    ret = tf.reduce_sum(tf.abs(fast_f() - target_f) ** 2)
    global plot_count
    if plot_count % 10 == 1:
        print(ret)
    plot_count += 1
    return ret


best_params = {}
best_loss = np.inf
best_fit_result = None
for i in range(1):
    fit_result = time_print(config_mi.vm.minimize)(f_loss)
    if fit_result.fun < best_loss:
        best_loss = fit_result.fun
        best_params = config_mi.get_params()
        best_fit_result = fit_result
    # reset random parameters
    config_mi2 = ConfigLoader("config_mi.yml")
    config_mi.set_params(config_mi2.get_params())

config_mi.set_params(best_params)
config_mi.save_params("final_params.json")

print(best_fit_result)

plt.plot(m, tf.math.imag(f(m)).numpy(), label="image fit")
plt.plot(m, tf.math.imag(f1(m) + f2(m)).numpy(), label="imag target")
plt.plot(m, tf.math.real(f(m)).numpy(), label="real fit")
plt.plot(m, tf.math.real(f1(m) + f2(m)).numpy(), label="real target")

plt.legend()
plt.savefig("fit_results.png")
