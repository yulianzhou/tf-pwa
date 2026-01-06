"""
Examples for Helicity Angle
---------------------------

Here we use the same config in `ex3_particle_config.py`

"""

config_str = """
decay:
    A:
       - [R1, B]
       - [R2, C]
       - [R3, D]
    R1: [C, D]
    R2: [B, D]
    R3: [B, C]

particle:
    $top:
       A: { mass: 1.86, J: 0, P: -1}
    $finals:
       B: { mass: 0.494, J: 0, P: -1}
       C: { mass: 0.139, J: 0, P: -1}
       D: { mass: 0.139, J: 0, P: -1}
    R1: [ R1_a, R1_b ]
    R1_a: { mass: 0.7, width: 0.05, J: 1, P: -1}
    R1_b: { mass: 0.5, width: 0.05, J: 0, P: +1}
    R2: { mass: 0.824, width: 0.05, J: 0, P: +1}
    R3: { mass: 0.824, width: 0.05, J: 0, P: +1}

"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.particle import BaseParticle

config = ConfigLoader(yaml.full_load(config_str))

# %%
# TFPWA provide useful class to use helicity angle.
# we can import it from data_trans module

from tf_pwa.data_trans.helicity_angle import HelicityAngle

# %%
# The class use DecayChain as input.
# We can get the DecayChain for ConfigLoader

decay_chain = config.get_decay(False).get_decay_chain("R1")

hel = HelicityAngle(decay_chain)

# %%
# HelcicityAngle provide two main method, `find_variable` and `build_data`.
# `find_variable` can find the variables in data dict

data = config.generate_phsp(1)

mass, costheta, phi = hel.find_variable(data)

print(mass)
print(costheta)
print(phi)

# %%
# `build_data` is the inverse function of `cal_angle`, it can build 4-momentum from the helicity angles

p4 = hel.build_data(mass, costheta, phi)

print(p4)

# %%
# We can check that the 4-momentums have the same angle as what we input, within the precision range.

data2 = hel.cal_angle(p4)
mass2, costheta2, phi2 = hel.find_variable(data2)

print(mass2)
print(costheta2)
print(phi2)

assert all(abs(mass[i] - mass2[i]) < 1e-6 for i in mass.keys())
assert all(abs(a - b) < 1e-6 for a, b in zip(costheta, costheta2))
assert all(abs(a - b) < 1e-6 for a, b in zip(phi, phi2))

# %%
# The helicity angle and related masses have the same number of degree of freedom as 4-momentums, it can be used in many cases.
#
# Phase space generator
# ^^^^^^^^^^^^^^^^^^^^^
# The most common usage is the Phase Space generator, helicity angle is uniform in phase space. After generate related masses, we can generate uniform random number for helicity angle to build the full 4-momentums.
# See :doc:`../phasespace` for more information.

from tf_pwa.phasespace import PhaseSpaceGenerator

gen = PhaseSpaceGenerator(1.86, [0.139, 0.139, 0.494])
N = 1
mass_gen = gen.generate_mass(N)
costheta_gen = [
    np.random.uniform(-1, 1, size=(N,)),
    np.random.uniform(-1, 1, size=(N,)),
]
phi_gen = [
    np.random.uniform(-np.pi, np.pi, size=(N,)),
    np.random.uniform(-np.pi, np.pi, size=(N,)),
]

p4 = hel.build_data({"R1": mass_gen[0]}, costheta_gen, phi_gen)

# %%
# Particle Function
# ^^^^^^^^^^^^^^^^^
# We can fix the other variables to see the dependences of one variable.

m = np.linspace(0.3, 0.8)
mass = {"R1": m}
costheta = [np.zeros(m.shape), np.zeros(m.shape)]
phi = [np.zeros(m.shape), np.zeros(m.shape)]

amp = config.get_amplitude()
data = config.data.cal_angle(hel.build_data(mass, costheta, phi))

with amp.temp_used_res(["R1_b"]):
    prob = amp(data)

# %%
# There is a useful function `get_particle_function` provided by ConfigLoader.

f = config.get_particle_function("R1_b")
prob2 = tf.abs(f(m)[:, 0]) ** 2

plt.clf()
plt.plot(m, prob2.numpy(), label="from particle function")
plt.plot(m, prob.numpy(), ls="--", label="from amplitude")
plt.legend()
plt.xlabel("mass")
plt.ylabel("$|A|^2$")
plt.show()


# %%
# Resolution
# ^^^^^^^^^^
# Another common usage in TFPWA is used in resolution, when we ony consider resolution in one mass, we can fix the helicity angles, replace the mass into different values, rebuild the 4-monmentums and sum over them to do the integration.
# See :doc:`../resolution` for more information.
#
# .. math:
#     \int f(p^{\mu}(m + \delta) )G(\delta) d \delta \approx \sum_i f(p^{\mu}(m + \delta_i) )G(\delta_i)


probs = []
delta = np.linspace(-0.2, 0.2, 20)
for delta_i in delta:
    masses_new = mass.copy()
    new_mass = m + delta_i
    # be careful for the boundary, the outer value would be NaN
    new_mass = tf.clip_by_value(
        new_mass, 0.139 + 0.139 + 1e-6, 1.86 - 0.494 - 1e-6
    )
    masses_new[BaseParticle("R1")] = new_mass
    p4 = hel.build_data(masses_new, costheta, phi)
    data = config.data.cal_angle(p4)
    with amp.temp_used_res(["R1_b"]):
        prob_i = amp(data)
    probs.append(prob_i)

gauss = tf.exp(-(delta**2) / 0.05**2 / 2)
gauss = gauss / tf.reduce_sum(gauss)

smear_prob = tf.reduce_sum(tf.stack(probs, axis=-1) * gauss, axis=-1)

plt.clf()
plt.plot(m, smear_prob.numpy(), label="smear $|A|^2$")
plt.plot(m, prob.numpy(), ls="--", label="origin $|A|^2$")
plt.legend()
plt.xlabel("mass")
plt.ylabel("$|A|^2$")
plt.show()

# %%
# Distribution transform
# ^^^^^^^^^^^^^^^^^^^^^^
# Finally, we can use the helicity angle to study the distribution of other variables.
#
# .. math::
#   P(y) = \left| \frac{\partial y}{\partial x} \right|^{-1} P(x)
#
# The jacobian can be calculated by automatic differentiation.
#
# For example, we can verify that the Dalitz plot variable is uniform distribution.
# First we generate some variable in phase space.

mass, costheta, phi = hel.find_variable(config.generate_phsp(3))


def mass2(p4):
    return tf.reduce_sum(p4 * p4 * np.array([1, -1, -1, -1]), axis=-1)


b, c, d = [BaseParticle(i) for i in "BCD"]

# %%
# and than calculate the jacobian

x = tf.Variable(tf.stack([mass[BaseParticle("R1")], costheta[1]], axis=-1))
with tf.GradientTape() as tape:
    mass[BaseParticle("R1")] = x[:, 0]
    costheta[1] = x[:, 1]
    p4 = hel.build_data(mass, costheta, phi)
    s12 = mass2(p4[b] + p4[c])
    s23 = mass2(p4[c] + p4[d])
    y = tf.stack([s12, s23], axis=-1)

jac = tape.batch_jacobian(y, x)

# %%
# These are Dalitz plot variables

print("s12, s23 = ", s12, s23)

# %%
# the jacobian of the first variable is

print(jac[0])

# %%
# We can see that the probability density at such points have the same values

prob = 1 / tf.abs(tf.linalg.det(jac)) * hel.eval_phsp_factor(mass)

print("prob = ", prob)
