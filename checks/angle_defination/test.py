import copy
from pprint import pprint

import numpy as np
import yaml
from decayangle.decay_topology import Topology

from tf_pwa.amp.preprocess import BasePreProcessor, register_preprocessor
from tf_pwa.cal_angle import (
    add_mass,
    infer_momentum,
    parity_trans,
    struct_momentum,
)
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data import data_merge, data_to_numpy


def create_topo_map(decay_chain):
    particle_map = {v: k + 1 for k, v in enumerate(decay_chain.outs)}
    decay_map = {}
    remain_decay = list(decay_chain)
    while remain_decay:
        used_decay = []
        for decay in remain_decay:
            if all(i in particle_map for i in decay.outs):
                topo_id = tuple(particle_map[i] for i in decay.outs)
                particle_map[decay.core] = topo_id
                decay_map[topo_id] = decay
                used_decay.append(decay)
        if not used_decay:
            raise ValueError(
                "unsupport decay structure: {}".format(remain_decay)
            )
        remain_decay = [i for i in remain_decay if i not in used_decay]
    ret = Topology(0, decay_topology=particle_map[decay_chain.top])
    return ret, particle_map, decay_map


def create_topo_map_all(decay_struct):
    ret = []
    for decay_chain in decay_struct:
        ret.append(create_topo_map(decay_chain))
    return ret


def p4_conv(p4):
    p4 = data_to_numpy(p4)
    return np.concatenate([p4[..., 1:], p4[..., 0:1]], axis=-1)


def cal_angle_from_decayangle(p4, decay_struct):
    # build decayangle topo
    decay_struct = decay_struct.topology_structure()
    all_topo = create_topo_map_all(decay_struct)

    # build decayangle input
    particle_map = all_topo[0][1]
    momenta = {particle_map[i]: p4_conv(p4[i]) for i in p4}

    # helicity angle
    hel_angles = [topo[0].helicity_angles(momenta) for topo in all_topo]

    # alignment angle
    ref = all_topo[0][0]
    align_angles = []
    for topo in all_topo:
        # print(topo[0], ref, topo[0] == ref)
        if topo[0] == ref:
            align_angles.append({})
        else:
            align_angles.append(ref.relative_wigner_angles(topo[0], momenta))
            # align_angles.append(topo[0].relative_wigner_angles(ref, momenta))

    # build data structure as TFPWA
    data_p = struct_momentum(p4)  # , center_mass=center_mass)
    for dec in decay_struct:
        data_p = infer_momentum(data_p, dec)
        # print(data_p)
        # exit()
        data_p = add_mass(data_p, dec)

    data_c = {}
    for topo, chain, hel, align in zip(
        all_topo, decay_struct, hel_angles, align_angles
    ):
        data_c[chain] = {}
        particle_map = topo[1]
        for decay in chain:
            data_c[chain][decay] = {}
            shift = 0
            for particle in decay.outs:
                topo_id1 = particle_map[decay.core]
                topo_id = particle_map[particle]
                tmp = {}
                if shift == 0:
                    tmp["ang"] = {
                        "alpha": -hel[topo_id1].psi_rf,
                        "beta": -hel[topo_id1].theta_rf,
                        "gamma": np.zeros_like(hel[topo_id1].psi_rf),
                    }
                    shift = 1
                else:
                    tmp["ang"] = {
                        "alpha": -np.pi - hel[topo_id1].psi_rf,
                        "beta": np.pi + hel[topo_id1].theta_rf,
                        "gamma": np.zeros_like(hel[topo_id1].psi_rf),
                    }
                if topo_id in align:
                    # tmp["aligned_angle"] = {
                    #     "alpha": align[topo_id].psi_rf,
                    #     "beta": align[topo_id].theta_rf,
                    #     "gamma": align[topo_id].phi_rf
                    # }
                    tmp["aligned_angle"] = {
                        "alpha": align[topo_id].phi_rf,
                        "beta": align[topo_id].theta_rf,
                        "gamma": align[topo_id].psi_rf,
                    }

                data_c[chain][decay][particle] = tmp
    return {"particle": data_p, "decay": data_c}


@register_preprocessor("decayangle")
class DecayAnglePreProcessor(BasePreProcessor):
    def __call__(self, x, **kwargs):
        p4 = x["p4"]
        if self.kwargs.get("cp_trans", False):
            charges = x.get("extra", {}).get("charge_conjugation", None)
            p4 = {k: parity_trans(v, charges) for k, v in p4.items()}
        ret = cal_angle_from_decayangle(p4, self.decay_struct, **kwargs)
        return ret


def angle_norm(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


def test_main():
    with open("config.yml") as f:
        config_dic = yaml.full_load(f)
    config_ref = ConfigLoader(config_dic)

    config_dic2 = copy.deepcopy(config_dic)
    config_dic2["data"]["preprocessor"] = "decayangle"
    config2 = ConfigLoader(config_dic2)

    # the same params required the same conversion, so not always the same
    config2.set_params(config_ref.get_params())

    phsp = config_ref.generate_phsp_p(2000)

    a1 = config_ref.eval_amplitude(phsp)
    a2 = config2.eval_amplitude(phsp)
    print(a1, a2)
    assert np.allclose(a1.numpy(), a2.numpy())
    data1 = config_ref.data.cal_angle(phsp)
    data2 = config2.data.cal_angle(phsp)

    pprint(data_merge(data1, data2))
    for decay_chain in data1["decay"]:
        for decay in decay_chain:
            for particle in decay.outs:
                data_a = data1["decay"][decay_chain][decay][particle]
                data_b = data2["decay"][decay_chain][decay][particle]
                # due to the different reference used, only "D" have the same alignment angle
                if str(particle) == "D" and "aligned_angle" in data_b:
                    for key in ["alpha", "beta", "gamma"]:
                        print(
                            angle_norm(
                                data_a["aligned_angle"][key]
                                - data_b["aligned_angle"][key]
                            )
                        )
                        assert np.allclose(
                            angle_norm(
                                data_a["aligned_angle"][key]
                                - data_b["aligned_angle"][key]
                            ),
                            0,
                        )
                if "ang" in data_b:
                    for key in ["alpha", "beta"]:
                        assert np.allclose(
                            angle_norm(
                                data_a["ang"][key] - data_b["ang"][key]
                            ),
                            0,
                        )


if __name__ == "__main__":
    test_main()
