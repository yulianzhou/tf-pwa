import os

import numpy as np
import yaml

from tf_pwa.config_loader import ConfigLoader

from .test_full import gen_toy, toy_config

this_dir = os.path.dirname(os.path.abspath(__file__))


def test_add_ref_amp(toy_config):
    with open(f"{this_dir}/config_cfit.yml") as f:
        config_dic = yaml.full_load(f)
    add_ref_amp = {
        "add_ref_amp": {
            "config": f"{this_dir}/config_bkg.yml",
            "params": {},
            "varname": "bg_value",
        }
    }
    config_dic["data"]["preprocessor"] = ["default", add_ref_amp]
    config = ConfigLoader(config_dic)
    data = config.get_data("data")[0]
    assert "bg_value" in data


def test_add_ref_amp_complex(toy_config):
    with open(f"{this_dir}/config_ref_amp.yml") as f:
        config_dic = yaml.full_load(f)
    add_ref_amp = {
        "add_ref_amp_complex": {
            "config": f"{this_dir}/config_toy.yml",
            "params": f"{this_dir}/exp_params.json",
        }
    }
    config_dic["data"]["preprocessor"] = ["default", add_ref_amp]
    config = ConfigLoader(config_dic)
    data = config.get_data("data")[0]
    amp = config.get_amplitude()
    a = amp(data)

    config_dic["data"]["preprocessor"] = [
        "default",
        add_ref_amp,
        "cached_shape",
    ]
    config_dic["data"]["amp_model"] = "cached_shape"
    config2 = ConfigLoader(config_dic)
    config2.set_params(config.get_params())
    data2 = config2.get_data("data")[0]
    amp2 = config2.get_amplitude()
    a2 = amp2(data2)
    assert np.allclose(a.numpy(), a2.numpy())


def test_repeat_values(toy_config, gen_toy):
    with open(f"{this_dir}/config_cfit.yml") as f:
        config_dic = yaml.full_load(f)
    repeat_values = {
        "repeat_values": {
            "varname": "tag",
            "values": [-1, 1],
            "data_type": ["data"],
        }
    }
    config_dic["data"]["preprocessor"] = ["default", repeat_values]
    config_dic["data"]["preprocessor_var"] = ["tag"]
    config = ConfigLoader(config_dic)
    data = config.get_data("data")[0]
    assert "tag" in data
    data = config.get_data("phsp")[0]
