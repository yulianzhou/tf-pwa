import copy
import functools
import random
import os
import yaml

from tf_pwa.amp import (
    DecayChain,
    DecayGroup,
    HelicityDecay,
    get_decay,
    get_decay_chain,
    get_particle,
    split_particle_type,
)

from .base_config import BaseConfig


def set_min_max(dic, name, name_min, name_max):
    if name not in dic and name_min in dic and name_max in dic:
        dic[name] = (
            random.random() * (dic[name_max] - dic[name_min]) + dic[name_min]
        )


def decay_cut_ls(decay):
    if isinstance(decay, HelicityDecay):
        if len(decay.get_ls_list()) == 0:
            return False, f"{decay} ls not aviable {decay.get_ls_list()}"
    return True, ""


def decay_cut_isospin(decay):
    for i in decay:
        if isinstance(i, HelicityDecay):
            if i.get_isospinbreak():
                return False, f"{i} is not aviable because of breacking isospin"
    return True, ""


def decay_cut_mass(decay):
    if isinstance(decay, HelicityDecay):
        if decay.core.mass is None or any(
            [j.mass is None for j in decay.outs]
        ):
            True, ""
        # print(i, i.core.mass, [j.mass for j in i.outs])
        if decay.core.mass < sum([j.mass for j in decay.outs]):
            return (
                False,
                f"{decay} mass break {decay.core.mass} < {[j.mass for j in decay.outs]}",
            )
    return True, ""


class DecayConfig(BaseConfig):
    decay_chain_cut_list = {}
    decay_cut_list = {
        "ls_cut": decay_cut_ls,
        "mass_cut": decay_cut_mass,
        "isospin_cut": decay_cut_isospin,
    }

    def __init__(self, dic, share_dict={}):
        self.config = copy.deepcopy(dic)
        self.decay_chain_config = dic.get("decay_chain", {})
        self.data_config = dic.get("data", {})
        self.share_dict = share_dict
        self.particle_key_map = {
            "Par": "P",
            "m0": "mass",
            "g0": "width",
            "J": "J",
            "P": "P",
            "I": "I",
            "spins": "spins",
            "bw": "model",
            "model": "model",
            "bw_l": "bw_l",
            "running_width": "running_width",
        }
        self.cut_list = self.data_config.get("decay_chain_cut", ["ls_cut"])
        self.decay_cut_list = self.data_config.get("decay_cut", self.cut_list)
        self.decay_key_map = {"model": "model"}
        self.dec = self.decay_item(self.config["decay"])
        (
            self.particle_map,
            self.particle_property,
            self.top,
            self.finals,
        ) = self.particle_item(self.config["particle"], share_dict)
        self.full_decay = DecayGroup(
            self.get_decay_struct(
                self.dec,
                self.particle_map,
                self.particle_property,
                self.top,
                self.finals,
                self.decay_chain_config,
            )
        )
        if self.data_config.get("cp_trans", True):
            self.disable_allow_cc(self.full_decay)
        self.decay_struct = DecayGroup(
            self.get_decay_struct(
                self.dec, {}, self.particle_property, process_cut=False
            )
        )
        identical_particles = self.data_config.get("identical_particles", None)
        cp_particles = self.data_config.get("cp_particles", None)
        if identical_particles is not None:
            self.decay_struct.identical_particles = identical_particles
            self.full_decay.identical_particles = identical_particles
        if cp_particles is not None:
            self.decay_struct.cp_particles = cp_particles
            self.full_decay.cp_particles = cp_particles
        # self.decaygroup_ccidx = []

    @staticmethod
    def load_config(file_name, share_dict={}):
        if isinstance(file_name, dict):
            return copy.deepcopy(file_name)
        if isinstance(file_name, str):
            if file_name in share_dict:
                return DecayConfig.load_config(share_dict[file_name])
            with open(file_name) as f:
                ret = yaml.safe_load(f)
                if ret is None:
                    ret = {}
            return ret
        raise TypeError("not support config {}".format(type(file_name)))

    def get_decay(self, full=True):
        if full:
            return self.full_decay
        else:
            return self.decay_struct

    @staticmethod
    def _list2decay(core, outs):
        parts = []
        params = {}
        for j in outs:
            if isinstance(j, dict):
                for k, v in j.items():
                    params[k] = v
            else:
                parts.append(j)
        dec = {"core": core, "outs": parts, "params": params}
        return dec

    @staticmethod
    def decay_item(decay_dict):
        decs = []
        for core, outs in decay_dict.items():
            is_list = [isinstance(i, list) for i in outs]
            if all(is_list):
                for i in outs:
                    dec = DecayConfig._list2decay(core, i)
                    decs.append(dec)
            else:
                dec = DecayConfig._list2decay(core, outs)
                decs.append(dec)
        return decs

    @staticmethod
    def _do_include_dict(d, o, share_dict={}):
        s = DecayConfig.load_config(o, share_dict)
        for i in s:
            if i in d:
                if isinstance(d[i], dict):
                    s[i].update(d[i])
                    d[i] = s[i]
            else:
                d[i] = s[i]

    @staticmethod
    def particle_item_list(particle_list):
        particle_map = {}
        particle_property = {}
        for particle, candidate in particle_list.items():
            if isinstance(candidate, list):  # particle map
                if len(candidate) == 0:
                    particle_map[particle] = []
                for i in candidate:
                    if isinstance(i, str):
                        particle_map[particle] = particle_map.get(
                            particle, []
                        ) + [i]
                    elif isinstance(i, dict):
                        map_i, pro_i = DecayConfig.particle_item_list(i)
                        for k, v in map_i.items():
                            particle_map[k] = particle_map.get(k, []) + v
                        particle_property.update(pro_i)
                    else:
                        raise ValueError(
                            "value of particle map {} is {}".format(i, type(i))
                        )
            elif isinstance(candidate, dict):
                particle_property[particle] = candidate
            else:
                raise ValueError(
                    "value of particle {} is {}".format(
                        particle, type(candidate)
                    )
                )
        return particle_map, particle_property

    @staticmethod
    def particle_item(particle_list, share_dict={}):
        top = particle_list.pop("$top", None)
        finals = particle_list.pop("$finals", None)
        includes = particle_list.pop("$include", None)
        if includes:
            if isinstance(includes, list):
                for i in includes:
                    DecayConfig._do_include_dict(
                        particle_list, i, share_dict=share_dict
                    )
            elif isinstance(includes, str):
                DecayConfig._do_include_dict(
                    particle_list, includes, share_dict=share_dict
                )
            else:
                raise ValueError(
                    "$include must be string or list of string not {}".format(
                        type(includes)
                    )
                )
        particle_map, particle_property = DecayConfig.particle_item_list(
            particle_list
        )

        if isinstance(top, dict):
            particle_property.update(top)
        if isinstance(finals, dict):
            particle_property.update(finals)
        return particle_map, particle_property, top, finals

    def rename_params(self, params, is_particle=True):
        ret = {}
        if is_particle:
            key_map = self.particle_key_map
        else:
            key_map = self.decay_key_map
        for k, v in params.items():
            ret[key_map.get(k, k)] = v
        return ret

    def decay_chain_cut(self, decays):
        ret = []
        chains = []
        chains_ccidx = []
        cc_num = 0
        if os.path.exists("chains.inp"):
            with open("chains.inp","r") as f:
                for line in f:
                    chains.append(line.rstrip())
        for i in decays:
            flag = True
            for name in self.cut_list:
                if name not in DecayConfig.decay_chain_cut_list:
                    continue
                f = DecayConfig.decay_chain_cut_list[name]
                new_flag, msg = f(i)
                flag = flag and new_flag
                if not flag:
                    print(
                        "remove decay chain",
                        i,
                        "by",
                        name,
                        "\n\tbecause of",
                        msg,
                    )
                    break
            if flag and len(chains) != 0:
                if str(i) in chains:
                    ret.append(i)
                    r1 = str(i).split('->')[2].split('+')[0][:-1]
                    r2 = str(i).split('->')[3].split(', ')[1][:-1]
                    print(f"c1 = {r1}")
                    print(f"c2 = {r2}")
                    if r1 == r2:
                        chains_ccidx.append(len(chains_ccidx))
                    else:
                        flag_cc = True
                        for item in range(len(ret)-1):
                            r1cc = str(ret[item]).split('->')[2].split('+')[0][:-1]
                            r2cc = str(ret[item]).split('->')[3].split(', ')[1][:-1]
                            if r1 == r1cc and r2 == r2cc or r1 == r2cc and r2 == r1cc:
                                chains_ccidx.append(item)
                                chains_ccidx[item] = len(chains_ccidx) - 1
                                cc_num+=1
                                flag_cc = False
                        if flag_cc:
                            chains_ccidx.append(len(chains_ccidx))
        chains_ccidx.append(cc_num)
        self.decaygroup_ccidx = chains_ccidx
        print(chains_ccidx)
        print(self.decaygroup_ccidx)
        return ret

    def decay_cut(self, decays):
        ret = []
        for decay_chain in decays:
            flag = True
            for i in decay_chain:
                for name in self.cut_list:
                    if name not in DecayConfig.decay_cut_list:
                        continue
                    f = DecayConfig.decay_cut_list[name]
                    new_flag, msg = f(i)
                    flag = flag and new_flag
                    if not flag:
                        print(
                            "remove decay chain",
                            decay_chain,
                            "by",
                            name,
                            "\n\tbecause of",
                            msg,
                        )
                        break
                if not flag:
                    if i in i.core.decay:
                        i.core.decay.remove(i)
                    for j in i.outs:
                        if i in j.creators:
                            j.creators.remove(i)
                    break
            if flag:
                ret.append(decay_chain)
        return ret

    def get_decay_struct(
        self,
        decay,
        particle_map=None,
        particle_params=None,
        top=None,
        finals=None,
        chain_params={},
        process_cut=True,
    ):
        """get decay structure for decay dict"""
        particle_map = particle_map if particle_map is not None else {}
        particle_params = (
            particle_params if particle_params is not None else {}
        )

        base_particle_set = {}
        particle_set = {}

        def add_particle(name, _id):
            name = "{}:{}".format(name, _id)
            if name in particle_set:
                return particle_set[name]
            names = name.split(":")
            params = particle_params.get(names[0], {})
            params = self.rename_params(params)
            set_min_max(params, "mass", "m_min", "m_max")
            set_min_max(params, "width", "g_min", "g_max")
            part = get_particle(name, **params)
            particle_set[name] = part
            return part

        def add_base_particle(name):
            if name in base_particle_set:
                return base_particle_set[name]
            part = get_particle(name)  # , **params)
            base_particle_set[name] = part
            return part

        def wrap_particle(name):
            name_list = particle_map.get(name, [name])
            return [add_base_particle(i) for i in name_list]

        def all_combine(out):
            if len(out) < 1:
                yield []
            else:
                for i in out[0]:
                    for j in all_combine(out[1:]):
                        yield [i] + j

        decs = []
        new_decay_params = {}
        for dec in decay:
            core = wrap_particle(dec["core"])
            outs = [wrap_particle(j) for j in dec["outs"]]
            for i in core:
                for j in all_combine(outs):
                    dec_i = get_decay(i, j)
                    new_decay_params[dec_i] = dec["params"]
                    decs.append(dec_i)

        decay_list = {}

        def add_decay(a, b, params):
            b = tuple(b)
            if (a, b) not in decay_list:
                decay_list[(a, b)] = get_decay(a, b, **params)
            return decay_list[(a, b)]

        top_tmp, finals_tmp = set(), set()
        if top is None or finals is None:
            top_tmp, res, finals_tmp = split_particle_type(decs)
        if top is None:
            top_tmp = list(top_tmp)
            assert len(top_tmp) == 1, "not only one top particle"
            top = list(top_tmp)[0]
        else:
            if isinstance(top, list):
                assert len(top) == 1, "only one initial supported"
                top = top[0]
            if isinstance(top, str):
                top = base_particle_set[top]
            elif isinstance(top, dict):
                keys = list(top.keys())
                assert len(keys) == 1
                top = base_particle_set[keys.pop()]
            else:
                top = base_particle_set[str(top)]
        if finals is None:
            finals = list(finals_tmp)
        elif isinstance(finals, (list, dict)):
            finals = [base_particle_set[i] for i in finals]
        else:
            raise TypeError("{}: {}".format(finals, type(finals)))

        dec_chain = top.chain_decay()
        ret = []
        for i in dec_chain:
            if sorted(DecayChain(i).outs) == sorted(finals):
                all_params = chain_params.get("$all", {})
                count_input = {}
                count_output = {}
                all_dec = []
                for dec in i:
                    count_input[dec.core.name] = (
                        count_input.get(dec.core.name, -1) + 1
                    )
                    core = add_particle(
                        dec.core.name, count_input[dec.core.name]
                    )
                    outs = []
                    for j in dec.outs:
                        count_output[j.name] = count_output.get(j.name, -1) + 1
                        out = add_particle(j.name, count_output[j.name])
                        outs.append(out)
                    dec_i = add_decay(core, outs, new_decay_params[dec])
                    all_dec.append(dec_i)
                dec_c = get_decay_chain(all_dec, **all_params)
                ret.append(dec_c)
        # print(f"{ret = ")
        # [[chic1->R_pip0pim0+R_pip1pim1, R_pip0pim0->pip0+pim0, R_pip1pim1->pip1+pim1], [chic1->R_pip0pip1pim0+pim1, R_pip0pip1pim0->R_pip0pim0+pip1, R_pip0pim0->pip0+pim0], [chic1->R_pim0pim1pip1+pip0, R_pim0pim1pip1->R_pip1pim1+pim0, R_pip1pim1->pip1+pim1]]
        if process_cut:
            ret = self.decay_cut(ret)
            ret = self.decay_chain_cut(ret)
        if len(ret) == 0:
            raise RuntimeError("not decay chain aviable, check you config.yml")
        return ret

    def disable_allow_cc(self, decay_group):
        for decay_chain in decay_group:
            for decay in decay_chain:
                if hasattr(decay, "allow_cc"):
                    decay.allow_cc = False
