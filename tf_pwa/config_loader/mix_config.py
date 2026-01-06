import functools

from tf_pwa.amp import AmplitudeModel
from tf_pwa.amp.core import Variable, variable_scope
from tf_pwa.data import EvalLazy
from tf_pwa.model import FCN

from .multi_config import MultiConfig


class MixAmplitude(AmplitudeModel):
    def __init__(
        self, amps, same_data=False, vm=None, base_idx=0, no_scale=False
    ):
        self.amps = amps
        self.vm = amps[0].vm
        self.same_data = same_data
        self.base_idx = base_idx
        self.cached_fun = []
        self.no_scale = no_scale
        with variable_scope(self.vm):
            self.scale = Variable("scale", shape=(len(self.amps),))
        self.scale.set_fix_idx(0, fix_vals=1)
        self.scale.set_value(1.0)

    def partial_weight(self, data, combine):
        if not self.same_data:
            data = data["datas"][self.base_idx]
        return self.amps[self.base_idx].partial_weight(data, combine)

    def __getattr__(self, name):
        return getattr(self.amps[self.base_idx], name)

    def pdf(self, data):
        ret = 0
        scale = 0
        scale_var = self.scale()
        for idx, amp in enumerate(self.amps):
            if not self.same_data and "datas" in data:
                data_i = data["datas"][idx]
            else:
                data_i = data
            w = data_i.get("weight", 1)
            ret = ret + amp(data_i) * w * scale_var[idx]
            scale = scale + w
        if self.no_scale:
            return ret
        return ret / scale * len(self.amps)

    def __call__(self, data):
        return self.pdf(data)


class MixConfig(MultiConfig):
    def __init__(
        self,
        *args,
        total_same=False,
        same_data=False,
        no_scale=False,
        **kwargs,
    ):
        super().__init__(*args, total_same=total_same, **kwargs)
        self.same_data = same_data
        self.no_scale = no_scale
        self.cached_amps = None

    def get_data(self, name):
        if self.same_data:
            return self.configs[0].get_data(name)
        all_data = [i.get_data(name) for i in self.configs]
        if all_data[0] is None:
            return all_data[0]
        ret = []
        n_data = len(all_data)
        for i in range(len(all_data[0])):
            tmp = {"datas": [j[i] for j in all_data]}
            tmp["weight"] = sum(j[i]["weight"] for j in all_data) / n_data
            ret.append(tmp)
        return ret

    def get_data_rec(self, name):
        ret = self.get_data(name + "_rec")
        if ret is None:
            ret = self.get_data(name)
        return ret

    def get_phsp_plot(self, tail=""):
        ret = self.get_data("phsp_plot" + tail)
        if ret is None:
            ret = self.get_data("phsp" + tail)
        return ret

    def get_all_data(self):
        datafile = ["data", "phsp", "bg", "inmc"]
        data, phsp, bg, inmc = [self.get_data(i) for i in datafile]
        self._Ngroup = len(data)
        assert len(phsp) == self._Ngroup
        if bg is None:
            bg = [None] * self._Ngroup
        if inmc is None:
            inmc = [None] * self._Ngroup
        assert len(bg) == self._Ngroup
        assert len(inmc) == self._Ngroup
        return data, phsp, bg, inmc

    def get_amplitude(self, vm=None, name="", base_idx=0):
        if self.cached_amps is None:
            self.cached_amps = self.get_amplitudes(vm=vm)
        return MixAmplitude(
            self.cached_amps,
            self.same_data,
            base_idx=base_idx,
            no_scale=self.no_scale,
        )

    @functools.lru_cache()
    def _get_model(self, vm=None, name=""):
        amp = self.get_amplitude(vm=vm, name=name)
        models = self.configs[0]._get_model(vm=vm, name=name, amp=amp)
        return models

    def get_fcn(self, datas=None, vm=None, batch=65000):
        all_data = datas
        if all_data is None:
            all_data = self.get_all_data()
        model = self._get_model(vm=vm)
        return self.configs[-1].get_fcn(
            all_data=all_data, vm=model[0].vm, batch=batch, model=model
        )

    def plot_partial_wave(self, *args, base_idx=0, **kwargs):
        amp = self.get_amplitude(base_idx=base_idx)
        data = self.get_data_rec("data")
        bg = self.get_data_rec("bg")
        phsp = self.get_phsp_plot()
        phsp_rec = self.get_phsp_plot("_rec")
        data_index_prefix = ()
        if not self.same_data:
            data_index_prefix = "datas", base_idx
        self.configs[base_idx].plot_partial_wave(
            *args,
            amp=amp,
            data=data,
            phsp=phsp,
            bg=bg,
            phsp_rec=phsp_rec,
            data_index_prefix=data_index_prefix,
            **kwargs,
        )
