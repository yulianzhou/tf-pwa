import contextlib

from tf_pwa.amp.core import variable_scope
from tf_pwa.data import LazyCall, data_shape, split_generator


class AbsPDF:
    def __init__(
        self,
        *args,
        name="",
        vm=None,
        polar=None,
        use_tf_function=False,
        no_id_cached=False,
        jit_compile=False,
        **kwargs
    ):
        self.name = name
        with variable_scope(vm) as vm:
            if polar is not None:
                vm.polar = polar
            self.init_params(name)
            self.vm = vm
        self.vm = vm
        self.no_id_cached = no_id_cached
        self.f_data = []
        if use_tf_function:
            from tf_pwa.experimental.wrap_function import WrapFun

            self.cached_fun = WrapFun(self.pdf, jit_compile=jit_compile)
        else:
            self.cached_fun = self.pdf

    def get_params(self, trainable_only=False):
        return self.vm.get_all_dic(trainable_only)

    def set_params(self, var):
        self.vm.set_all(var)

    @contextlib.contextmanager
    def temp_params(self, var):
        params = self.get_params()
        self.set_params(var)
        yield var
        self.set_params(params)

    @property
    def variables(self):
        return self.vm.variables

    @property
    def trainable_variables(self):
        return self.vm.trainable_variables

    def cached_available(self):
        return True

    def __call__(self, data, cached=False):
        if isinstance(data, LazyCall):
            data = data.eval()
        if id(data) in self.f_data or self.no_id_cached:
            if self.cached_available():  # decay_group.not_full:
                return self.cached_fun(data)
        else:
            self.f_data.append(id(data))
        ret = self.pdf(data)
        return ret


class AmplitudeModel(AbsPDF):
    def __init__(self, decay_group, **kwargs):
        self.decay_group = decay_group
        super().__init__(**kwargs)
        res = decay_group.resonances
        self.used_res = res
        self.res = res

    def init_params(self, name=""):
        self.decay_group.init_params(name)

    def __del__(self):
        if hasattr(self, "cached_fun"):
            del self.cached_fun
        # super(AmplitudeModel, self).__del__()

    def cache_data(self, data, split=None, batch=None):
        for i in self.decay_group:
            for j in i.inner:
                print(j)
        if split is None and batch is None:
            return data
        else:
            n = data_shape(data)
            if batch is None:  # split个一组，共batch组
                batch = (n + split - 1) // split
            ret = list(split_generator(data, batch))
            return ret

    def set_used_res(self, res):
        self.decay_group.set_used_res(res)

    def set_used_chains(self, used_chains):
        self.decay_group.set_used_chains(used_chains)

    def partial_weight(self, data, combine=None):
        if isinstance(data, LazyCall):
            data = data.eval()
        return self.decay_group.partial_weight(data, combine)

    def partial_weight_interference(self, data):
        return self.decay_group.partial_weight_interference(data)

    def chains_particle(self):
        return self.decay_group.chains_particle()

    def cached_available(self):
        return not self.decay_group.not_full

    def pdf(self, data):
        ret = self.decay_group.sum_amp(data)
        return ret
