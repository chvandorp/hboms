import hboms

class TestTransformedParams:
    def test_dependencies(self):
        # define a very simple model"
        params = [
            hboms.Parameter("a", 1.0, "random"),
            hboms.Parameter("b", 1.0, "fixed"),
            hboms.Parameter("c", "a + b", "trans"),
            hboms.Parameter("tau", 0.1, "fixed"),
            hboms.Parameter("sigma", "1/tau", "trans"),
        ]

        state = [hboms.Variable("x")]
        obs = [hboms.Observation("X")]
        dists = [hboms.StanDist("normal", "X", ["x", "sigma"])]
        init = "x_0 = 0.1;"
        odes = "ddt_x = -c * x;"
        model = hboms.HbomsModel(
            "decay-model", state, odes, init, params, obs, dists,
            compile_model=False
        )
        trans_params = [p for p in model._params if p.get_type() in ["trans", "trans_indiv"]]
        dependencies = {p.name : p._dependencies for p in trans_params}

        # we expect that c has dependencies a and b and sigma has tau
        deps_c = [p.name for p in dependencies["c"]]
        deps_sigma = [p.name for p in dependencies["sigma"]]

        assert deps_c == ["a", "b"]

        assert deps_sigma == ["tau"]


    def test_ordering(self):
        params = [
            hboms.Parameter("d", "c", "trans"),
            hboms.Parameter("c", "b", "trans"),
            hboms.Parameter("b", "a", "trans"),
            hboms.Parameter("a", 1.0, "random"),
            hboms.Parameter("sigma", 0.1, "fixed"),
        ]

        state = [hboms.Variable("x")]
        obs = [hboms.Observation("X")]
        dists = [hboms.StanDist("normal", "X", ["x", "sigma"])]
        init = "x_0 = 0.1;"
        odes = "ddt_x = -c * x;"
        model = hboms.HbomsModel(
            "decay-model", state, odes, init, params, obs, dists,
            compile_model=False
        )

        trans_params = [p for p in model._params if p.get_type() in ["trans", "trans_indiv"]]
        trans_params.sort(key = lambda x: x.get_rank())
        trans_param_names = [p.name for p in trans_params]

        # we expect that e.g. b comes before c as c depends on b
        assert trans_param_names == ["b", "c", "d"]

        

        








