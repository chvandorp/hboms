import pytest
import hboms
import os
import numpy as np

class TestDeltaPrior:
    model_dir = os.path.join("tests", "stan-cache")

    def test_delta_prior_parsing(self):
        import hboms.prior

        par_a = hboms.Parameter("a", 1.0, "random")
        ddp_1 = hboms.DiracDeltaPrior("loc_a")
        par_b = hboms.Parameter("b", 2.0, "random", loc_prior=ddp_1)
        ddp_2 = hboms.DiracDeltaPrior("scale_a")
        par_c = hboms.Parameter("c", 3.0, "random", scale_prior=ddp_2, noncentered=True)

        model = hboms.HbomsModel(
            "test_delta_prior", 
            params=[par_a, par_b, par_c],
            state=[hboms.Variable("x")],
            init="x_0 = 1.0;",
            odes="ddt_x = -a * x + b + c*t;",
            obs=[hboms.Observation("X")],
            dists=[hboms.StanDist("normal", "X", ["x", "0.2"])],
            model_dir=self.model_dir,
        )

        cpar_b = model._params[1]

        assert isinstance(cpar_b._loc_prior, hboms.prior.DiracDeltaPrior), "Expected loc_prior to be a DiracDeltaPrior"

        assert cpar_b._loc_prior.param == "loc_a", "Expected loc_prior param to be 'loc_a'" 

        cpar_c = model._params[2]

        assert isinstance(cpar_c._scale_prior, hboms.prior.DiracDeltaPrior), "Expected scale_prior to be a DiracDeltaPrior"

        assert cpar_c._scale_prior.param == "scale_a", "Expected scale_prior param to be 'scale_a'"

    def test_delta_prior_parsing_corr(self):
        par_a = hboms.Parameter("a", 1.0, "random")
        ddp_1 = hboms.DiracDeltaPrior("loc_a")
        par_b = hboms.Parameter("b", 2.0, "random", loc_prior=ddp_1)
        ddp_2 = hboms.DiracDeltaPrior("scale_a")
        par_c = hboms.Parameter("c", 3.0, "random", scale_prior=ddp_2)

        # include a correlation between all three parameters
        corr = hboms.Correlation(["a", "b", "c"])

        # test if the model compiles...
        model = hboms.HbomsModel(
            "test_delta_prior_corr", 
            params=[par_a, par_b, par_c],
            state=[hboms.Variable("x")],
            init="x_0 = 1.0;",
            odes="ddt_x = -a * x + b + c*t;",
            obs=[hboms.Observation("X")],
            dists=[hboms.StanDist("normal", "X", ["x", "0.2"])],
            correlations=[corr],
            model_dir=self.model_dir,
        )







    def test_delta_prior_sampling(self):
        par_a = hboms.Parameter("a", 0.3, "random", scale=0.1)
        ddp_loc = hboms.DiracDeltaPrior("loc_a")
        ddp_scale = hboms.DiracDeltaPrior("scale_a")
        par_b = hboms.Parameter("b", 0.5, "random", loc_prior=ddp_loc, scale_prior=ddp_scale, scale=0.2)

        model = hboms.HbomsModel(
            "test_delta_prior_sampling", 
            params=[par_a, par_b],
            state=[hboms.Variable("x")],
            init="x_0 = 0.01;",
            odes="ddt_x = a * x + b;",
            obs=[hboms.Observation("X")],
            dists=[hboms.StanDist("normal", "X", ["x", "0.2"])],
            model_dir=self.model_dir,
        )

        R = 15
        data = {
            "Time" : [np.linspace(1, 5, 10) for _ in range(R)]
        }

        sim_data_pars = model.simulate(data, 10, output_dir=self.model_dir, seed=17)

        sim_data, _ = sim_data_pars[0]

        model.sample(sim_data, iter_warmup=200, iter_sampling=200, chains=1, step_size=0.01, seed=19)

        loc_a_hat = model.fit.stan_variable("loc_a")
        loc_b_hat = model.fit.stan_variable("loc_b")

        scale_a_hat = model.fit.stan_variable("scale_a")
        scale_b_hat = model.fit.stan_variable("scale_b")

        assert np.allclose(loc_b_hat, loc_a_hat, atol=1e-5), "loc_b not equal to loc_a within tolerance"
        assert np.allclose(scale_b_hat, scale_a_hat, atol=1e-5), "scale_b not equal to scale_a within tolerance"






