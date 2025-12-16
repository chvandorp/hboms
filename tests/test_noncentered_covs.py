import pytest
import hboms
import numpy as np
import scipy.stats as sts
import os


class TestNoncenteredCatCov:
    def test_noncentered_catcov(self):
        """
        Test that non-centered parameterization works with categorical covariates.
        Here we just check that the model compiles and samples without errors.
        """

        num_units = 10
        ts = np.arange(1, 10, 0.5)
        data = {
            "Time": [ts for _ in range(num_units)],
            "Group": ["A" if i % 2 == 0 else "B" for i in range(num_units)],
            "X" : [np.full(ts.shape, fill_value=0.0) for _ in range(num_units)]
        }

        params = [
            hboms.Parameter(
                "a", 0.2, "random", lbound=None, scale=0.1,
                covariates=["Group"],
                noncentered=True
            ),
            hboms.Parameter("sigma", 0.1, "fixed")
        ]

        group_cov = hboms.Covariate("Group", cov_type="cat", categories=["A", "B"])

        model = hboms.HbomsModel(
            name = "nc_test_model_non_centered_catcov",
            params = params,
            state = [hboms.Variable("x")],
            odes= "ddt_x = a * x * (1-x);",
            init= "x_0 = 0.01;",
            obs = [hboms.Observation("X")],
            dists = [hboms.StanDist("normal", "X", ["x", "sigma"])],
            covariates=[group_cov],
            model_dir="tests/stan-cache",
        )

        # the simulator ignores non-centered parameterization

        sim_data_pars = model.simulate(
            data, num_simulations=10, seed=123,
        )
        sim_data, _ = sim_data_pars[0]

        num_samples = 25

        model.sample(
            data=sim_data,
            iter_warmup=num_samples,
            iter_sampling=num_samples,
            chains=1,
            seed=123,
            step_size=0.01,
            show_progress=False,
        )

        loc_a = model.fit.stan_variable("loc_a")

        assert loc_a.shape == (num_samples, 2), f"loc_a should have shape ({num_samples}, {2})"

        a = model.fit.stan_variable("a")

        assert a.shape == (num_samples, num_units), f"a should have shape ({num_samples}, {num_units})"

        scale_a = model.fit.stan_variable("scale_a")

        assert scale_a.shape == (num_samples, ), f"scale_a should have shape ({num_samples}, )"



