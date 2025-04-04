"""
Test a number of simple models. Do they compile? Do they run?
Do we get reasonable output?
"""

import hboms
import numpy as np
import scipy.stats as sts
import os


class TestSolvedModel:
    model_dir = os.path.join("tests", "stan-cache")

    def test_linregress(self):
        """model with no ODEs, but just the solution defined as a transform"""

        params = [
            hboms.Parameter("a", 1.0, "fixed", lbound=None),
            hboms.Parameter("b", 0.5, "random", lbound=None),
            hboms.Parameter("sigma", 0.5, "fixed"),
        ]

        state = []
        trans_state = [hboms.Variable("x")]
        odes = ""
        init = ""
        transform = "x = a*t + b;"
        obs = [hboms.Observation("X")]
        dists = [hboms.StanDist("normal", "X", ["x", "sigma"])]

        # compile the model
        model = hboms.HbomsModel(
            name = "solved-model",
            state = state,
            odes = odes,
            init = init,
            params = params,
            obs = obs,
            dists = dists,
            trans_state = trans_state,
            transform = transform,
            model_dir=self.model_dir,
        )

        # create some data
        np.random.seed(seed=3454)
        ts = np.linspace(0, 5, 11)[1:] # skip t=0
        num_units = 12
        a_gt = 1.0
        b_gt = 0.5
        sigma_gt = 0.5
        X = [
            a_gt * ts + b_gt + sts.norm.rvs(scale=sigma_gt, size=ts.shape)
            for r in range(num_units) 
        ]

        data = {
            "X": X,
            "Time": [ts for r in range(num_units)],
        }

        # fit the model
        model.sample(
            data=data,
            chains=1,
            show_progress=False,
            iter_warmup=500,
            iter_sampling=500,
            seed=4562,
        )

        # check the results
        a_hat = model.fit.stan_variable("a")
        b_hat = model.fit.stan_variable("loc_b")
        sigma_hat = model.fit.stan_variable("sigma")

        lb, ub = np.percentile(a_hat, q=[0.25, 99.75])
        assert lb < a_gt and a_gt < ub
        lb, ub = np.percentile(b_hat, q=[0.25, 99.75])
        assert lb < b_gt and b_gt < ub
        lb, ub = np.percentile(sigma_hat, q=[0.25, 99.75])
        assert lb < sigma_gt and sigma_gt < ub

        
