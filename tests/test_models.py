"""
Test a number of simple models. Do they compile? Do they run?
Do we get reasonable output?
"""

import hboms
import numpy as np
import scipy.stats as sts
import os


class TestModel:
    model_dir = os.path.join("tests", "stan-cache")
    odes = "ddt_x = -theta * x;"
    init = "x_0 = x0;"
    state = [hboms.Variable("x")]
    obs = [hboms.Observation("X")]
    dists = [hboms.StanDist("lognormal", "X", ["log(x)", "sigma"])]

    def test_simplest_model(self):
        """two-parameter model, 1D state space"""
        # build a model

        params = [
            hboms.Parameter("theta", 0.2, "random", scale=0.1),
            hboms.Parameter("sigma", 0.1, "fixed"),
            hboms.Parameter("x0", 1.0, "const"),
        ]
        name = "expon_decay"
        hbm = hboms.HbomsModel(
            name,
            self.state,
            self.odes,
            self.init,
            params,
            self.obs,
            self.dists,
            model_dir=self.model_dir,
        )

        # simulate a dataset

        R = 20
        time = [list(range(1, 11)) for _ in range(R)]
        data = {"Time": time}
        sims = hbm.simulate(data, 1, seed=4353)
        sim_data, sim_params = sims[0]

        # fit the model to the simulated data set

        hbm.sample(
            sim_data,
            chains=1,
            show_progress=False,
            iter_warmup=500,
            iter_sampling=500,
            seed=45643,
        )

        # test that the estimates are OK

        theta_hat = hbm.fit.stan_variable("theta")
        theta_hat_med = np.median(theta_hat, axis=0)
        res = sts.spearmanr(sim_params["theta"], theta_hat_med)
        assert res.pvalue < 1e-6 and res.statistic > 0.8

    def test_covariate_model(self):
        """3 parameter model, 1D state space, with covariate"""

        # build a model

        covs = [hboms.Covariate("U")]
        weight_U_theta_gt = 0.1
        params = [
            hboms.Parameter(
                "theta",
                0.2,
                "random",
                scale=0.05,
                covariates="U",
                cw_values={"U": weight_U_theta_gt},
            ),
            hboms.Parameter("sigma", 0.1, "fixed"),
            hboms.Parameter("x0", 1.0, "const"),
        ]
        name = "expon_decay_covar"
        hbm = hboms.HbomsModel(
            name,
            self.state,
            self.odes,
            self.init,
            params,
            self.obs,
            self.dists,
            covariates=covs,
            model_dir=self.model_dir,
        )

        # simulate a dataset

        R = 20
        time = [list(range(1, 11)) for _ in range(R)]
        U = sts.norm.rvs(loc=0, scale=1, size=R)
        data = {"Time": time, "U": U}
        sims = hbm.simulate(data, 1, seed=3452)
        sim_data, sim_params = sims[0]

        # fit the model to the simulated data set

        hbm.sample(
            sim_data,
            chains=1,
            show_progress=False,
            iter_warmup=500,
            iter_sampling=500,
            seed=4562,
        )

        # is the covariate weight estimate any good?

        weight_U_theta_hat = hbm.fit.stan_variable("weight_U_theta")

        llb, lb, uub = np.percentile(weight_U_theta_hat, q=[0.25, 2.5, 99.75])

        assert llb < weight_U_theta_gt and weight_U_theta_gt < uub and lb > 0.0

    def test_correlation_model(self):
        """3 parameter model, 1D state space, with correlation"""

        # build a model

        params = [
            hboms.Parameter("theta", 0.2, "random", scale=0.1),
            hboms.Parameter("x0", 1.0, "random", scale=0.1),
            hboms.Parameter("sigma", 0.1, "fixed"),
        ]
        rho_gt = 0.8
        corr_gt = np.array([[1.0, rho_gt], [rho_gt, 1.0]])
        corrs = [hboms.Correlation(["theta", "x0"], value=corr_gt)]
        name = "expon_decay_corr"
        hbm = hboms.HbomsModel(
            name,
            self.state,
            self.odes,
            self.init,
            params,
            self.obs,
            self.dists,
            correlations=corrs,
            model_dir=self.model_dir,
        )

        # simulate a dataset

        R = 20
        time = [list(range(1, 11)) for _ in range(R)]
        data = {"Time": time}
        sims = hbm.simulate(data, 1, seed=3452)
        sim_data, sim_params = sims[0]

        # fit the model to the simulated data set

        hbm.sample(
            sim_data,
            chains=1,
            show_progress=False,
            iter_warmup=500,
            iter_sampling=500,
            seed=4562,
        )

        # is the correlation estimate any good?

        rho_hat = hbm.fit.stan_variable("corr_theta_x0")[:, 0, 1]

        llb, lb, uub = np.percentile(rho_hat, q=[0.25, 2.5, 99.75])

        assert llb < rho_gt and rho_gt < uub and lb > 0.0        
