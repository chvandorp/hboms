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
        np.random.seed(1234)  # for reproducibility
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


    def test_correlation_level_model(self):
        """
        Correlated parameters can have a fixed level.
        Make sure such models compile and estimate 
        reasonable correlation values.
        """

        ## use a fixed numpy seed
        np.random.seed(seed=233423)

        ## set-up the model
        K_A = 20
        cat_list = [f"A{i}" for i in range(K_A)]

        covs = [hboms.Covariate("A", "cat", cat_list)]

        params = [
            hboms.Parameter("a", 1.0, "random", level="A"),
            hboms.Parameter("b", 1.0, "random", level="A"),
            hboms.Parameter("sigma", 0.1, "fixed")
        ]

        corr_gt = 0.7 # the ground truth correlation value
        Sigma = (1 - corr_gt) * np.eye(2) + corr_gt * np.ones((2,2))

        corrs = [hboms.Correlation(["a", "b"], value=Sigma)]
        state = [hboms.Variable("x")]
        odes = "ddt_x = a * x;"
        init = "x_0 = b;"
        obs = [hboms.Observation("X")]
        dists = [hboms.StanDist("lognormal", "X", ["log(x)", "sigma"])]

        model = hboms.HbomsModel(
            name   = "corr_level_test",
            params = params,
            covariates   = covs,
            correlations = corrs,
            state  = state,
            odes   = odes,
            init   = init,
            obs    = obs,
            dists  = dists,
            model_dir = self.model_dir,
        )

        ## simulate data

        a_gt, b_gt = 0.1, 0.1

        R, N = 50, 10

        cats = [cat_list[np.random.choice(K_A)] for _ in range(R)]

        ts = np.linspace(1, 10, 10)
        Time = [ts for _ in range(R)]

        omeg = 0.1 ## standard deviation of the random effects
        corr_re = np.array([sts.multivariate_normal.rvs(mean=np.zeros(2), cov=Sigma) for _ in range(K_A)])

        a_gt_indiv = a_gt * np.exp(omeg * corr_re[:,0])
        b_gt_indiv = b_gt * np.exp(omeg * corr_re[:,1])

        cat_idx = [cat_list.index(c) for c in cats]

        sig = 0.01 ##  standard deviation of the measurement noise

        # generate the data
        Xs = [
            b_gt_indiv[cat_idx[r]] * np.exp(a_gt_indiv[cat_idx[r]] * ts + sts.norm.rvs(scale=sig, size=N)) 
            for r in range(R)
        ]

        ## fit the model

        stan_data = {"Time" : Time, "X" : Xs, "A" : cats}
        model.sample(
            stan_data, 
            step_size=0.01,
            show_progress=False,
            chains=1,
            iter_warmup=500,
            iter_sampling=500,
            seed=56756,
        )


        ## compare estimated value with ground truth

        Sigma_est = model.fit.stan_variable("corr_a_b")
        corr_est = Sigma_est[:,0,1]
        lb, ub = np.percentile(corr_est, q=[2.5, 97.5])

        assert lb < corr_gt and corr_gt < ub, f"ground truth correlation not in estimated 95% CrI: [{lb:0.3f}, {ub:0.3f}]"

        assert lb > 0, f"estimated correlation is not significantly positive: {lb:0.3f} < 0.0"



