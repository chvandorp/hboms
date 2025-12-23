"""
Test a number of simple models. Do they compile? Do they run?
Do we get reasonable output?
"""

import pytest
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








class TestCatCovsWithFixedLevels:
    """
    Test that parameters with categorical covariates and fixed levels
    are handled correctly.
    """

    model_dir = os.path.join("tests", "stan-cache")
    odes="ddt_x = a*x*(1-x);"
    init="x_0 = 0.001;"
    state=[hboms.Variable("x")]
    obs=[hboms.Observation("X")]
    dists=[hboms.StanDist(obs_name="X", name="normal", params=["x", "sigma"])]

    covariates = [
        hboms.Covariate("A", cov_type="cat", categories=["A1", "A2", "A3"]),
        hboms.Covariate("B", cov_type="cat", categories=["B1", "B2", "B3", "B4", "B5", "B6"]),
    ]

    R = 10
    data = {
        "Time" : [np.linspace(1, 20, 20) for _ in range(R)],
        "A" : ["A1", "A1", "A1", "A2", "A2", "A3", "A3", "A3", "A3", "A3"],
        "B" : ["B1", "B1", "B2", "B3", "B3", "B4", "B5", "B6", "B6", "B6"],
    }


    def test_catcov_fixed_level(self):
        params = [
            hboms.Parameter("a", 0.5, "random", scale=0.2, covariates=["A"], level="B", level_type="fixed"),
            hboms.Parameter("b", 1.0, "random", covariates=["B"], noncentered=True),
            hboms.Parameter("sigma", 0.1, "fixed"),
        ]

        model = hboms.HbomsModel(
            name="cat_fixed_lev_test",
            params=params,
            odes=self.odes,
            init=self.init,
            state=self.state,
            obs=self.obs,
            dists=self.dists,
            covariates=self.covariates,
            model_dir=self.model_dir,
        )

        import numpy as np
        np.random.seed(12345)

        sims_pars = model.simulate(data=self.data, num_simulations=10)

        # fit the model
        sim_data, sim_par = sims_pars[0]
        model.sample(
            data = sim_data, step_size=0.01, chains=1, 
            iter_warmup=500, iter_sampling=500, 
            show_progress=False, seed=5678
        )

        # check that the parameters have the right shape

        a = model.fit.stan_variable("a")
        L = len(self.covariates[1].categories)  # covariate B defines the level for a
        assert a.shape == (500, L), f"expected {L} levels for parameter 'a' based on level 'B'"

        # check that loc of a has the right shape 

        loc_a = model.fit.stan_variable("loc_a")
        K = len(self.covariates[0].categories)  # covariate A defines the categories for a
        assert loc_a.shape == (500, K), f"expected loc of 'a' to have {K} components for 'A'"


        # check that b has the right shape too

        b = model.fit.stan_variable("b")
        assert b.shape == (500, self.R), f"expected {self.R} individuals for parameter 'b' based on R={self.R}"


        # check that a is within reasonable bounds
        a_gt = sim_par["a"]
        a_hat = np.median(a, axis=0)
        slope = sts.linregress(a_gt, a_hat).slope
        assert slope > 0.5, "estimated 'a' values do not correspond to ground truth"


    def test_catcov_fixed_level_nc(self):
        params = [
            hboms.Parameter("a", 0.5, "random", scale=0.2, covariates=["A"], 
                            level="B", level_type="fixed", noncentered=True),
            hboms.Parameter("b", 1.0, "random", covariates=["B"], noncentered=True),
            hboms.Parameter("sigma", 0.1, "fixed"),
        ]

        model = hboms.HbomsModel(
            name="cat_fixed_lev_test_nc",
            params=params,
            odes=self.odes,
            init=self.init,
            state=self.state,
            obs=self.obs,
            dists=self.dists,
            covariates=self.covariates,
            model_dir=self.model_dir,
        )

        import numpy as np
        np.random.seed(12345)

        sims_pars = model.simulate(data=self.data, num_simulations=10)

        # fit the model
        sim_data, sim_par = sims_pars[0]
        model.sample(
            data = sim_data, step_size=0.01, chains=1, 
            iter_warmup=500, iter_sampling=500, 
            show_progress=False, seed=5678
        )

        # check that the parameters have the right shape
        a = model.fit.stan_variable("a")
        L = len(self.covariates[1].categories)  # covariate B defines the level for a
        assert a.shape == (500, L), f"expected {L} levels for parameter 'a' based on level 'B'"

        # check that loc of a has the right shape 
        loc_a = model.fit.stan_variable("loc_a")
        K = len(self.covariates[0].categories)  # covariate A defines the categories for a
        assert loc_a.shape == (500, K), f"expected loc of 'a' to have {K} components for 'A'"

        # check that we estimated "random effects" for a 

        assert "rand_a" in model.fit.stan_variables(), "expected random effects for parameter 'a'"

        # check that rand_a has the right shape
        rand_a = model.fit.stan_variable("rand_a")
        L = len(self.covariates[1].categories)  # covariate B defines the level for a
        assert rand_a.shape == (500, L), f"expected {L} levels for random effects of 'a' based on level 'B'"
        
    def test_catcov_fixed_level_error(self):
        """
        try to sample from a model with a categorical covariate
        and a fixed level that are not compatible.
        This should raise an error.
        """
        params = [
            hboms.Parameter("a", 0.5, "random", scale=0.2, covariates=["A"], level="B", level_type="fixed"),
            hboms.Parameter("sigma", 0.1, "fixed"),
        ]

        model = hboms.HbomsModel(
            name="cat_fixed_lev_test_error",
            params=params,
            odes=self.odes,
            init=self.init,
            state=self.state,
            obs=self.obs,
            dists=self.dists,
            covariates=self.covariates,
            model_dir=self.model_dir,
        )

        import numpy as np
        np.random.seed(12345)

        R = 10
        err_data = {
            "Time" : [np.linspace(1, 20, 20) for _ in range(R)],
            "A" : ["A1", "A1", "A1", "A2", "A2", "A3", "A3", "A3", "A3", "A3"],
            "B" : ["B1", "B1", "B2", "B1", "B3", "B2", "B4", "B5", "B6", "B6"],
        }
        with pytest.raises(ValueError):
            sims_pars = model.simulate(data=err_data, num_simulations=10)


        # simulate data using correct level definition
        sims_pars = model.simulate(data=self.data, num_simulations=10)

        # fit the model with incorrect level definition

        sim_data, sim_par = sims_pars[0]
        with pytest.raises(ValueError):
            sim_data["B"] = err_data["B"]  # incompatible level definition
            model.sample(
                data = sim_data, step_size=0.01, chains=1, 
                iter_warmup=500, iter_sampling=500, 
                show_progress=False, seed=5678
            )

