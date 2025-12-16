import pytest
import hboms
import numpy as np
import scipy.stats as sts
import os


def compile_model(params, correlations):
    model = hboms.HbomsModel(
        name = "test_model_non_centered_init",
        params = params,
        state = [hboms.Variable("x")],
        obs = [hboms.Observation("X")],
        odes = "ddt_x = a;",
        init = "x_0 = 0.0;",
        dists = [hboms.StanDist("normal", "X", ["x", "1.0"])],
        correlations=correlations,
        model_dir="tests/stan-cache",
        compile_model=False,
    )
    return model




class TestNonCentered:
    @pytest.fixture(autouse=True) 
    def _setup(self):
        self.model_dir = os.path.join("tests", "stan-cache")
        # set fixed seed for numpy
        np.random.seed(5673)
        

    def test_noncentered(self):
        """
        make sure that the non-centered parameterization is equivalent to the centered one.
        The two models would have different lp__ values, but the posteriors should be the same.
        We use the Kolmogorov-Smirnov test to check if the posterior distributions
        of the parameters are the same for both models.
        """

        params1 = [
            hboms.Parameter(
                "a", 0.0, "random", lbound=None, scale=0.1,
                loc_prior=hboms.StanPrior("normal", [0.0, 0.1]),
                scale_prior=hboms.StanPrior("exponential", [1.0])
            )
        ]

        params2 = [
            hboms.Parameter(
                "a", 0.0, "random", lbound=None, scale=0.1,
                loc_prior=hboms.StanPrior("normal", [0.0, 0.1]),
                scale_prior=hboms.StanPrior("exponential", [1.0]),
                noncentered=True
            )
        ]

        state = []
        trans_state = [hboms.Variable("x")]
        obs   = [hboms.Observation("X")]
        odes  = "" # ddt_x = a; solved in transform for faster sampling
        trans = "x = a*t;"
        init  = ""
        dists = [hboms.StanDist("normal", "X", ["x", "1.0"])]

        # model with centered parameterization
        model1 = hboms.HbomsModel(
            name = "nc_test_model_centered",
            params = params1,
            state = state,
            trans_state = trans_state,
            obs = obs,
            odes = odes,
            transform = trans,
            init = init,
            dists = dists,
            model_dir=self.model_dir,
        ) 

        # model with non-centered parameterization
        model2 = hboms.HbomsModel(
            name = "nc_test_model_non_centered",
            params = params2,
            state = state,
            trans_state = trans_state,
            obs = obs,
            odes = odes,
            transform = trans,
            init = init,
            dists = dists,
            model_dir=self.model_dir,
        )

        num_units = 20

        # generate data for the centered model
        ts = np.arange(1, 10, 0.25)
        data = {
            "Time" : [ts for _ in range(num_units)]
        }
        sim_datas = model1.simulate(
            data=data, num_simulations=1, seed=576567
        )

        for sim_data, _ in sim_datas:
            kwargs = dict(
                data=sim_data, 
                iter_warmup=1000, iter_sampling=1000, 
                chains=1, seed=67457, thin=1,
                step_size=0.01, adapt_delta=0.95,
                show_progress=False
            )

            model1.sample(**kwargs)
            model2.sample(**kwargs)

            loc_a1 = model1.fit.stan_variable("loc_a")
            loc_a2 = model2.fit.stan_variable("loc_a")

            scale_a1 = model1.fit.stan_variable("scale_a")
            scale_a2 = model2.fit.stan_variable("scale_a")

            pvals = []
            for i in range(num_units):
                a_hat1 = model1.fit.stan_variable("a")[:,i]
                a_hat2 = model2.fit.stan_variable("a")[:,i]
                res = sts.kstest(a_hat1, a_hat2)    
                pvals.append(res.pvalue)

            pvals_comb = sts.combine_pvalues(pvals)
            assert pvals_comb.pvalue > 0.02, "too many pairwise KS tests failed"

            res = sts.kstest(loc_a1, loc_a2)
            assert res.pvalue > 0.02, "locations do not have the same distributrion"

            res = sts.kstest(scale_a1, scale_a2)
            assert res.pvalue > 0.02, "scales do not have the same distributrion"


    def test_init(self):
        """
        Test that the initial values for non-centered parameters are set to the right values.
        For non-centered parameters, the initial values should be set to zero if
        the user provides a single initial value.

        If the user provides a list of initial values, these values should be centered
        and they should be scaled with the (optional) scale value.
        """

        num_units = 3
        ts = np.arange(1, 10, 0.5)
        data = {
            "Time": [ts for _ in range(num_units)],
            "X" : [np.full(ts.shape, fill_value=0.0) for _ in range(num_units)]
        }


        # Test with a single initial value
        a_guess = 1.0
        params = [
            hboms.Parameter(
                "a", a_guess, "random", lbound=None, scale=0.1,
                noncentered=True
            )
        ]

        model = compile_model(params, correlations=None)
        
        data_dict, init_dict = model.stan_data_and_init(data=data, n_sim=10)

        zscores = [0.0] * num_units  # since a_guess is a single value, all z-scores should be zero
        assert np.all(init_dict["rand_a"] == zscores), "Initial values for non-centered parameters should be centered at zero"
        assert np.all(init_dict["scale_a"] == 0.1), "Scale values for non-centered parameters not set correctly"
        assert np.all(init_dict["loc_a"] == a_guess), "Location values for non-centered parameters not set correctly"



    def test_init_list(self):
        """
        Test with a list of initial values.
        Here we provide a list of initial values that are not zero
        and we expect the initial values to be centered and scaled
        """

        num_units = 3
        ts = np.arange(1, 10, 0.5)
        data = {
            "Time": [ts for _ in range(num_units)],
            "X" : [np.full(ts.shape, fill_value=0.0) for _ in range(num_units)]
        }

        a_guess = [1.0, 2.0, 3.0]
        a_scale_guess = 0.1
        params = [
            hboms.Parameter(
                "a", a_guess, "random", lbound=None, scale=a_scale_guess,
                noncentered=True
            )
        ]
        model = compile_model(params, correlations=None)
        data_dict, init_dict = model.stan_data_and_init(data=data, n_sim=10)

        zscores = (np.array(a_guess) - np.mean(a_guess)) / a_scale_guess
        assert np.all(init_dict["rand_a"] == zscores), "Initial values for non-centered parameters should be centered at zero"
        assert np.all(init_dict["loc_a"] == np.mean(a_guess)), "Location values for non-centered parameters not set correctly"
        assert np.all(init_dict["scale_a"] == a_scale_guess), "Scale values for non-centered parameters not set correctly"  


    def test_init_with_correlations(self):
        """
        Test with a list of initial values and correlations
        """

        num_units = 3
        ts = np.arange(1, 10, 0.5)
        data = {
            "Time": [ts for _ in range(num_units)],
            "X" : [np.full(ts.shape, fill_value=0.0) for _ in range(num_units)]
        }

        a_guess = [1.0, 2.0, 3.0]
        a_scale_guess = 0.1
        b_guess = 5.0
        params = [
            hboms.Parameter(
                "a", a_guess, "random", lbound=None, scale=a_scale_guess,
                noncentered=True
            ),
            hboms.Parameter(
                "b", b_guess, "random", lbound=0.0,
            )
        ]

        correlations = [
            hboms.Correlation(["a", "b"])
        ]

        model = compile_model(params, correlations=correlations)

        data_dict, init_dict = model.stan_data_and_init(data=data, n_sim=10)

        init_a = (np.array(a_guess) - np.mean(a_guess)) / a_scale_guess
        init_b = np.log(np.full((num_units,), b_guess)) ## zero lower bound for b, so we use log transformation
        init_a_b = np.stack((init_a, init_b), axis=1)

        assert np.all(init_dict["block_a_b"] == init_a_b), "parameter block initial guess is not correct"


    def test_init_with_bounds(self):
        """
        Noncentered parameter with lower bound
        """
        num_units = 3
        ts = np.arange(1, 10, 0.5)
        data = {
            "Time": [ts for _ in range(num_units)],
            "X" : [np.full(ts.shape, fill_value=0.0) for _ in range(num_units)]
        }

        a_guess = [1.0, 2.0, 3.0]
        a_scale_guess = 0.1
        b_guess = 5.0
        a_lbound = 0.5
        b_ubound = 8.0
        params = [
            hboms.Parameter(
                "a", a_guess, "random", lbound=a_lbound, scale=a_scale_guess,
                noncentered=True
            ),
            hboms.Parameter(
                "b", b_guess, "random", ubound=b_ubound, lbound=None
            )
        ]

        correlations = [
            hboms.Correlation(["a", "b"])
        ]

        model = compile_model(params, correlations=correlations)

        data_dict, init_dict = model.stan_data_and_init(data=data, n_sim=10)

        a_guess_unc = np.log(np.array(a_guess) - a_lbound)
        init_a = (np.array(a_guess_unc) - np.mean(a_guess_unc)) / a_scale_guess
        b_guess_unc = -np.log(b_ubound - b_guess)
        init_b = np.full((num_units,), b_guess_unc)
        init_a_b = np.stack((init_a, init_b), axis=1)

        assert np.all(init_dict["block_a_b"] == init_a_b), "parameter block initial guess is not correct"



    def test_noncentered_correlations(self):
        parameters1 = [
            hboms.Parameter("a", 0.0, "random", scale=1.0, lbound=None, noncentered=False, 
                            loc_prior=hboms.StanPrior("normal", ["0.0", "1.0"])),
            hboms.Parameter("b", 0.0, "random", scale=0.1, lbound=None,
                            scale_prior=hboms.StanPrior("exponential", ["1.0"])),
            hboms.Parameter("sigma", 0.5, "const"),
        ]

        parameters2 = [
            hboms.Parameter("a", 0.0, "random", scale=1.0, lbound=None, noncentered=True, 
                            loc_prior=hboms.StanPrior("normal", ["0.0", "1.0"])),
            hboms.Parameter("b", 0.0, "random", scale=0.1, lbound=None,
                            scale_prior=hboms.StanPrior("exponential", ["1.0"])),
            hboms.Parameter("sigma", 0.5, "const"),
        ]

        corr_gt = -0.8
        corr_mat_gt = (1-corr_gt) * np.eye(2) + corr_gt * np.ones((2,2))
        corrs = [hboms.Correlation(["a", "b"], value=corr_mat_gt)]
        odes = ""
        transform = "x = a*t + 0.5*b*t^2;" ## solved model: faster sampling with one core...
        inits = ""
        obs = [hboms.Observation("X")]
        dists = [hboms.StanDist("normal", "X", ["x", "sigma"])]
        state = []
        trans_state = [hboms.Variable("x")]


        model1 = hboms.HbomsModel(
            name = "nc-corr-c",
            odes=odes,
            init=inits,
            transform=transform,
            obs=obs,
            dists=dists,
            params=parameters1,
            correlations=corrs,
            state=state,
            trans_state=trans_state,
            model_dir=self.model_dir,
        )

        model2 = hboms.HbomsModel(
            name = "nc-corr-nc",
            odes=odes,
            init=inits,
            transform=transform,
            obs=obs,
            dists=dists,
            params=parameters2,
            correlations=corrs,
            state=state,
            trans_state=trans_state,
            model_dir=self.model_dir,
        )

        ## simulate data (does not matter which model we use)

        num_units = 20
        ts = np.arange(1, 10, 0.5)
        data = {"Time" : [ts for _ in range(num_units)]}
        sim_datas = model1.simulate(data=data, num_simulations=10, seed=3453)
        sim_data, sim_pars = sim_datas[0]


        model1.sample(
            data=sim_data, iter_warmup=200, iter_sampling=500, 
            step_size=0.01, chains=1, seed=7348
        )

        model2.sample(
            data=sim_data, iter_warmup=200, iter_sampling=500, 
            step_size=0.01, chains=1, seed=7348
        )

        corr_est1 = model1.fit.stan_variable("corr_a_b")[:,0,1]
        corr_est2 = model2.fit.stan_variable("corr_a_b")[:,0,1]

        res = sts.kstest(corr_est1, corr_est2)

        assert res.pvalue > 0.02, "correlations do not have the same distribution"

        c1l, c1u = np.percentile(corr_est1, [2.5, 97.5])
        c2l, c2u = np.percentile(corr_est2, [2.5, 97.5])
        
        assert c1l < 0.0 and c1u < 0.0, "correlation for centered model should be negative"
        assert c2l < 0.0 and c2u < 0.0, "correlation for non-centered model should be negative"

        assert c1l < corr_gt < c1u, "correlation for centered model should be close to ground truth"
        assert c2l < corr_gt < c2u, "correlation for non-centered model should be close to ground truth"


