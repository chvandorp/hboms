import hboms
import os
import numpy as np
import scipy.stats as sts
import pytest



class TestFixedPriorSamples:
    @pytest.fixture(autouse=True) 
    def _setup(self):
        self.model_dir = os.path.join("tests", "stan-cache")

        self.a_mean_gt = 2.0
        self.a_std_gt = 1.0

        self.params = [
            hboms.Parameter("a", 0.0, "fixed", lbound=None, 
                            prior=hboms.StanPrior("normal", [self.a_mean_gt, self.a_std_gt])),
            hboms.Parameter("sigma", 0.5, "const"),
        ]
        self.obs = [hboms.Observation("X")]
        self.dists = [hboms.StanDist("normal", "X", ["x", "sigma"])]
        self.trans_state = [hboms.Variable("x")]
        self.transform = "x = a;"
    
        # create some data
        np.random.seed(seed=3454)
        ts = np.ones(1)
        num_units = 5
        self.x_gt = 3.0
        self.sigma_gt = 0.5
        X = [
            sts.norm.rvs(loc=self.x_gt, scale=self.sigma_gt, size=ts.shape)
            for r in range(num_units) 
        ]

        self.data = {
            "X": X,
            "Time": [ts for r in range(num_units)],
        }


    def test_no_prior_samples_generation(self):
        # Test that prior samples skipped when option is False (default)

        # compile the model
        model = hboms.HbomsModel(
            name = "test_no_prior_samples_model_fixed",
            state = [], odes = "", init = "",
            params = self.params,
            obs = self.obs, dists = self.dists,
            trans_state = self.trans_state,
            transform = self.transform,
            model_dir=self.model_dir,
        )

        # fit the model
        model.sample(
            data=self.data,
            chains=1,
            show_progress=False,
            iter_warmup=100,
            iter_sampling=100,
            seed=4562,
        )

        # make sure prior samples were NOT generated
        assert "prior_sample_a" not in model.fit.stan_variables(), "Prior samples for 'a' found when they should not be"



    def test_prior_samples_generation(self):
        # Test that prior samples are generated correctly

        # compile the model
        model = hboms.HbomsModel(
            name = "test_prior_samples_model_fixed",
            state = [], odes = "", init = "",
            params = self.params,
            obs = self.obs, dists = self.dists,
            trans_state = self.trans_state,
            transform = self.transform,
            options = {"prior_samples": True},
            model_dir=self.model_dir,
        )

        # fit the model
        model.sample(
            data=self.data,
            chains=1,
            show_progress=False,
            iter_warmup=1000,
            iter_sampling=1000,
            seed=4562,
        )

        # make sure prior samples were generated
        assert "prior_sample_a" in model.fit.stan_variables(), "Prior samples for 'a' not found"

        # check the results
        
        sample_a = model.fit.stan_variable("prior_sample_a")

        # check that the sample has a similar distribution to the prior
        err_msg = "Mean of prior samples for 'a' is not close to {}".format(self.a_mean_gt)
        assert np.abs(np.mean(sample_a) - self.a_mean_gt) < 0.025, err_msg
        err_msg = "Std of prior samples for 'a' is not close to {}".format(self.a_std_gt)
        assert np.abs(np.std(sample_a) - self.a_std_gt) < 0.05, err_msg




class TestRandomPriorSamples:
    @pytest.fixture(autouse=True) 
    def _setup(self):
        self.model_dir = os.path.join("tests", "stan-cache")

        self.loc_a_mean_gt = 2.0
        self.loc_a_std_gt = 1.0
        self.rate_a_gt = 3.0

        self.params = [
            hboms.Parameter(
                "a", 0.0, "random", lbound=None, 
                loc_prior=hboms.StanPrior("normal", [self.loc_a_mean_gt, self.loc_a_std_gt]),
                scale_prior=hboms.StanPrior("exponential", [self.rate_a_gt])
            ),
            hboms.Parameter("sigma", 0.5, "const"),
        ]
        self.obs = [hboms.Observation("X")]
        self.dists = [hboms.StanDist("normal", "X", ["x", "sigma"])]
        self.trans_state = [hboms.Variable("x")]
        self.transform = "x = a;"
    
        # create some data
        np.random.seed(seed=3454)
        ts = np.ones(1)
        num_units = 5
        self.x_gt = 3.0
        self.sigma_gt = 0.5
        X = [
            sts.norm.rvs(loc=self.x_gt, scale=self.sigma_gt, size=ts.shape)
            for r in range(num_units) 
        ]

        self.data = {
            "X": X,
            "Time": [ts for r in range(num_units)],
        }


    def test_prior_samples_generation(self):
        # Test that prior samples are generated correctly

        # compile the model
        model = hboms.HbomsModel(
            name = "test_prior_samples_model_random",
            state = [], odes = "", init = "",
            params = self.params,
            obs = self.obs, dists = self.dists,
            trans_state = self.trans_state,
            transform = self.transform,
            options = {"prior_samples": True},
            model_dir=self.model_dir,
        )

        # fit the model
        model.sample(
            data=self.data,
            chains=1,
            show_progress=False,
            iter_warmup=1000,
            iter_sampling=1000,
            seed=4562,
        )

        # make sure prior samples were generated
        assert "prior_sample_a" in model.fit.stan_variables(), "Prior samples for 'a' not found"

        # check the results
        
        sample_a = model.fit.stan_variable("prior_sample_a")

        # check that the sample has a similar distribution to the prior

        locs = sts.norm.rvs(loc=self.loc_a_mean_gt, scale=self.loc_a_std_gt, size=sample_a.shape[0])
        scales = sts.expon.rvs(scale=1.0/self.rate_a_gt, size=sample_a.shape[0])
        prior_samples = sts.norm.rvs(loc=locs, scale=scales)

        err_msg = "Mean of prior samples for 'a' is not close to prior mean"
        assert np.abs(np.mean(sample_a) - np.mean(prior_samples)) < 0.05, err_msg
        err_msg = "Std of prior samples for 'a' is not close to prior std"
        assert np.abs(np.std(sample_a) - np.std(prior_samples)) < 0.05, err_msg

        # check distance with KS test
        ks_stat, ks_pvalue = sts.ks_2samp(sample_a, prior_samples)
        err_msg = "KS test p-value is too low, prior samples do not match prior distribution"
        assert ks_pvalue > 0.1, err_msg




class TestCorrelatedPriorSamples:
    @pytest.fixture(autouse=True) 
    def _setup(self):
        self.model_dir = os.path.join("tests", "stan-cache")

        self.loc_a_mean_gt = 2.0
        self.loc_a_std_gt = 1.0
        self.rate_a_gt = 3.0

        self.loc_b_mean_gt = 1.0
        self.loc_b_std_gt = 0.5
        self.rate_b_gt = 2.0


        self.params = [
            hboms.Parameter(
                "a", 0.0, "random", lbound=None, 
                loc_prior=hboms.StanPrior("normal", [self.loc_a_mean_gt, self.loc_a_std_gt]),
                scale_prior=hboms.StanPrior("exponential", [self.rate_a_gt])
            ),
            hboms.Parameter(
                "b", 0.0, "random", lbound=None, 
                loc_prior=hboms.StanPrior("normal", [self.loc_b_mean_gt, self.loc_b_std_gt]),
                scale_prior=hboms.StanPrior("exponential", [self.rate_b_gt])
            ),
            hboms.Parameter("sigma", 0.5, "const"),
        ]
        self.obs = [hboms.Observation("X")]
        self.dists = [hboms.StanDist("normal", "X", ["x", "sigma"])]
        self.trans_state = [hboms.Variable("x")]
        self.transform = "x = a;"
        self.correlations = [hboms.Correlation(["a", "b"])]

        # create some data
        np.random.seed(seed=3454)
        ts = np.ones(1)
        num_units = 5
        self.x_gt = 3.0
        self.sigma_gt = 0.5
        X = [
            sts.norm.rvs(loc=self.x_gt, scale=self.sigma_gt, size=ts.shape)
            for r in range(num_units) 
        ]

        self.data = {
            "X": X,
            "Time": [ts for r in range(num_units)],
        }


    def test_prior_samples_generation(self):
        # Test that prior samples are generated correctly

        # compile the model
        model = hboms.HbomsModel(
            name = "test_prior_samples_model_correlated",
            state = [], odes = "", init = "",
            params = self.params,
            correlations = self.correlations,
            obs = self.obs, dists = self.dists,
            trans_state = self.trans_state,
            transform = self.transform,
            options = {"prior_samples": True},
            model_dir=self.model_dir,
        )

        # fit the model
        model.sample(
            data=self.data,
            chains=1,
            show_progress=False,
            iter_warmup=1000,
            iter_sampling=1000,
            seed=4562,
        )

        # make sure prior samples were generated
        assert "prior_sample_a" in model.fit.stan_variables(), "Prior samples for 'a' not found"
        assert "prior_sample_b" in model.fit.stan_variables(), "Prior samples for 'b' not found"

        # check the results
        
        sample_a = model.fit.stan_variable("prior_sample_a")
        sample_b = model.fit.stan_variable("prior_sample_b")
        # check that the sample has a similar distribution to the prior

        # TODO: implement correlated prior sampling test