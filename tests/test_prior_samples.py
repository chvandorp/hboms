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
        assert np.abs(np.mean(sample_a) - self.a_mean_gt) < 0.1, err_msg
        err_msg = "Std of prior samples for 'a' is not close to {}".format(self.a_std_gt)
        assert np.abs(np.std(sample_a) - self.a_std_gt) < 0.1, err_msg


    def test_prior_samples_generation_logitnormal(self):
        # Test that prior samples are generated correctly for logitnormal prior
        # This requires a custom RNG in Stan

        lbound = -1.0
        ubound = 2.5

        # modify parameter prior to logitnormal
        params = self.params.copy() # make a copy to avoid modifying other tests
        params[0] = hboms.Parameter(
            "a", 0.0, "fixed", lbound=lbound, ubound=ubound,
            prior=hboms.StanPrior("logitnormal", [self.a_mean_gt, self.a_std_gt, lbound, ubound])
        )

        # compile the model
        model = hboms.HbomsModel(
            name = "test_prior_samples_model_logitnormal",
            state = [], odes = "", init = "",
            params = params,
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

        # make sure that the samples are within bounds
        assert np.all(sample_a >= lbound), "Some prior samples for 'a' are below lower bound"
        assert np.all(sample_a <= ubound), "Some prior samples for 'a' are above upper bound"




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
        assert np.abs(np.mean(sample_a) - np.mean(prior_samples)) < 0.1, err_msg
        err_msg = "Std of prior samples for 'a' is not close to prior std"
        assert np.abs(np.std(sample_a) - np.std(prior_samples)) < 0.1, err_msg

        # check distance with KS test
        ks_stat, ks_pvalue = sts.ks_2samp(sample_a, prior_samples)
        err_msg = "KS test p-value is too low, prior samples do not match prior distribution"
        assert ks_pvalue > 0.02, err_msg




    def test_prior_samples_generation_catcov(self):
        # Test that prior samples are generated correctly with categorical covariates

        covars = [hboms.Covariate("group", cov_type="cat", categories=["A", "B"])]

        params = [
            hboms.Parameter(
                "a", 0.0, "random", lbound=None, 
                loc_prior=hboms.StanPrior("normal", [self.loc_a_mean_gt, self.loc_a_std_gt]),
                scale_prior=hboms.StanPrior("exponential", [self.rate_a_gt]),
                covariates = ["group"],
            ),
            hboms.Parameter("sigma", 0.5, "const")
        ]
        

        # compile the model
        model = hboms.HbomsModel(
            name = "test_prior_samples_model_random_catcov",
            state = [], odes = "", init = "",
            params = params, # changed to params with covariate
            obs = self.obs, dists = self.dists,
            trans_state = self.trans_state,
            transform = self.transform,
            covariates = covars, # added covariates
            options = {"prior_samples": True},
            model_dir=self.model_dir,
        )

        ## add group covariate to data
        data_with_covars = self.data.copy()
        num_units = len(data_with_covars["X"])
        data_with_covars["group"] = (["A", "B"] * num_units)[:num_units]

        # fit the model
        model.sample(
            data=data_with_covars,
            chains=1,
            show_progress=False,
            iter_warmup=1000,
            iter_sampling=1000,
            seed=4562,
        )

        # make sure prior samples were generated
        assert "prior_sample_a" in model.fit.stan_variables(), "Prior samples for 'a' not found"

        # check the results:
        
        sample_a = model.fit.stan_variable("prior_sample_a")

        # check that the shape is correct

        assert len(sample_a.shape) == 1, "Currently, cat covars are not considered in prior samples"




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

        def validate_random_distributions(loc_mean_gt, loc_std_gt, rate_gt, sample, name):
            locs = sts.norm.rvs(loc=loc_mean_gt, scale=loc_std_gt, size=sample.shape[0])
            scales = sts.expon.rvs(scale=1.0/rate_gt, size=sample.shape[0])
            prior_samples = sts.norm.rvs(loc=locs, scale=scales)

            err_msg = f"Mean of prior samples for '{name}' is not close to prior mean"
            assert np.abs(np.mean(sample) - np.mean(prior_samples)) < 0.1, err_msg
            err_msg = f"Std of prior samples for '{name}' is not close to prior std"
            assert np.abs(np.std(sample) - np.std(prior_samples)) < 0.1, err_msg

            # check distance with KS test
            ks_stat, ks_pvalue = sts.ks_2samp(sample, prior_samples)
            err_msg = f"KS test p-value for '{name}' is too low, prior samples do not match prior distribution"
            assert ks_pvalue > 0.02, err_msg

        validate_random_distributions(self.loc_a_mean_gt, self.loc_a_std_gt, self.rate_a_gt, sample_a, "a")
        validate_random_distributions(self.loc_b_mean_gt, self.loc_b_std_gt, self.rate_b_gt, sample_b, "b")



class TestPriorSamplerModel():
    @pytest.fixture(autouse=True) 
    def _setup(self):
        self.model_dir = os.path.join("tests", "stan-cache")

        self.mean_a_gt = 0.4
        self.std_a_gt = 1.2

        self.params = [
            hboms.Parameter("a", 0.0, "fixed", lbound=None, 
                            prior=hboms.StanPrior("normal", [self.mean_a_gt, self.std_a_gt])),
            hboms.Parameter("sigma", 0.5, "const"),
        ]
        self.obs = [hboms.Observation("X")]
        self.dists = [hboms.StanDist("normal", "X", ["x", "sigma"])]
        self.state = [hboms.Variable("x")]
        self.odes = "ddt_x = a;"
        self.init = "x_0 = 0.0;"

        # generate some data
        # it does not matter what data we use here 
        # since we are only testing prior sampling
        np.random.seed(seed=3454)
        ts = np.ones(1)
        self.num_units = 5
        self.x_gt = 3.0
        self.sigma_gt = 0.5
        X = [
            sts.norm.rvs(loc=self.x_gt, scale=self.sigma_gt, size=ts.shape)
            for r in range(self.num_units)
        ]
        self.data = {
            "X": X,
            "Time": [ts for r in range(self.num_units)],
        }

    def test_prior_sampler(self):
        # compile the model
        model = hboms.HbomsModel(
            name = "test_prior_sampler_model",
            state = self.state, odes = self.odes, init = self.init,
            params = self.params,
            obs = self.obs, dists = self.dists,
            model_dir=self.model_dir,
            compile_model=False, # we do not need the full model compiled.
        )

        # check that model contains a field for prior sampler code
        assert hasattr(model, "_prior_sampler_code"), "Model does not have _prior_sampler_code attribute"
        msg = "_prior_sampler_code attribute should be None before sample_from_prior is called"
        assert model._prior_sampler_code is None, msg

        # sample from the prior
        model.sample_from_prior(
            self.data,
            chains=1,
            show_progress=False,
            iter_warmup=1000,
            iter_sampling=2000,
            seed=7890,
        )

        # check that model contains a field for prior fit
        assert hasattr(model, "_prior_fit"), "Model does not have _prior_fit attribute"
        assert model.prior_fit is not None, "Model _prior_fit attribute is None after sampling from prior"

        # check that prior fit contains samples for prior_sample_a
        assert "a" in model._prior_fit.stan_variables(), "Prior fit does not contain samples for 'a'"

        # check that a has the correct distribution

        sample_a = model.prior_fit.stan_variable("a")

        err_msg = "Mean of prior samples for 'a' is not close to 0.4"
        assert np.abs(np.mean(sample_a) - self.mean_a_gt) < 0.1, err_msg
        err_msg = "Std of prior samples for 'a' is not close to 1.2"
        assert np.abs(np.std(sample_a) - self.std_a_gt) < 0.1, err_msg

        # check distance with KS test
        gt_prior_dist = sts.norm(loc=self.mean_a_gt, scale=self.std_a_gt).cdf
        ks_stat, ks_pvalue = sts.kstest(sample_a, gt_prior_dist)
        err_msg = "KS test p-value is too low, prior samples do not match prior distribution"
        assert ks_pvalue > 0.02, err_msg


    def test_prior_sampler_random(self):
        """
        Test that sampling from prior works with random parameters.
        """

        sd_a_gt = 3.0
        # modify parameter to be random
        params = [
            hboms.Parameter(
                "a", 0.0, "random", lbound=None,
                loc_prior=hboms.StanPrior("normal", [self.mean_a_gt, self.std_a_gt]),
                scale_prior=hboms.StanPrior("exponential", [sd_a_gt])
            ),
            hboms.Parameter("sigma", 0.5, "const"),
        ]

        # compile the model
        model = hboms.HbomsModel(
            name = "test_prior_sampler_model_random",
            state = self.state, odes = self.odes, init = self.init,
            params = params,
            obs = self.obs, dists = self.dists,
            model_dir=self.model_dir,
            compile_model=False, # we do not need the full model compiled.
        )

        num_samples = 2000

        # sample from the prior
        model.sample_from_prior(
            self.data,
            chains=1,
            show_progress=False,
            iter_warmup=1000,
            iter_sampling=num_samples,
            seed=7890,
        )

        # check that prior fit contains samples for prior_sample_a
        assert "a" in model.prior_fit.stan_variables(), "Prior fit does not contain samples for 'a'"

        # check that a has the correct shape 

        sample_a = model.prior_fit.stan_variable("a")
        assert sample_a.shape == (num_samples, self.num_units), "Prior samples for 'a' have incorrect shape"
        
        # check that a has the correct distribution

        sample_a = sample_a[:,0]  # take only the first unit for testing

        locs = sts.norm.rvs(loc=self.mean_a_gt, scale=self.std_a_gt, size=sample_a.shape[0])
        scales = sts.expon.rvs(scale=1/sd_a_gt, size=sample_a.shape[0])
        prior_samples = sts.norm.rvs(loc=locs, scale=scales)

        err_msg = "Mean of prior samples for 'a' is not close to prior mean"
        assert np.abs(np.mean(sample_a) - np.mean(prior_samples)) < 0.1, err_msg
        err_msg = "Std of prior samples for 'a' is not close to prior std"
        assert np.abs(np.std(sample_a) - np.std(prior_samples)) < 0.1, err_msg

        # check distance with KS test
        ks_stat, ks_pvalue = sts.ks_2samp(sample_a, prior_samples)
        err_msg = "KS test p-value is too low, prior samples do not match prior distribution"
        assert ks_pvalue > 0.02, err_msg









