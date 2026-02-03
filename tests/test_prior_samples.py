import hboms
import os
import numpy as np
import scipy.stats as sts
import pytest




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









