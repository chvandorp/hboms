

class TestCatOrder:
    def test_cat_order(self):
        from hboms.covariate import CatCovariate
        from hboms.model_helper_funcs import map_cat_covars

        # deliberately define categories in non-alphabetical order
        cov = CatCovariate("mycov", categories=["cat3", "cat2", "cat1"])
        
        data = {"mycov": ["cat2", "cat1", "cat3", "cat1"]}
        mapped_cat_covariates, num_cats = map_cat_covars(data, [cov])

        assert mapped_cat_covariates["mycov"].tolist() == [2, 3, 1, 3], "Categories should be mapped according to the order defined in CatCovariate"

        assert num_cats["K_mycov"] == 3, "Number of categories should be correctly identified as 3"

    
    def test_cat_order_model(self):
        import scipy.stats as sts
        import numpy as np
        import os
        import hboms 

        np.random.seed(144169)

        covar = ["cat3", "cat2", "cat1", "cat1", "cat2", "cat3", "cat2", "cat1", "cat3"]
        cats = ["cat3", "cat2", "cat1"] # order of categories in the model
        cov = hboms.Covariate("X", "cat", categories=cats)
        R = len(covar)
        N = 5
        Time = [[0.0 for _ in range(N)] for _ in range(R)]
        mu_gt = [0.0, 1.0, 2.0] # arbitrary values for each category
        Y = [(mu_gt[cats.index(cat)] + sts.norm.rvs(0, 0.1, size=N)).tolist() for cat in covar]

        data = {
            "X": covar,
            "Time": Time,
            "Y": Y            
        }

        params = [
            hboms.Parameter("mu", 1.0, "random", covariates=["X"], lbound=None,
                            loc_prior=hboms.StanPrior("normal", [1.0, 1.0]),
                            scale_prior=hboms.StanPrior("normal", [0.0, 0.5]))
        ]

        model = hboms.HbomsModel(
            "cat_order_model",
            state=[],
            odes="",
            init="",
            params=params,
            obs=[hboms.Observation("Y")],
            dists=[hboms.StanDist("normal", "Y", ["mu", "0.1"])],
            trans_state=[hboms.Variable("x")],
            transform="x = mu;",
            covariates=[cov],
            model_dir=os.path.join("tests", "stan-cache")
        )

        model.sample(data, chains=1, seed=144169, iter_warmup=200, iter_sampling=200, adapt_delta=0.95)

        mu_hat = model.fit.stan_variable("mu")
        # Check that the estimated mu values are close to the true values for each category

        for i, cat in enumerate(cats):
            mui = mu_hat[:, i]
            meanmui = np.mean(mui)
            assert abs(meanmui - mu_gt[i]) < 0.1, f"Estimated mu for {cat} should be close to true value, got {meanmui} vs {mu_gt[i]}"
