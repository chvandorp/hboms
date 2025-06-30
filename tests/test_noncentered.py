import hboms
import numpy as np
import scipy.stats as sts

class TestNonCentered:
    def test_concentered(self):
        """
        make sure that the non-centered parameterization is equivalent to the centered one.
        The two models would have different lp__ values, but the posteriors should be the same.
        We use the Kolmogorov-Smirnov test to check if the posterior distributions
        of the parameters are the same for both models.
        """

        # set fixed seed for numpy
        np.random.seed(4563)


        params1 = [
            hboms.Parameter(
                "a", 0.0, "random", lbound=None, scale=0.1,
                loc_prior=hboms.StanPrior("std_normal", []),
                scale_prior=hboms.StanPrior("exponential", ["1.0"])
            )
        ]

        params2 = [
            hboms.Parameter(
                "a", 0.0, "random", lbound=None, scale=0.1,
                loc_prior=hboms.StanPrior("std_normal", []),
                scale_prior=hboms.StanPrior("exponential", ["1.0"]),
                noncentered=True
            )
        ]

        state = [hboms.Variable("x")]
        obs   = [hboms.Observation("X")]
        odes  = "ddt_x = a;"
        init  = "x_0 = 0.0;"
        dists = [hboms.StanDist("normal", "X", ["x", "1.0"])]

        # model with centered parameterization
        model1 = hboms.HbomsModel(
            name = "nc_test_model_centered",
            params = params1,
            state = state,
            obs = obs,
            odes = odes,
            init = init,
            dists = dists,
            model_dir="tests/stan-cache",
        ) 

        # model with non-centered parameterization
        model2 = hboms.HbomsModel(
            name = "nc_test_model_non_centered",
            params = params2,
            state = state,
            obs = obs,
            odes = odes,
            init = init,
            dists = dists,
            model_dir="tests/stan-cache",
        )

        num_units = 20

        # generate data for the centered model
        ts = np.arange(1, 10, 0.5)
        data = {
            "Time" : [ts for _ in range(num_units)]
        }
        sim_datas = model1.simulate(
            data=data, num_simulations=3, seed=567567
        )

        for sim_data, _ in sim_datas:
            kwargs = dict(data=sim_data, iter_warmup=1000, iter_sampling=2000, chains=1, seed=1274, step_size=0.01, adapt_delta=0.95)

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
            assert pvals_comb.pvalue > 0.05, "too many pairwise KS tests failed"

            res = sts.kstest(loc_a1, loc_a2)
            assert res.pvalue > 0.05, "locations do not have the same distributrion"

            res = sts.kstest(scale_a1, scale_a2)
            assert res.pvalue > 0.05, "scales do not have the same distributrion"
