import numpy as np
import pytest
import os

from hboms import HbomsModel
from hboms import Parameter, Covariate, Correlation
import hboms


class TestCovCorr:
    model_dir = os.path.join("tests", "stan-cache")

    def test_covcorr(self):
        """
        Test parameters with both correlations and covariates.

        In this test, we create a parameter block with three parameters:
        - Parameter 'a' depends on categorical covariate 'X1'
        - Parameter 'b' has no covariate dependence.
        - Parameter 'c' depends on categorical covariate 'X2' and is non-centered.
        """

        params = [
            Parameter("a", 1.0, "random", covariates=['X1']),
            Parameter("b", 1.0, "random"),
            Parameter("c", 1.0, "random", noncentered=True, covariates=['X2']),
        ]

        covs = [
            Covariate(name="X1", cov_type='cat', categories=['A1', 'B1']),
            Covariate(name="X2", cov_type='cat', categories=['A2', 'B2', 'C2']),
        ]
        corrs = [Correlation(params=['a', 'b', 'c'])]
        state = [hboms.Variable('y')]
        obs = [hboms.Observation("Y")]
        dists = [hboms.StanDist("normal", "Y", ["y", "0.1"])]

        # compile test...

        model = HbomsModel(
            "test_covcorr",
            state=state,
            odes="ddt_y = a * y + b;",
            init="y_0 = 0.0;",
            obs=obs,
            dists=dists,
            params=params, 
            covariates=covs, 
            correlations=corrs,
            model_dir=self.model_dir,
        )

    def test_covcorr_level(self):
        """
        Test parameters with both correlations and covariates with levels.
        
        In this test, we create a parameter block with three parameters:
        - Parameter 'a' depends on categorical covariate 'X1' and level 'L1'
        - Parameter 'b' has no covariate dependence and level 'L1'.
        - Parameter 'c' depends on categorical covariate 'X2', is non-centered, and has level 'L1'.
        """

        params = [
            Parameter("a", 1.0, "random", covariates=['X1'], level='L1'),
            Parameter("b", 1.0, "random", level='L1'),
            Parameter("c", 1.0, "random", noncentered=True, covariates=['X2'], level='L1'),
        ]

        covs = [
            Covariate(name="X1", cov_type='cat', categories=['A1', 'B1']),
            Covariate(name="X2", cov_type='cat', categories=['A2', 'B2', 'C2']),
            Covariate(name="L1", cov_type='cat', categories=['AL1', 'BL1']),
        ]
        corrs = [Correlation(params=['a', 'b', 'c'])]
        state = [hboms.Variable('y')]
        obs = [hboms.Observation("Y")]
        dists = [hboms.StanDist("normal", "Y", ["y", "0.1"])]

        # compile test...

        model = HbomsModel(
            "test_covcorr_level",
            state=state,
            odes="ddt_y = a * y + b;",
            init="y_0 = 0.0;",
            obs=obs,
            dists=dists,
            params=params, 
            covariates=covs, 
            correlations=corrs,
            model_dir=self.model_dir,
        )


    def test_covcorr_contcorr(self):
        """
        Test parameters with both correlations and covariates.
        Now with both categorical AND continuous covariates.
        """

        params = [
            Parameter("a", 1.0, "random", covariates=['X1']),
            Parameter("b", 1.0, "random", covariates=['C1']),
        ]

        covs = [
            Covariate(name="X1", cov_type='cat', categories=['A1', 'B1']),
            Covariate(name="C1", cov_type='cont'),
        ]

        corrs = [Correlation(params=['a', 'b'])]
        state = [hboms.Variable('y')]
        obs = [hboms.Observation("Y")]
        dists = [hboms.StanDist("normal", "Y", ["y", "0.1"])]

        # compile test...

        model = HbomsModel(
            "test_covcorr_contcorr",
            state=state,
            odes="ddt_y = a * y + b;",
            init="y_0 = 0.0;",
            obs=obs,
            dists=dists,
            params=params, 
            covariates=covs, 
            correlations=corrs,
            model_dir=self.model_dir,
        )

    def test_covcorr_sample(self):
        """
        Test sampling from a model with correlated parameters and covariates.
        """

        cw_values = {'X': [-2.0, -0.5]}

        params = [
            Parameter("a", 0.1, "random", covariates=['X'], scale=0.05, cw_values=cw_values),
            Parameter("b", 0.1, "random", scale=0.05),
        ]

        covs = [
            Covariate(name="X", cov_type='cat', categories=['A', 'B']),
        ]
        corrs = [Correlation(params=['a', 'b'])]
        state = [hboms.Variable('y')]
        obs = [hboms.Observation("Y")]
        dists = [hboms.StanDist("normal", "Y", ["y", "0.1"])]

        model = HbomsModel(
            "test_covcorr_sample",
            state=state,
            odes="ddt_y = a * y + b;",
            init="y_0 = 0.0;",
            obs=obs,
            dists=dists,
            params=params, 
            covariates=covs, 
            correlations=corrs,
            model_dir=self.model_dir,
        )

        R = 20
        data = {
            "Time" : [np.linspace(1, 10, 5) for _ in range(R)],
            "X": ["A" if r % 2 == 0 else "B" for r in range(R)],
        }

        sims = model.simulate(data, num_simulations=1, seed=144169)

        sim_data, sim_params = sims[0]

        print(sim_params, sim_data)

        # check that the a is sampled with the covariate effect included
        a_sim = sim_params["a"]

        a_A = a_sim[np.array(data["X"]) == "A"]
        a_B = a_sim[np.array(data["X"]) == "B"]

        mean_a_A = np.mean(np.log(a_A))
        mean_a_B = np.mean(np.log(a_B))
        # 0.5 is very lenient, but should be sufficient to show an effect
        assert mean_a_B - mean_a_A > 0.5, "no covariate effect detected in sampled parameter 'a'"


        model.sample(sim_data, chains=1, iter_warmup=200, iter_sampling=200, seed=144169,
                     show_progress=False, step_size=0.01)
        samples = model.fit.stan_variables()

        assert "a" in samples
        assert "b" in samples

        assert samples["a"].shape == (200, R)
        assert samples["b"].shape == (200, R)

        # check that the categorical covariate has an effect

        mean_a_A_post = np.mean(np.log(samples["a"][:, np.array(data["X"]) == "A"]))
        mean_a_B_post = np.mean(np.log(samples["a"][:, np.array(data["X"]) == "B"]))

        print("Posterior means:", mean_a_A_post, mean_a_B_post)

        assert mean_a_B_post - mean_a_A_post > 0.5, "no covariate effect detected in posterior of parameter 'a'"


