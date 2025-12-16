import pytest

import hboms



class TestNoncenteredLevel():
    def test_noncentered_random_level(self):

        # FIXME: currently non-centered with random levels is not implemented

        with pytest.raises(NotImplementedError):
            group_cov = hboms.Covariate("Group", cov_type="cat", categories=["A", "B"])

            params = [
                hboms.Parameter(
                    "a", 0.2, "random", 
                    level="Group", level_type="random",
                    noncentered=True,
                ),
                hboms.Parameter("sigma", 0.1, "fixed")
            ]

            model = hboms.HbomsModel(
                name = "nc_test_model_non_centered_randlevel",
                params = params,
                state = [hboms.Variable("x")],
                odes= "ddt_x = a * x * (1-x);",
                init= "x_0 = 0.01;",
                obs = [hboms.Observation("X")],
                dists = [hboms.StanDist("normal", "X", ["x", "sigma"])],
                covariates=[group_cov],
                model_dir="tests/stan-cache",
            )
