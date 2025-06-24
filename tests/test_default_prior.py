# test default prior choice based on lower and upper bounds

from hboms.parameter import RandomParameter, select_default_rand_param_dist
from hboms.deparse import deparse_stmt

class TestDefaultPrior:
    def test_default_prior(self):
        # Test with no bounds
        prior = select_default_rand_param_dist(lbound=None, ubound=None)
        assert prior == "normal"

        # Test with lower bound
        prior = select_default_rand_param_dist(lbound=0.0, ubound=None)
        assert prior == "lognormal"

        # Test with upper bound
        prior = select_default_rand_param_dist(lbound=None, ubound=0.0)
        assert prior == "lognormal"

        # Test with both bounds
        prior = select_default_rand_param_dist(lbound=0.0, ubound=1.0)
        assert prior == "logitnormal"

    def test_prior_stmt_lu_bounds(self):
        # lower and upper bound
        p = RandomParameter(
            name="x", value = 0.5,
            lbound=0.0, ubound=1.0,
        )
        priors = p.genstmt_model()
        expected_stmt = "x ~ logitnormal(loc_x, scale_x, 0.0, 1.0);"
        stmt_strs = [deparse_stmt(s) for s in priors]
        assert expected_stmt in stmt_strs

    def test_prior_stmt_no_bounds(self):
        # no bounds
        p = RandomParameter(
            name="y", value = 0.5,
            lbound=None, ubound=None,
        )
        priors = p.genstmt_model()
        expected_stmt = "y ~ normal(loc_y, scale_y);"
        stmt_strs = [deparse_stmt(s) for s in priors]
        assert expected_stmt in stmt_strs

    def test_prior_stmt_zero_l_bound(self):
        # lower bound only
        p = RandomParameter(
            name="z", value = 0.5,
            lbound=0.0, ubound=None,
        )
        priors = p.genstmt_model()
        expected_stmt = "z ~ lognormal(loc_z, scale_z);"
        stmt_strs = [deparse_stmt(s) for s in priors]
        assert expected_stmt in stmt_strs

    def test_prior_stmt_nonzero_l_bound(self):
        # lower bound only
        p = RandomParameter(
            name="z", value = 0.5,
            lbound=3.14, ubound=None,
        )
        priors = p.genstmt_model()
        expected_stmt = "to_vector(z) - 3.14 ~ lognormal(loc_z, scale_z);"
        stmt_strs = [deparse_stmt(s) for s in priors]
        assert expected_stmt in stmt_strs

    def test_prior_stmt_zero_u_bound(self):
        # upper bound only
        p = RandomParameter(
            name="w", value = 0.5,
            lbound=None, ubound=0.0,
        )
        priors = p.genstmt_model()
        expected_stmt = "-to_vector(w) ~ lognormal(loc_w, scale_w);"
        stmt_strs = [deparse_stmt(s) for s in priors]
        assert expected_stmt in stmt_strs

    def test_prior_stmt_nonzero_u_bound(self):
        # lower bound only
        p = RandomParameter(
            name="z", value = 0.5,
            ubound=3.14, lbound=None,
        )
        priors = p.genstmt_model()
        expected_stmt = "3.14 - to_vector(z) ~ lognormal(loc_z, scale_z);"
        stmt_strs = [deparse_stmt(s) for s in priors]
        assert expected_stmt in stmt_strs


