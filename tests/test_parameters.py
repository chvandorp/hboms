from hboms.parameter import RandomParameter, ContCovariate
from hboms.deparse import deparse_stmt


class TestParameter:
    def test_random_param(self):
        """test basic random parameter"""
        p = RandomParameter("theta", 1.0)
        stmts = p.genstmt_params()
        stmt_strs = [deparse_stmt(stmt) for stmt in stmts]
        expected_strs = [
            "array[R] real<lower=0.0> theta;",
            "real loc_theta;",
            "real<lower=0.0> scale_theta;",
        ]
        assert stmt_strs == expected_strs

        stmts = p.genstmt_model()
        stmt_strs = [deparse_stmt(stmt) for stmt in stmts]
        expected_strs = [
            "theta ~ lognormal(loc_theta, scale_theta);",
            "loc_theta ~ student_t(3.0, 0.0, 2.5);",
            "scale_theta ~ student_t(3.0, 0.0, 2.5);",
        ]
        assert stmt_strs == expected_strs

    def test_random_param_cov(self):
        """test random parameter with a covariate"""
        cov = ContCovariate("x")
        p = RandomParameter("theta", 1.0, covariates=[cov])
        stmts = p.genstmt_params()
        stmt_strs = [deparse_stmt(stmt) for stmt in stmts]
        expected_strs = [
            "array[R] real<lower=0.0> theta;",
            "real loc_theta;",
            "real<lower=0.0> scale_theta;",
            "real weight_x_theta;",
        ]
        assert stmt_strs == expected_strs

        stmts = p.genstmt_model()
        stmt_strs = [deparse_stmt(stmt) for stmt in stmts]
        expected_strs = [
            "for ( r in 1:R ) theta[r] ~ lognormal(loc_theta + weight_x_theta * x[r], scale_theta);",
            "loc_theta ~ student_t(3.0, 0.0, 2.5);",
            "scale_theta ~ student_t(3.0, 0.0, 2.5);",
            "weight_x_theta ~ student_t(3.0, 0.0, 2.5);",
        ]
        assert stmt_strs == expected_strs
