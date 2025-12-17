from hboms.covariate import CatCovariate
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

    def test_default_prior(self):
        p = RandomParameter("theta", 1.0, lbound=None, ubound=None)
        assert p._distribution == "normal"

        p = RandomParameter("theta", 1.0, lbound=0.0, ubound=None)
        assert p._distribution == "lognormal"

        p = RandomParameter("theta", 1.0, lbound=0.0, ubound=1.0)
        assert p._distribution == "logitnormal"

    def test_random_param_level(self):
        cov = CatCovariate("x", categories=["A", "B", "C"])
        p = RandomParameter("a", 1.0, level=cov, level_type="random", lbound=None)

        # check that correct priors are generated
        stmts = p.genstmt_model()

        stmt_strs = [deparse_stmt(stmt) for stmt in stmts]
        expected_strs = [
            "to_vector(a) ~ multi_normal(rep_vector(loc_a, R), scale_a^2 * identity_matrix(R) + scale_a_x^2 * level_x);",
            "loc_a ~ student_t(3.0, 0.0, 2.5);",
            "scale_a ~ student_t(3.0, 0.0, 2.5);",
            "scale_a_x ~ student_t(3.0, 0.0, 2.5);",
        ]

        assert stmt_strs == expected_strs, "Priors not generated as expected"

    def test_random_param_level_nc(self):
        cov = CatCovariate("x", categories=["A", "B", "C"])
        p = RandomParameter("a", 1.0, level=cov, level_type="random", lbound=None, noncentered=True)

        # check that correct priors are generated
        stmts = p.genstmt_model()

        stmt_strs = [deparse_stmt(stmt) for stmt in stmts]
        expected_strs = [
            "rand_a ~ std_normal();",
            "loc_a ~ student_t(3.0, 0.0, 2.5);",
            "scale_a ~ student_t(3.0, 0.0, 2.5);",
            "scale_a_x ~ student_t(3.0, 0.0, 2.5);",
        ]

        assert stmt_strs == expected_strs, "Priors not generated as expected"