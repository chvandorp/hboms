from hboms.prior import Prior
from hboms import stanlang as sl
from hboms.deparse import deparse_stmt


class TestPrior:
    def test_prior(self):
        x = sl.Var(sl.Real(), "x")
        prior = Prior("normal", [0.0, 1.0])
        stmts = prior.gen_sampling_stmt(x)
        stmt_strs = [deparse_stmt(stmt) for stmt in stmts]
        expected_stmt_strs = [
            "x ~ normal(0.0, 1.0);"
        ]
        assert stmt_strs == expected_stmt_strs
        
    def test_transformed_prior(self):
        x = sl.Var(sl.Real(), "x")
        def transform(x):
            return sl.Call("log", x)
        prior = Prior("normal", [0.0, 1.0], transform=transform)
        stmts = prior.gen_sampling_stmt(x)
        stmt_strs = [deparse_stmt(stmt) for stmt in stmts]
        expected_stmt_strs = [
            "log(x) ~ normal(0.0, 1.0);"
        ]
        assert stmt_strs == expected_stmt_strs
        