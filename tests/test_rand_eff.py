import pytest
import hboms
import numpy as np
import os

class TestRandomEffects:
    output_dir = os.path.join("tests", "stan-cache")

    def test_rand_eff_gq_code(self):
        from hboms.parameter import RandomParameter
        from hboms.deparse import deparse_stmt

        param_a = RandomParameter("a", 0.1, noncentered=True)

        stmts = param_a.genstmt_gq()
        assert len(stmts) == 0, "GQ statements should be empty for non-centered random parameters."

        param_b = RandomParameter("b", 0.2)

        stmts = param_b.genstmt_gq()
        assert len(stmts) == 1, "GQ statements should contain one statement for centered random parameters."

        stmt = stmts[0]
        stmt_str = deparse_stmt(stmt).strip()
        expected_str = "array[R] real rand_b = to_array_1d((to_vector(log(b)) - loc_b) / scale_b);"
        assert stmt_str == expected_str, f"Expected statement: {expected_str}, but got: {stmt_str}"

        param_c = RandomParameter("c", 0.3, lbound=0.0, ubound=1.0)

        stmts = param_c.genstmt_gq()
        assert len(stmts) == 1, "GQ statements should contain one statement for bounded random parameters."

        stmt = stmts[0]
        stmt_str = deparse_stmt(stmt).strip()
        expected_str = "array[R] real rand_c = to_array_1d((to_vector(logit(c)) - loc_c) / scale_c);"
        assert stmt_str == expected_str, f"Expected statement: {expected_str}, but got: {stmt_str}"

        param_d = RandomParameter("d", 2.4, lbound=1.0)
        stmts = param_d.genstmt_gq()
        assert len(stmts) == 1, "GQ statements should contain one statement for lower-bounded random parameters."

        stmt = stmts[0]
        stmt_str = deparse_stmt(stmt).strip()
        expected_str = "array[R] real rand_d = to_array_1d((log(to_vector(d) - 1.0) - loc_d) / scale_d);"
        assert stmt_str == expected_str, f"Expected statement: {expected_str}, but got: {stmt_str}"

        param_e = RandomParameter("e", -1.0, ubound=0.0, lbound=None)
        stmts = param_e.genstmt_gq()
        assert len(stmts) == 1, "GQ statements should contain one statement for upper-bounded random parameters."

        stmt = stmts[0]
        stmt_str = deparse_stmt(stmt).strip()
        expected_str = "array[R] real rand_e = to_array_1d((-log(-to_vector(e)) - loc_e) / scale_e);"
        assert stmt_str == expected_str, f"Expected statement: {expected_str}, but got: {stmt_str}"

        param_f = RandomParameter("f", 0.5, lbound=1.0, ubound=5.0)
        stmts = param_f.genstmt_gq()
        assert len(stmts) == 1, "GQ statements should contain one statement for bounded random parameters."

        stmt = stmts[0]
        stmt_str = deparse_stmt(stmt).strip()
        expected_str = "array[R] real rand_f = to_array_1d((logit((to_vector(f) - 1.0) / (5.0 - 1.0)) - loc_f) / scale_f);"
        assert stmt_str == expected_str, f"Expected statement: {expected_str}, but got: {stmt_str}"


    def test_rand_eff_gq_code_catcov(self):
        from hboms.parameter import RandomParameter
        from hboms.deparse import deparse_stmt
        from hboms.covariate import CatCovariate

        # test with categorical covariate: index loc with categores

        cat_covar = CatCovariate("X", ["A", "B", "C"])

        param_a = RandomParameter("a", 0.5, covariates=[cat_covar])

        stmts = param_a.genstmt_gq()
        assert len(stmts) == 1, "GQ statements should contain one statement for categorical covariate random parameters."

        stmt = stmts[0]
        stmt_str = deparse_stmt(stmt).strip()
        expected_str = "array[R] real rand_a = to_array_1d((to_vector(log(a)) - to_vector(loc_a[X])) / scale_a);"
        assert stmt_str == expected_str, f"Expected statement: {expected_str}, but got: {stmt_str}"


        # test with level: index loc with restriced categories

        level = CatCovariate("Y", ["Low", "High"])

        param_b = RandomParameter("b", 1.0, covariates=[cat_covar], level=level)

        stmts = param_b.genstmt_gq()
        assert len(stmts) == 1, "GQ statements should contain one statement for categorical covariate random parameters."

        stmt = stmts[0]
        stmt_str = deparse_stmt(stmt).strip()
        expected_str = "array[K_Y] real rand_b = to_array_1d((to_vector(log(b)) - to_vector(loc_b[X_restrict_Y])) / scale_b);"
        assert stmt_str == expected_str, f"Expected statement: {expected_str}, but got: {stmt_str}"


    def test_rand_eff_gq_code_rand_level(self):
        from hboms.parameter import RandomParameter
        from hboms.deparse import deparse_stmt
        from hboms.covariate import CatCovariate

        # test with categorical covariate: index loc with categores

        level = CatCovariate("X", ["A", "B", "C"])

        param_a = RandomParameter("a", 0.5, level=level, level_type="random")

        stmts = param_a.genstmt_gq()
        assert len(stmts) == 1, "GQ statements should contain one statement for categorical covariate random parameters."

        stmt = stmts[0]
        stmt_str = deparse_stmt(stmt).strip()
        expected_str = "array[R] real rand_a = to_array_1d((to_vector(log(a)) - loc_a) / sqrt(scale_a^2 + scale_a_X^2));"
        assert stmt_str == expected_str, f"Expected statement: {expected_str}, but got: {stmt_str}"



        
    def test_rand_eff_model(self):
        params = [
            hboms.Parameter("a", 0.3, "random", scale=0.1),
            hboms.Parameter("b", 0.2, "random", scale=0.1, noncentered=True)
        ]

        model = hboms.HbomsModel(
            name = "test_rand",
            state = [hboms.Variable("x")],
            odes = "ddt_x = a * x + b;",
            init = "x_0 = 0.01;",
            params = params,
            obs = [hboms.Observation("X")],
            dists = [hboms.StanDist("normal", "X", ["x", "0.1"])],
            model_dir = self.output_dir,
        )

        R = 10
        data = {"Time" : [np.linspace(1, 5, 5) for _ in range(R)]}
        sims = model.simulate(data, 1)
        sim_data, sim_params = sims[0]

        model.sample(
            sim_data, iter_warmup=200, iter_sampling=200, 
            chains=1, seed=42, show_progress=False,
            step_size=0.01
        )

        assert "rand_a" in model.fit.stan_variables(), "rand_a should be in the fitted model variables."
        assert "rand_b" in model.fit.stan_variables(), "rand_b should be in the fitted model variables."

        rand_a_samples = model.fit.stan_variable("rand_a")

        # Check that rand_a samples are centered around 0 with unit variance

        mean_rand_a = np.mean(rand_a_samples)
        std_rand_a = np.std(rand_a_samples)

        assert abs(mean_rand_a) < 0.1, f"Mean of rand_a samples should be close to 0, got {mean_rand_a}"
        assert abs(std_rand_a - 1.0) < 0.1, f"Std of rand_a samples should be close to 1, got {std_rand_a}"

