import pytest

import hboms
from hboms import deparse
from hboms.parameter import RandomParameter
from hboms.covariate import CatCovariate



class TestNoncenteredLevel():
    def test_noncentered_random_level_codegen(self):

        group_cov = CatCovariate("Group", categories=["A", "B"])
        # positive parameter with non-centered random level
        param = RandomParameter(
            "a", 0.2, level=group_cov, level_type="random", noncentered=True,
        )

        # check generated declarations for parameter block
        decls = param.genstmt_params()

        deparsed = [deparse.deparse_stmt(decl) for decl in decls]

        expected = [
            "array[R] real rand_a;",
            "real loc_a;",
            "real<lower=0.0> scale_a;",
            "real<lower=0.0> scale_a_Group;",
        ]

        assert deparsed == expected, "Generated declarations do not match expected."

        # check generated statements for transformed parameters block

        decls = param.genstmt_trans_params()
        deparsed = [deparse.deparse_stmt(decl) for decl in decls]
        expected = [
            "array[R] real<lower=0.0> a = to_array_1d(exp(loc_a + cholesky_decompose(scale_a^2 * identity_matrix(R) + scale_a_Group^2 * level_Group) * to_vector(rand_a)));"
        ]

        assert deparsed == expected, "Generated transformed parameters do not match expected."


    def test_noncentered_random_level_codegen_covar(self):
        """
        Add a categorical covariate to the location of the random level parameter.
        """

        group_cov = CatCovariate("Group", categories=["A", "B"])
        property_cov = CatCovariate("Property", categories=["low", "high"])
        # positive parameter with non-centered random level
        param = RandomParameter(
            "a", 0.2, level=group_cov, level_type="random", noncentered=True,
            covariates = [property_cov],
        )

        # check generated declarations for parameter block
        decls = param.genstmt_params()

        deparsed = [deparse.deparse_stmt(decl) for decl in decls]

        expected = [
            "array[R] real rand_a;",
            "array[K_Property] real loc_a;",
            "real<lower=0.0> scale_a;",
            "real<lower=0.0> scale_a_Group;",
        ]

        assert deparsed == expected, "Generated declarations do not match expected."

        # check generated statements for transformed parameters block

        decls = param.genstmt_trans_params()
        deparsed = [deparse.deparse_stmt(decl) for decl in decls]
        expected = [
            "array[R] real<lower=0.0> a = to_array_1d(exp(to_vector(loc_a[Property])" \
                 " + cholesky_decompose(scale_a^2 * identity_matrix(R) + scale_a_Group^2" \
                 " * level_Group) * to_vector(rand_a)));"
        ]

        assert deparsed == expected, "Generated transformed parameters do not match expected."


    def test_noncentered_random_level(self):
        """
        Test non-centered parameterization for random level effects.
        These tests create Hboms models, and then try to compile them.
        """

        group_cov = hboms.Covariate("Group", cov_type="cat", categories=["A", "B"])

        # positive parameter with non-centered random level

        params = [
            hboms.Parameter(
                "a", 0.2, "random", 
                level="Group", level_type="random",
                noncentered=True,
            ),
            hboms.Parameter("sigma", 0.1, "fixed")
        ]

        model_kwargs = dict(
            state = [hboms.Variable("x")],
            odes= "ddt_x = a * x * (1-x);",
            init= "x_0 = 0.01;",
            obs = [hboms.Observation("X")],
            dists = [hboms.StanDist("normal", "X", ["x", "sigma"])],
            covariates=[group_cov],
            model_dir="tests/stan-cache",
        )

        model = hboms.HbomsModel(
            name = "nc_test_model_non_centered_randlevel",
            params = params,
            **model_kwargs
        )

        # parameter in some interval 

        params = [
            hboms.Parameter(
                "a", 0.2, "random", lbound=-2.5, ubound=4.0,
                level="Group", level_type="random",
                noncentered=True,
            ),
            hboms.Parameter("sigma", 0.1, "fixed")
        ]

        model = hboms.HbomsModel(
            name = "nc_test_model_non_centered_randlevel_bounded",
            params = params,
            **model_kwargs
        )



    def test_noncentered_random_level_covar(self):
        """
        Test non-centered parameterization for random level effects with
        a categorical covariate on the location.
        These tests create Hboms models, and then try to compile them.
        """

        group_cov = hboms.Covariate("Group", cov_type="cat", categories=["A", "B"])
        property_cov = hboms.Covariate("Property", cov_type="cat", categories=["low", "high"])

        params = [
            hboms.Parameter(
                "a", 0.2, "random", 
                covariates=["Property"],
                level="Group", level_type="random",
                noncentered=True,
            ),
            hboms.Parameter("sigma", 0.1, "fixed")
        ]

        model_kwargs = dict(
            state = [hboms.Variable("x")],
            odes= "ddt_x = a * x * (1-x);",
            init= "x_0 = 0.01;",
            obs = [hboms.Observation("X")],
            dists = [hboms.StanDist("normal", "X", ["x", "sigma"])],
            covariates=[group_cov, property_cov],
            model_dir="tests/stan-cache",
        )

        model = hboms.HbomsModel(
            name = "nc_test_model_non_centered_randlevel_covar",
            params = params,
            **model_kwargs
        )
