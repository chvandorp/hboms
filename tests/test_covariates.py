from hboms.covariate import (
    ContCovariate, CatCovariate, group_covars, covar_dispatch
)
import hboms.stanlang as sl
from hboms.deparse import deparse_stmt


class TestCovariate:
    def test_cont_covariate(self):
        """test for scaler continuous covariate"""
        cov = ContCovariate("X")
        
        stmts = cov.genstmt_data()
        stmt_strs = [deparse_stmt(stmt) for stmt in stmts]
        
        assert stmt_strs == ["array[R] real X;"]
        
        weight_var_decl = sl.Decl(cov.weight_var("theta"))
        weight_var_decl_str = deparse_stmt(weight_var_decl)
        
        assert weight_var_decl_str == "real weight_X_theta;"
        
    def test_cont_vv_covariate(self):
        """test for vector-valued continuous covariate"""
        cov = ContCovariate("X", dim=3)
        
        stmts = cov.genstmt_data()
        # remove comments 
        for stmt in stmts:
            stmt.comment = []
        stmt_strs = [deparse_stmt(stmt) for stmt in stmts]
        
        assert stmt_strs == ["int<lower=1> K_X;", "array[R] vector[K_X] X;"]
        
        weight_var_decl = sl.Decl(cov.weight_var("theta"))
        weight_var_decl_str = deparse_stmt(weight_var_decl)
        
        assert weight_var_decl_str == "vector[K_X] weight_X_theta;"
        
    def test_cat_covariate(self):
        categories = ["A", "B", "C"]
        cov = CatCovariate("X", categories)
        
        assert cov.num_cats == 3
        
        stmts = cov.genstmt_data()
        # remove comments 
        for stmt in stmts:
            stmt.comment = []
        stmt_strs = [deparse_stmt(stmt) for stmt in stmts]
        
        assert stmt_strs == ["int<lower=1> K_X;", "array[R] int<lower=1, upper=K_X> X;"]
        
    def test_covar_grouping(self):
        covs = [
            ContCovariate("X", dim=3), 
            CatCovariate('Z', ["A", "B"]),
            ContCovariate("Y"),
            CatCovariate('W', ["A", "B", "C"])
        ]
        
        cont, cat = group_covars(covs)
        
        assert [cov.name for cov in cont] == ["X", "Y"]
        
        assert [cov.name for cov in cat] == ["Z", "W"]
        
    def test_dispatch(self):
        cov = covar_dispatch("X", "cont")
        
        assert isinstance(cov, ContCovariate)
        assert cov.name == "X"
        
        cats = ["A", "B"]
        cov = covar_dispatch("Y", "cat", categories=cats)
        
        assert isinstance(cov, CatCovariate)
        assert cov.name == "Y"     
        assert cov.cats == cats


class TestLevelRestrictedCovariate:
    def test_genstmt_transformed_data(self):
        categories = ["A", "B", "C"]
        cov = CatCovariate("X", categories)
        
        level = CatCovariate("L", ["G1", "G2", "G3", "G4"])
        
        stmts = cov.genstmt_transformed_data(level)
        # remove comments 
        for stmt in stmts:
            stmt.comment = []
        stmt_strs = [deparse_stmt(stmt) for stmt in stmts]
        
        expected_stmts = [
            "array[K_L] int<lower=1, upper=K_X> X_restrict_L;",
            "for ( r in 1:R ) X_restrict_L[L[r]] = X[r];",
        ]
        
        assert stmt_strs == expected_stmts


    def test_gen_covariate_restrictions(self):
        from hboms.parameter import RandomParameter
        from hboms.gencommon import gen_covariate_restrictions

        level1 = CatCovariate("L", ["G1", "G2", "G3", "G4"])
        level2 = CatCovariate("M", ["H1", "H2", "H3", "H4"])
        cov1 = CatCovariate("X", ["A", "B"])
        cov2 = CatCovariate("Y", ["C", "D"])
        
        
        p1 = RandomParameter(
            name="param1",
            value=0.5,
            level=level1,
            level_type="fixed",
            covariates=[cov1],
        )
        
        # multiple restrictions for same covariate and different level
        p2 = RandomParameter(
            name="param2",
            value=0.5,
            level=level2,
            level_type="fixed",
            covariates=[cov1],
        )
        
        p3 = RandomParameter(
            name="param3",
            value=0.5,
            level=level2,
            level_type="fixed",
            covariates=[cov2],
        )

        # test that duplicate restrictions are not added
        p4 = RandomParameter(
            name="param4",
            value=0.5,
            level=level1,
            level_type="fixed",
            covariates=[cov1],
        )

        # parameter with random level should not generate restrictions
        p5 = RandomParameter(
            name="param5",
            value=0.5,
            level=level1,
            level_type="random",
            covariates=[cov2],
        )
        
        params = [p1, p2, p3, p4, p5]
        
        stmts = gen_covariate_restrictions(params)
        # remove comments 
        for stmt in stmts:
            stmt.comment = []
        stmt_strs = [deparse_stmt(stmt) for stmt in stmts]
        
        expected_stmts = [
            "array[K_L] int<lower=1, upper=K_X> X_restrict_L;",
            "for ( r in 1:R ) X_restrict_L[L[r]] = X[r];",
            "array[K_M] int<lower=1, upper=K_X> X_restrict_M;",
            "for ( r in 1:R ) X_restrict_M[M[r]] = X[r];",
            "array[K_M] int<lower=1, upper=K_Y> Y_restrict_M;",
            "for ( r in 1:R ) Y_restrict_M[M[r]] = Y[r];",
        ]
        
        assert stmt_strs == expected_stmts