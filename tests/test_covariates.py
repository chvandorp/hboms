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