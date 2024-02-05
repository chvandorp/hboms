import hboms.stanlang as sl
from hboms.analyze import find_variables_expr, find_variables_stmt



class TestFindVariables:
    def test_find_single_variable(self):
        """simplest case"""
        x = sl.Var(sl.Int(), "x")
        expr = x
        res = find_variables_expr(expr)
        assert res == [x]
        
    def test_find_multiple_variables(self):
        x = sl.Var(sl.Real(), "x")
        y = sl.Var(sl.Real(lower=x), "y")
        z = sl.Var(sl.Real(), "z")
        w = sl.Var(sl.Real(lower=z), "w")
        expr = (y + w) * z
        res = find_variables_expr(expr)
        expected_variables = [x, y, z, w]
        
        assert all([var in res for var in expected_variables])
        
    def test_find_dim_variables(self):
        x = sl.Var(sl.Int(), "x")
        y = sl.Var(sl.Vector(x, lower=sl.LiteralReal(0.0)), "y")
        
        res = find_variables_expr(y)
        
        expected_variables = [x, y]
        
        assert all([var in res for var in expected_variables])
        
        n = sl.Var(sl.Int(), "n")
        m = sl.Var(sl.Int(), "m")
        z = sl.Var(sl.Array(sl.Int(lower=x), [n, m]), "z")
        
        res = find_variables_expr(z)
        
        expected_variables = [z, x, n, m]
        
        assert all([var in res for var in expected_variables])
        
    def test_find_vars_in_stmt(self):
        stmt = sl.ForLoop(
            (i := sl.Var(sl.Int(), "i")),
            sl.Range(sl.LiteralInt(1), sl.Var(sl.Int(), "N")),
            sl.Scope([
                sl.DeclAssign((x := sl.Var(sl.Real(), "x")), i * i),
                sl.IfStatement(sl.GrEqOp(x, sl.Var(sl.Real(), "y")), sl.Break())
            ])
        )
        
        res = find_variables_stmt(stmt)
        variable_names = {v.name for v in res}
        
        expected_variable_names = {"i", "N", "x", "y"}
        
        assert variable_names == expected_variable_names
        
        