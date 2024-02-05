import hboms.stanlang as sl
from hboms.deparse import deparse_stmt


class TestStatement:
    def test_for_loop(self):
        index = sl.Var(sl.Int(), "i")
        sequence = sl.Range(sl.LiteralInt(1), sl.LiteralInt(10))
        x = sl.Var(sl.Int(), "x")
        content = sl.AddAssign(x, index)
        stmt = sl.ForLoop(index, sequence, content)
        assert deparse_stmt(stmt) == "for ( i in 1:10 ) x += i;"

    def test_if_stmt(self):
        x = sl.Var(sl.Real(), "x")
        y = sl.Var(sl.Real(), "y")
        condition = sl.LeEqOp(x, y)
        content = sl.Break()
        stmt = sl.IfStatement(condition, content)
        assert deparse_stmt(stmt) == "if ( x <= y ) break;"

    def test_sampling_stmt(self):
        x = sl.Var(sl.Real(), "x")
        zero, one = sl.LiteralReal(0.0), sl.LiteralReal(1.0)
        stmt = sl.Sample(x, sl.Call("normal", [zero, one]))
        assert deparse_stmt(stmt) == "x ~ normal(0.0, 1.0);"

    def test_return_stmt(self):
        x = sl.Var(sl.Real(), "x")
        stmt = sl.Return(x^sl.LiteralInt(2))
        assert deparse_stmt(stmt) == "return x^2;"
        
    def test_decl_assing_stmt(self):
        x = sl.Var(sl.Vector(sl.LiteralInt(3)), "x")
        vec = sl.LiteralVector([
            sl.LiteralReal(1.0), sl.LiteralReal(2.0), sl.LiteralReal(3.0)
        ])
        stmt = sl.DeclAssign(x, vec)
        assert deparse_stmt(stmt) == "vector[3] x = [1.0, 2.0, 3.0]';"