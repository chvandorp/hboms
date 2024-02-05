import hboms.stanlang as sl
from hboms.deparse import deparse_expr


class TestExpression:
    def test_binary(self):
        x = sl.Var(sl.Real(), "x")
        y = sl.Var(sl.Real(), "y")

        expr = x + y
        assert deparse_expr(expr) == "x + y"

        expr = x * y
        assert deparse_expr(expr) == "x * y"

        expr = x / y
        assert deparse_expr(expr) == "x / y"

        expr = x - y
        assert deparse_expr(expr) == "x - y"

        expr = sl.LeOp(x, y)
        assert deparse_expr(expr) == "x < y"

        expr = sl.LeEqOp(x, y)
        assert deparse_expr(expr) == "x <= y"

        expr = sl.GrOp(x, y)
        assert deparse_expr(expr) == "x > y"

        expr = sl.GrEqOp(x, y)
        assert deparse_expr(expr) == "x >= y"

        expr = sl.EqOp(x, y)
        assert deparse_expr(expr) == "x == y"

        expr = sl.NeqOp(x, y)
        assert deparse_expr(expr) == "x != y"

    def test_unary(self):
        x = sl.Var(sl.Real(), "x")
        v = sl.Var(sl.Vector(sl.LiteralInt(3)), "v")

        expr = sl.Negate(x)
        assert deparse_expr(expr) == "-x"

        expr = sl.TransposeOp(v)
        assert deparse_expr(expr) == "v'"

    def test_function_call(self):
        x = sl.Var(sl.Real(), "x")
        y = sl.Var(sl.Real(), "y")

        call = sl.Call("func", [])
        assert deparse_expr(call) == "func()"

        call = sl.Call("func", [x])
        assert deparse_expr(call) == "func(x)"

        call = sl.Call("func", [x, y])
        assert deparse_expr(call) == "func(x, y)"

    def test_prob_call(self):
        x = sl.Var(sl.Real(), "x")
        y = sl.Var(sl.Real(), "y")
        z = sl.Var(sl.Real(), "z")

        call = sl.PCall("dist", x, [y], "lpdf")
        assert deparse_expr(call) == "dist_lpdf(x | y)"

        call = sl.PCall("dist", x, [y, z], "lcdf")
        assert deparse_expr(call) == "dist_lcdf(x | y, z)"
