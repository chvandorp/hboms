from hboms.state import StateVar, State
from hboms import stanlang as sl
from hboms.deparse import deparse_stmt


class TestStateVar:
    def test_default_decl(self):
        x = StateVar("x")
        assert deparse_stmt(x.gen_decl()) == "real x;"

    def test_vector_decl(self):
        x = StateVar("x", dim=3)
        assert deparse_stmt(x.gen_decl()) == "vector[3] x;"

    def test_array_decl(self):
        base_tp = sl.Matrix(sl.LiteralInt(2), sl.LiteralInt(3))
        array_tp = sl.Array(base_tp, [sl.LiteralInt(4), sl.LiteralInt(5)])
        x = StateVar("Ms", stan_type=array_tp)
        assert deparse_stmt(x.gen_decl()) == "array[4, 5] matrix[2, 3] Ms;"


class TestState:
    def test_state_decl(self):
        x = StateVar("x")
        y = StateVar("y", dim=3)
        z = StateVar("z")
        v = StateVar("v", dim=3)
        w = StateVar("w", dim=7)

        
        s = State([x, y, z, v, w])
        
        decls = s.gen_decl()
        
        decl_strs = [deparse_stmt(stmt) for stmt in decls]
        expected_decl_strs = [
            "real x, z;",
            "vector[3] y, v;",
            "vector[7] w;",
        ]
        
        assert decl_strs == expected_decl_strs
        
    def test_is_scalar(self):
        x = StateVar("x")
        y = StateVar("y")
        
        s = State([x,y])
        
        assert s.all_scalar()
        
        z = StateVar("z", dim=3)
        
        s = State([x, y, z])
        
        assert not s.all_scalar()