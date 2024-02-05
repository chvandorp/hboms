from hboms.observation import Observation
import hboms.stanlang as sl 
from hboms.deparse import deparse_stmt


class TestObservation:
    def test_observation(self):
        obs = Observation("X")
        
        stmts = obs.genstmt_data()
        
        stmt_strs = [deparse_stmt(stmt) for stmt in stmts]
        
        assert stmt_strs == ["array[R, max(N)] real X;"]

    def test_int_observation(self):
        obs = Observation("X", obs_type=sl.Int())
        
        stmts = obs.genstmt_data()
        
        stmt_strs = [deparse_stmt(stmt) for stmt in stmts]
        
        assert stmt_strs == ["array[R, max(N)] int X;"]

    def test_vector_observation(self):
        obs = Observation("X", obs_type=sl.Vector(sl.LiteralInt(3)))
        
        stmts = obs.genstmt_data()
        
        stmt_strs = [deparse_stmt(stmt) for stmt in stmts]
        
        assert stmt_strs == ["array[R, max(N)] vector[3] X;"]

    def test_array_observation(self):
        obs = Observation("X", obs_type=sl.Array(sl.Int(), sl.LiteralInt(3)))
        
        stmts = obs.genstmt_data()
        
        stmt_strs = [deparse_stmt(stmt) for stmt in stmts]
        
        assert stmt_strs == ["array[R, max(N), 3] int X;"]

    def test_censored_observation(self):
        obs = Observation("X", censored=True)
        
        assert obs.cc_name == "cc_X"
        
        stmts = obs.genstmt_data()
        
        # remove comments
        for stmt in stmts:
            stmt.comment = []
        
        stmt_strs = [deparse_stmt(stmt) for stmt in stmts]
        
        assert stmt_strs == [
            "array[R, max(N)] real X;", 
            "array[R, max(N)] int<lower=-1, upper=2> cc_X;"
        ]
