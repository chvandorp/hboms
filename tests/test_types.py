import pytest

from hboms import stanlang as sl
from hboms.deparse import deparse_decl


class TestStanTypes:
    def test_int(self):
        x = sl.Int()
        assert deparse_decl(x) == "int"
        x = sl.Int(lower=sl.LiteralInt(1))
        assert deparse_decl(x) == "int<lower=1>"

    def test_real(self):
        x = sl.Real()
        assert deparse_decl(x) == "real"
        x = sl.Real(lower=sl.LiteralReal(0.0))
        assert deparse_decl(x) == "real<lower=0.0>"

    def test_vector(self):
        x = sl.Vector(sl.LiteralInt(3))
        assert deparse_decl(x) == "vector[3]"

    def test_matrix(self):
        x = sl.Matrix(sl.LiteralInt(3), sl.LiteralInt(4))
        assert deparse_decl(x) == "matrix[3, 4]"

    def test_array(self):
        v = sl.Vector(sl.LiteralInt(4))
        x = sl.Array(v, [sl.LiteralInt(2), sl.LiteralInt(3)])
        assert deparse_decl(x) == "array[2, 3] vector[4]"

    def test_tuple(self):
        x = sl.Tuple((sl.Int(), sl.Real()))
        assert deparse_decl(x) == "tuple(int, real)"
        y = sl.Tuple((x, sl.Vector(sl.LiteralInt(3), lower=sl.LiteralReal(-1.0))))
        assert deparse_decl(y) == "tuple(tuple(int, real), vector<lower=-1.0>[3])"

    def test_type_equality(self):
        """types with different bounds should not be equal"""
        x = sl.Real(lower=sl.LiteralReal(0.0))
        xprime = sl.Real(lower=sl.LiteralReal(0.0))
        y = sl.Real()
        z = sl.Real(lower=sl.LiteralReal(1.0))
        
        assert x != y
        assert x == xprime
        assert x != z

