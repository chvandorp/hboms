from hboms import transform
import numpy as np
import scipy


class TestTransform:
    def test_transform(self):
        tra = transform.Transform(np.log, np.exp)
        
        assert tra(1.0) == 0.0
        
        test_values = [0.5, 1.0, 2.0, 4.0, 8.0]
        
        for x in test_values:
            y = tra(x)
            assert np.isclose(x, tra.inverse()(y))
        
    def test_composition(self):
        tra1 = transform.Transform(lambda x: 2*x, lambda x: 0.5*x)
        tra2 = transform.Transform(lambda x: x+1, lambda x: x-1)
        
        tra3 = tra1 @ tra2
        
        test_values = np.array([1.0, 2.0, 3.0])
        test_answers = 2*(test_values + 1)
        
        for x, y in zip(test_values, test_answers):
            assert tra3(x) == y
            assert tra3.inverse()(y) == x
            
    def test_log_transform(self):
        tra = transform.LogTransform()
        
        test_values = np.linspace(0.1, 3.0, 30)
        
        for x in test_values:
            y = np.log(x)
            assert np.isclose(tra(x), y)
            assert np.isclose(tra.inverse()(y), x)
            
    def test_logit_transformm(self):
        tra = transform.LogitTransform()
        
        test_values = np.linspace(0.01, 0.99, 30)
        
        for x in test_values:
            y = scipy.special.logit(x)
            assert np.isclose(tra(x), y)
            assert np.isclose(tra.inverse()(y), x)



from hboms.parameter import domain_transform
import hboms.stanlang as sl
from hboms.deparse import deparse_expr


class TestDomainTransform:
    def test_domain_transform_loc_scale(self):
        # test with loc and scale

        var = sl.Var(sl.Real(), "x")
        lbound, ubound = 0.0, 1.0
        loc = sl.LiteralReal(0.5)
        scale = sl.LiteralReal(0.1)
        transformed_var = domain_transform(var, lbound, ubound, loc=loc, scale=scale)

        expected_str = "inv_logit(0.5 + 0.1 * x)"
        gen_str = deparse_expr(transformed_var)
        assert gen_str == expected_str, "Expected: {}, Got: {}".format(expected_str, gen_str)

    def test_domain_transform_array_lbound(self):
        # test with non-zero lower bound and array input
        var = sl.Var(sl.Array(sl.Real(), sl.LiteralInt(3)), "x")
        lbound = 1.0
        transformed_var = domain_transform(var, lbound, None, array=True)

        expected_str = "to_array_1d(1.0 + to_vector(exp(x)))"
        gen_str = deparse_expr(transformed_var)
        assert gen_str == expected_str, "Expected: {}, Got: {}".format(expected_str, gen_str)

    def test_domain_transform_array_lbound_ubound(self):
        # test with non-zero lower and upper bounds and array input
        var = sl.Var(sl.Array(sl.Real(), sl.LiteralInt(3)), "x")
        lbound, ubound = 1.0, 2.0
        transformed_var = domain_transform(var, lbound, ubound, array=True)

        expected_str = "to_array_1d(1.0 + (2.0 - 1.0) * to_vector(inv_logit(x)))"
        gen_str = deparse_expr(transformed_var)
        assert gen_str == expected_str, "Expected: {}, Got: {}".format(expected_str, gen_str)