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
        
        