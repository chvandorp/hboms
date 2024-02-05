from hboms.correlation import Correlation
import numpy as np
import pytest

class TestCorrelation:
    def test_correlation(self):
        parnames = ["a", "b", "c"]
        
        corr = Correlation(parnames)
        
        assert corr.dim == len(parnames)
        
        assert (corr.value == np.eye(len(parnames))).all()
        
    def test_bad_value(self):
        parnames = ["a", "b", "c"]
        wrong_shaped_value = 0.9 * np.eye(2) + np.full((2, 2), 0.1)
        
        # test that an exception is raised
        with pytest.raises(Exception) as ex:
            corr = Correlation(parnames, value=wrong_shaped_value)
        
        wrong_typed_value = "this is not a correlation matrix"
        
        # test that an exception is raised
        with pytest.raises(Exception) as ex:
            corr = Correlation(parnames, value=wrong_typed_value)
        
        