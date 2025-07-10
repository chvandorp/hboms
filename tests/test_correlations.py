from hboms.correlation import Correlation
import hboms
import numpy as np
import pytest
import os

class TestCorrelation:
    model_dir = os.path.join("tests", "stan-cache")

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

    def test_bad_parnames(self):
        params = [
            hboms.Parameter("a", 1.0, "random"),
            hboms.Parameter("b", 2.0, "random"),
            hboms.Parameter("c", 3.0, "random")
        ]
        corrs = [hboms.Correlation(["b", "c", "d", "e"])]

        message = "Correlation contains parameter names that are not in the model: d, e"

        with pytest.raises(ValueError, match=message):
            model = hboms.HbomsModel(
                name="test_corrnames_model",
                state=[hboms.Variable("x")],
                odes="ddt_x = -x;",
                init="x_0 = 1.0",
                obs=[hboms.Observation("X")],
                dists=[hboms.StanDist("normal", "X", ["x", "0.1"])],
                params=params,
                correlations=corrs,
                compile_model=False,
                model_dir=self.model_dir
            )



        