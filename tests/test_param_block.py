from hboms.parameter import RandomParameter, FixedParameter, ParameterBlock
from hboms.correlation import Correlation
from hboms.covariate import CatCovariate
import numpy as np
import pytest

class TestParameterBlock:
    def test_parameter_block(self):
        """Test basic parameter block functionality."""
        par1 = RandomParameter("theta", 1.0)
        par2 = RandomParameter("sigma", 0.5)
        corr = Correlation(["theta", "sigma"])

        block = ParameterBlock([par1, par2], corr_value=corr.value, intensity=corr.intensity)

        assert len(block) == 2, "incorrect number of parameters in block"

        # check that the values of the parameters are set correctly
        assert np.all(block.corr_value == np.eye(2))

        # check that the initial value is set correctly
        expected_value = np.array([0.0, np.log(0.5)])
        assert np.all(block.value == expected_value)

        # check that the level is None by default
        assert block.level is None, "ParameterBlock did not initiate with None level"

        ## now create a block with a level
        
        # check that levels are accepted
        cov = CatCovariate("A", ["x", "y", "z"])
        par1 = RandomParameter("theta", 1.0, level=cov)
        par2 = RandomParameter("sigma", 0.5, level=cov)

        block = ParameterBlock([par1, par2], corr_value=corr.value, intensity=corr.intensity)

        assert block.level == cov, "ParameterBlock did not initiate with correct level"



    def test_parameter_block_bad_types(self):
        """Test that ParameterBlock raises exceptions for bad types."""
        par1 = RandomParameter("theta", 1.0)
        par2 = FixedParameter("sigma", 0.5)
        corr = Correlation(["theta", "sigma"])

        # add parameters must be of type RandomParameter
        with pytest.raises(Exception) as ex:
            block = ParameterBlock([par1, par2], corr_value=corr.value, intensity=corr.intensity)
            

        # levels must be the same for all parameters
        cov1 = CatCovariate("A", ["x", "y", "z"])
        cov2 = CatCovariate("B", ["a", "b"])

        par1 = RandomParameter("theta", 1.0, level=cov1)
        par2 = RandomParameter("sigma", 0.5, level=cov2)

        with pytest.raises(Exception) as ex:
            block = ParameterBlock([par1, par2], corr_value=corr.value, intensity=corr.intensity)
        