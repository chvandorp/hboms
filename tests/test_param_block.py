from hboms.parameter import RandomParameter, FixedParameter, ParameterBlock
from hboms.correlation import Correlation
from hboms.covariate import CatCovariate, ContCovariate
from hboms.deparse import deparse_stmt
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



    def test_parameter_block_no_params(self):
        """Test that ParameterBlock raises exception when no parameters are provided."""
        with pytest.raises(Exception) as ex:
            block = ParameterBlock([], corr_value=np.eye(2), intensity=1.0)



    def test_parameter_block_cont_cov(self):
        """Test that ParameterBlock handles continuous covariates correctly."""
        cov1 = ContCovariate("A")
        par1 = RandomParameter("theta", 1.0, covariates=[cov1])
        par2 = RandomParameter("sigma", 0.5)
        corr = Correlation(parnames=["theta", "sigma"], value=np.eye(2), intensity=1.0)

        block = ParameterBlock([par1, par2], corr_value=corr.value, intensity=corr.intensity)

        assert len(block._contcovs) == 1, "ParameterBlock did not register continuous covariate"

        contcov = block._contcovs[0]
        assert contcov.name == "A", "Continuous covariate name mismatch"
        assert contcov.dim is None, "Continuous covariate dimension mismatch"


        # multi-dimensional continuous covariate
        cov2 = ContCovariate("B", dim=3)
        par1 = RandomParameter("theta", 1.0, covariates=[cov2])
        par2 = RandomParameter("sigma", 0.5)
        corr = Correlation(parnames=["theta", "sigma"], value=np.eye(2), intensity=1.0)

        block = ParameterBlock([par1, par2], corr_value=corr.value, intensity=corr.intensity)
        assert len(block._contcovs) == 1, "ParameterBlock did not register continuous covariate"
        contcov = block._contcovs[0]
        assert contcov.name == "B", "Continuous covariate name mismatch"
        assert contcov.dim == 3, "Continuous covariate dimension mismatch"

        # make sure the model code generation works with continuous covariates
        stmts = block.genstmt_model()
        assert len(stmts) > 0, "No model statements generated for ParameterBlock with continuous covariate"

        deparsed_stmts = [deparse_stmt(stmt) for stmt in stmts]
        stmt_str = "\n".join(deparsed_stmts)
        expected_weight_var = "weight_B_theta_sigma'"

        assert expected_weight_var in stmt_str, "Model code does not contain expected weight variable"

        # make sure that the weight variable is declared correctly

        stmts = block.genstmt_trans_params()
        deparsed_stmts = [deparse_stmt(stmt) for stmt in stmts]
        stmt_str = "\n".join(deparsed_stmts)
        expected_decl = "matrix[K_B, 2] weight_B_theta_sigma = [weight_B_theta', zeros_vector(K_B)']';"
        assert expected_decl in stmt_str, "Transformed parameter code does not contain expected weight variable declaration"


        # make sure that the dim for B is defined in data block
        
        stmts = cov2.genstmt_data()
        deparsed_stmts = [deparse_stmt(stmt) for stmt in stmts]
        stmt_str = "\n".join(deparsed_stmts)
        expected_dim_decl = "int<lower=1> K_B;"

        assert expected_dim_decl in stmt_str, "Data block code does not contain expected dimension declaration for continuous covariate"