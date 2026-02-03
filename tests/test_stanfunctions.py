import numpy as np

import hboms
from hboms.parameter import RandomParameter, FixedParameter, ParameterBlock
from hboms.genmodel import require_logitnormal_functions
from hboms.prior import AbsContPrior


class TestLogitNormalDist():
    def test_require_logitnorm(self):
        """
        Test if the logitnormal distribution functions are added to the Stan
        functions block when required.
        """

        # a single random parameter with logitnormal distribution
        params = [
            RandomParameter("a", 0.5, lbound=0.0, ubound=1.0, distribution="logitnormal")
        ]
        corr_params = []

        assert require_logitnormal_functions(params, corr_params), "logitnormal functions should be required"


        # a single random parameter with normal distribution
        params = [
            RandomParameter("a", 0.5, lbound=None, ubound=None, distribution="normal")
        ]

        assert not require_logitnormal_functions(params, corr_params), "logitnormal functions should not be required"


        # a single fixed parameter with logitnormal prior
        prior = AbsContPrior("logitnormal", [0.0, 1.0, 0.0, 1.0])

        params = [
            FixedParameter("b", 0.3, lbound=0.0, ubound=1.0, prior=prior)
        ]

        assert require_logitnormal_functions(params, corr_params), "logitnormal functions should be required"


        # two correlated parameters, one with logitnormal distribution
        params = [
            RandomParameter("a", 0.5, lbound=None, ubound=None, distribution="normal"),
            RandomParameter("b", 0.3, lbound=0.0, ubound=1.0, distribution="logitnormal")
        ]

        corr_params = [
            ParameterBlock(params, np.eye(2), 2.0)            
        ]

        assert params[1] in corr_params[0], "parameter 'b' should be in correlated parameters"

        assert not require_logitnormal_functions(params, corr_params), "logitnormal functions should not be required"
