"""
Test setting initial parameters.
"""

import hboms
import numpy as np
from hboms.transform import constr_to_unconstr_float

class TestInitSet:
    def test_set_init(self):
        # choose initial param values
        init_pars = {
            "mu" : 1.0,
            "sigma" : 1.0
        }
        # create a simple model
        model = hboms.HbomsModel(
            name="init_test_model",
            params=[
                hboms.Parameter('mu', init_pars["mu"], "random", lbound=None), 
                hboms.Parameter('sigma', init_pars["sigma"], "random")
            ],
            obs=[hboms.Observation('X')],
            state=[hboms.Variable('x')],
            odes="ddt_x = t;",
            init="x_0 = mu;",
            dists=[hboms.StanDist("normal", "X", params=['x', 'sigma'])],
            compile_model=False
        )
        # check initial params
        for param in model._params:
            assert param.value == init_pars[param.name]
        
        # choose new initial param values
        new_init_pars = {
            "mu" : 2.0,
            "sigma" : 2.0
        }
        # set new initial param values
        model.set_init(new_init_pars)


    def test_set_init_corr(self):
        # choose initial param values
        init_pars = {
            "mu" : 1.0,
            "sigma" : 1.0
        }
        # create a simple model
        model = hboms.HbomsModel(
            name="init_test_model_corr",
            params=[
                hboms.Parameter('mu', init_pars["mu"], "random", lbound=None), 
                hboms.Parameter('sigma', init_pars["sigma"], "random")
            ],
            obs=[hboms.Observation('X')],
            state=[hboms.Variable('x')],
            odes="ddt_x = t;",
            init="x_0 = mu;",
            dists=[hboms.StanDist("normal", "X", params=['x', 'sigma'])],
            correlations=[hboms.Correlation(["mu", "sigma"])],
            compile_model=False
        )
        # check initial params
        for param in model._params:
            assert param.value == init_pars[param.name]

        def get_trans_val(param_block, init_pars):
            names = [param.name for param in param_block._params]
            vals = [init_pars[name] for name in names]
            lbs = [param.lbound for param in param_block._params]
            ubs = [param.ubound for param in param_block._params]
            trans_vals = [
                constr_to_unconstr_float(val, lb, ub) 
                for val, lb, ub in zip(vals, lbs, ubs)
            ]
            return trans_vals
        
        for param_block in model._corr_params:
            trans_vals = get_trans_val(param_block, init_pars)
            assert np.allclose(trans_vals, param_block._value)
        
        # choose new initial param values
        new_init_pars = {
            "mu" : 2.0,
            "sigma" : 2.0
        }
        # set new initial param values
        model.set_init(new_init_pars)

        for param_block in model._corr_params:
            trans_vals = get_trans_val(param_block, new_init_pars)
            assert np.allclose(trans_vals, param_block._value)