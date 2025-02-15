"""
test auxiliary functions for the model class. Such as preparing data sets,
preparing initial values etc.
"""


import hboms
from hboms.model import prepare_data, prepare_init, prepare_simulation_times
import numpy as np
import scipy.stats as sts
import os


class TestPrepareData:
    def test_prepare_data(self):
        N = [10, 20, 15]
        R = len(N)
        data = {
            "Time": [np.arange(1, n + 1) for n in N],
            "X": [sts.norm.rvs(0.0, 1.0, size=n) for n in N],
            "U": sts.norm.rvs(0.0, 1.0, size=R),
        }
        obs = [hboms.observation.Observation("X")]
        covs = [hboms.covariate.ContCovariate("U")]
        params = [hboms.parameter.ConstParameter("theta", 1.0)]
        n_sim = 100

        # test the prepare_data function

        prep_data = prepare_data(data, params, obs, covs, n_sim)

        # test that all keys are present

        expected_keys = [
            "R", "N", "Time", "X", "theta", "U", "NSim", "TimeSim"
        ]

        assert all([k in prep_data for k in expected_keys])

        # test that R = 3

        assert prep_data["R"] == R

        # test that the number of time points is correct

        assert prep_data["N"] == N

        # test that Time and X are rectangular

        X_rect = prep_data["X"]
        N_rect = [len(x) for x in X_rect]

        assert all([n == np.max(N) for n in N_rect])

        Time_rect = prep_data["Time"]
        N_rect = [len(t) for t in Time_rect]

        assert all([n == np.max(N) for n in N_rect])

        # TODO: test that X and Time have the correct values.

        # TODO: test that NSim and TimeSim are correct.
        
    def test_prepare_init(self):
        values = {"a" : 1.0, "b" : 2.0, "c" : 3.0, "d" :4.0}
        scales = {"b" : 0.1, "d" : 1.0} # d gets the default
        params = [
            hboms.parameter.FixedParameter("a", values["a"]),
            hboms.parameter.RandomParameter("b", values["b"], scale=scales["b"]),
            hboms.parameter.IndivParameter("c", values["c"]),
            hboms.parameter.RandomParameter("d", values["d"], lbound=None),
        ]
        corr_params = [] # TODO add correlated parameters in other test
        R = 10
        num_cats = {} # TODO add categorical covariates in other test
        
        init_dict = prepare_init(params, corr_params, R, num_cats)
        
        expected_keys = [
            "a", "b", "c", "d", "loc_b", "scale_b", "loc_d", "scale_d"
        ]
        
        # check that all expected parameters are present
        assert all([k in init_dict for k in expected_keys])
        
        # check that the values are correct
        expected_types = {"a" : float, "b" : list, "c" : list, "d" : list}
        
        for p in params:
            val = init_dict[p.name]
            assert isinstance(val, expected_types[p.name])
        
            match val:
                case float():
                    assert val == values[p.name]
                case list():
                    # correct value
                    assert all([v == values[p.name] for v in val])
                    # correct size
                    assert len(val) == R
                case _:
                    assert False, "incorrect type of value encountered"
                    
        # check that "b" has the right loc and scale
        assert init_dict["scale_b"] == scales["b"]
        
        assert init_dict["loc_b"] == np.log(values["b"])
        
        # check that "d" has the right loc and scale
        
        assert init_dict["scale_d"] == scales["d"]
        
        assert init_dict["loc_d"] == values["d"]
        


class TestSimTime():
    model_dir = os.path.join("tests", "stan-cache")

    def test_prepare_simulation_times(self):
        Time = [[1,2,3], [4,5,6,7]]
        n_sim = 10
        data = {"Time" : Time}
        SimTime = prepare_simulation_times(data, n_sim)

        assert len(SimTime) == 2, "expected simulation time lists for 2 units"

        assert len(SimTime[0]) == n_sim and len(SimTime[1]) == n_sim, "number of time points is not correct"

        assert SimTime[0][0] == Time[0][0] and SimTime[0][-1] == Time[0][-1], "simulation time end points shoud correspond to given observation times"


    def test_get_simulation_times(self):
        """
        fit a simple model and check that we get the right simulation time points,
        without having to specify the number n_sim.
        """
        a_gt = 0.2
        x0 = 0.001
        params = [hboms.Parameter("a", a_gt, "fixed")]
        state = [hboms.Variable("x")]
        obs = [hboms.Observation("X")]
        dists = [hboms.StanDist("normal", "X", ["x", "0.1"])]
        odes = "ddt_x = a*x*(1-x);"
        init = f"x_0 = {x0};"
        model = hboms.HbomsModel(
            name = "test_sim_time",
            state = state,
            odes = odes,
            init = init,
            params = params,
            obs = obs,
            dists = dists,
            model_dir = self.model_dir
        )
        Time = [[1, 2, 3, 4], [1, 2, 3, 4, 5]]
        def solved_model(t):
            return 1 / (1 + (1/x0-1)*np.exp(-a_gt*t))
        X = [list(solved_model(np.array(ts))) for ts in Time]
        data = {
            "Time" : Time,
            "X" : X
        }
        n_sim = 20
        model.sample(data, n_sim = n_sim, iter_warmup=100, iter_sampling=100, chains=1)

        SimTime = model.get_simulation_times(data)

        assert len(SimTime) == 2, "expected simulation time lists for 2 units"

        assert len(SimTime[0]) == n_sim and len(SimTime[1]) == n_sim, "number of time points is not correct"

        assert SimTime[0][0] == Time[0][0] and SimTime[0][-1] == Time[0][-1], "simulation time end points shoud correspond to given observation times"



        



