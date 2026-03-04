from . import genmodel, gensimulator, frontend, prior, covariate, parameter
from .distribution import StanDist
from .parameter import gen_corr_block
from .correlation import Correlation
from .covariate import group_covars
from . import plots
from .observation import Observation
from .state import StateVar
from . import stanlang as sl
from . import deparse
from . import optimize
from . import code_checks
from . import stanlexer # required for transformed parameters' dependencies
from . import name_checks

from .model_helper_funcs import (
    compile_stan_model,
    fit_stan_model,
    simulate_stan_model,
    prepare_data,
    prepare_init,
    prepare_simulation_times,
    complete_options,
    StanInferenceMethod
)

from cmdstanpy import (
    CmdStanModel,
    CmdStanMCMC,
    CmdStanVB,
    CmdStanPathfinder,
)  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import os
import networkx as nx
from typing import Optional, Any



class HbomsModel:
    """
    Object that represents an HBOMS model

    This serves as an interface to the actual Stan model
    """

    def __init__(
        self,
        name: str,
        state: list[frontend.Variable],
        odes: str,
        init: str,
        params: list[frontend.Parameter],
        obs: list[frontend.Observation],
        dists: list[frontend.StanDist],
        trans_state: Optional[list[frontend.Variable]] = None,
        transform: Optional[str] = None,
        covariates: Optional[list[frontend.Covariate]] = None,
        correlations: Optional[list[frontend.Correlation]] = None,
        plugin_code: Optional[dict[str, str]] = None,
        options: Optional[dict] = None,
        compile_model: bool = True,
        optimize_code: bool = True,
        model_dir: Optional[str] = None,
    ) -> None:
        """
        Build an HBOMS model. This means generating the Stan source code
        and compiling the Stan model.

        Parameters
        ----------
        name : str
            The name of the model. Used for filenames
        state : list[frontend.Variable]
            A list of variables of the ODE system. Use the `Variable` class to
            create them.
        odes : str
            The system of ODEs written in the Stan language. Write an equation
            for each variable. Example: if you have defined a variable called
            `x`, then the time derivative is called `ddt_x`.
            Then write a line like "ddt_x = a - b * x;", where `a` and `b` are
            parameters.
        init : str
            Initial conditions of the IVP written in the Stan language.
            Write an equation for each variable. Example: if you have defined
            a variable called `x`, then the initial value is called `x_0`.
            Then write a line like "x_0 = a / b;", where `a` and `b` ar
            parameters.
        params : list[frontend.Parameter]
            A list of the model parameters. Use the `Parameter` class to define
            each parameter.
        obs : list[frontend.Observation]
            A list of observations. These have to correspond to the data that
            the model is going to be fit to. Use the `Observarion` class to
            define each observation.
        dists : list[frontend.StanDist]
            A list of distributions. These define the observation model that
            links the ODE predictions to the observations. Use the `StanDist`
            object to define each distribution. Example: if we have a variable
            `x` and a corresponding observation `X`, and we assume that `X` has
            a normal distribution with mean `x` and standard deviation `sigma`,
            then we add a distribution `StanDist("normal", "X", ["x", "sigma"])`.
            Here `sigma` must be added to the `params` list.
        trans_state : Optional[list[frontend.Variable]], optional
            A list of additional variables that are calculated with the
            parameters and the state variables. The default is None.
        transform : Optional[str], optional
            Stan code for computing the transformed state. The default is None.
        covariates : Optional[list[frontend.Covariate]], optional
            A list of covariates. Use the `Covariate` class to define them.
            The default is None.
        correlations : Optional[list[frontend.Correlation]], optional
            A list of correlations between parameters. Use the `Correlation`
            class to define them. The default is None.
        plugin_code : Optional[dict[str, str]], optional
            A dictionary with plugin code, i.e. code that is inserted in the
            code blocks as-is. The keys correspond to the names of the code
            blocks in the Stan model.
        options : Optional[dict], optional
            A dictionary with additional options. Such as the ODE integrator
            and telerance parameters. The default is None.
        compile_model : bool, optional
            If this is set to `False`, the Stan model is not compiled.
            This is for debugging purposes. The default is True.
        optimize_code : bool, optional
            If this is set to `False`, the Stan code is not optimized.
            For instance, expressions like `1+2` are not simplified to `3`.
            This is for debugging purposes. The default is True.
        model_dir : Optional[str], optional
            Directory for storing the Stan model files. If `None`, the current
            working directory is used. The default is None.

        """
        self._model_def = frontend.HbomsModelDef(
            name,
            state,
            odes,
            init,
            params,
            obs,
            dists,
            trans_state,
            transform,
            covariates,
            correlations,
        )

        # Verify correctness of user input.

        # check validity of names
        name_checks.check_names( ## TODO: include more components: correlations, trans_state
            [p.name for p in params],
            [s.name for s in state],
            [c.name for c in (covariates or [])], # covariates can be None
            [o.name for o in obs]
        )

        # use auxiliary function to "compile" the model. 
        # This uses self._model_def
        params, obs, dists, state, trans_state, covariates, correlations = self._compile_hboms_model()

        # now store compiled objects as attributes
        self._model_name = name
        self._dists = dists
        self._params = params
        self._state = state
        self._obs = obs
        # optional state transform
        self._trans_state = trans_state if trans_state is not None else []
        self._options = complete_options(options)
        ## optional correlations
        if correlations is None:
            self._corr_params = []  # no parameters are correlated
        else:
            self._corr_params = [
                gen_corr_block(corr, self._params) for corr in correlations
            ]
        # optional covariates
        self._covs = covariates if covariates is not None else []

        # check user code
        state_var_names = [s.name for s in self._state]
        code_checks.verify_odes(odes, state_var_names)  ## shows warning if not OK
        code_checks.verify_init(init, state_var_names)
        # check transform
        transform = "" if transform is None else transform
        if len(self._trans_state) > 0:
            trans_var_names = [s.name for s in self._trans_state]
            code_checks.verify_transform(transform, trans_var_names)

        # TODO: replace mixins with parsed and resolved ASTs
        self._odes = sl.MixinStmt(odes)
        self._init = sl.MixinStmt(init)
        self._transform = sl.MixinStmt(transform)

        # check that keys in plugin_code are correct
        plugin_code = {} if plugin_code is None else plugin_code
        invalid_keys = [k for k in plugin_code.keys() if k not in sl.model_block_names]
        if invalid_keys:
            message = "Found keys in plugin code dictionary that do not correspond to "
            message += "Stan model blocks: " + ", ".join(invalid_keys) + ". "
            raise Exception(message)

        # TODO: veryfy that plugin code parses. Check other issues?
        self._plugin_code = {k : sl.MixinStmt(v) for k, v in plugin_code.items()}

        # generate the stan model
        AST: sl.Stmt = genmodel.gen_stan_model(
            self._odes,
            self._init,
            self._dists,
            self._params,
            self._corr_params,
            self._state,
            self._trans_state,
            self._transform,
            self._obs,
            self._covs,
            self._plugin_code,
            self._options,
        )
        # optionally optimize the AST
        self._optimize_code = optimize_code
        if optimize_code:
            AST = optimize.optimize_stmt(AST)
        self._AST = AST
        self._model_code = deparse.deparse_stmt(AST)
        # set model directory
        if model_dir is None:
            self._model_dir = os.getcwd()
        else:
            self._model_dir = model_dir
        # compile the stan model
        self._stan_model: CmdStanModel | None = None
        if compile_model:
            self._stan_model = compile_stan_model(
                self._model_code, self._model_name, self._model_dir
            )

        # create field that can contain the fit
        self._fit: CmdStanMCMC | CmdStanVB | CmdStanPathfinder | None = None
        # create a field that can contain the simulator code
        self._simulator_code: str | None = None
        self._stan_simulator: CmdStanModel | None = None
        # create a field that can contain the prior sampler code
        self._prior_sampler_code: str | None = None
        self._stan_prior_sampler: CmdStanModel | None = None
        # create a field that can contain the prior predictive fit
        self._prior_fit: CmdStanMCMC | None = None

    def _compile_hboms_model(self):
        """
        function used by __init__ to compile the self_model_def
        into usable components.
        """
        # parse covariates before parameters
        covariates: list[covariate.Covariate] | None = None
        covar_dict: dict[str, covariate.Covariate] = {}
        if self._model_def.covariates is not None:
            for cov in self._model_def.covariates:
                cov_kwargs = {"categories": cov.categories, "dim": cov.dim}
                # remove kwargs with None values
                cov_kwargs = {k: v for k, v in cov_kwargs.items() if v != None}

                covar_dict[cov.name] = covariate.covar_dispatch(
                    cov.name, cov.cov_type, **cov_kwargs
                )

            covariates = [covar_dict[cov.name] for cov in self._model_def.covariates]
        
        params = [None for _ in self._model_def.params]
        for i, p in enumerate(self._model_def.params):
            if p.par_type in ["trans", "trans_indiv"]:
                continue # wait with creating transformed parameters
            covs = None
            if p.covariates is not None:
                covs = [covar_dict[cov] for cov in p.covariates]
            par_kwargs = {
                "scale": p.scale,
                "covariates": covs,
                "space": p.space,
                "cw_values": p.cw_values,
            }
            # convert Prior objects

            priors = {"loc_prior" : p.loc_prior, "scale_prior": p.scale_prior, "prior": p.prior}
            # add level_scale_prior if needed
            if p.level is not None:
                priors["level_scale_prior"] = p.level_scale_prior
            # process priors
            for key, prior_obj in priors.items():
                match prior_obj:
                    case None:
                        pass
                    case frontend.StanPrior(priorname, hypparams):
                        par_kwargs[key] = prior.AbsContPrior(
                            priorname, hypparams
                        )
                    case frontend.DiracDeltaPrior(hypparam):
                        par_kwargs[key] = prior.DiracDeltaPrior(hypparam)
                    case _:
                        raise ValueError(f"invalid prior type for {key} of parameter {p.name}")
                    
            if p.level is not None:
                if p.level not in covar_dict:
                    raise ValueError(f"level covariate '{p.level}' for parameter '{p.name}' is not defined.")
                cov = covar_dict[p.level]
                if not isinstance(cov, covariate.CatCovariate):
                    raise ValueError(f"level covariate '{p.level}' for parameter '{p.name}' must be categorical.")
                par_kwargs["level"] = cov
                par_kwargs["level_type"] = p.level_type
                par_kwargs["level_scale"] = p.level_scale
                # level scale prior is already handled above

            if p.noncentered is not None:
                par_kwargs["noncentered"] = p.noncentered

            # remove kwargs with None values
            par_kwargs = {k: v for k, v in par_kwargs.items() if v is not None}
            params[i] = parameter.param_dispatch(
                p.name,
                p.value,
                p.par_type,
                lbound=p.lbound,
                ubound=p.ubound,
                **par_kwargs,
            )

        # figure out dependencies between transformed parameters
        parnames = [p.name for p in self._model_def.params]
        trans_param_defs = [
            p for p in self._model_def.params
            if p.par_type in ["trans", "trans_indiv"]
        ]
        trans_param_names = [p.name for p in trans_param_defs]
        trans_param_dependencies = [
            stanlexer.find_used_names(p.value, parnames) 
            for p in trans_param_defs
        ]
        dept_graph = nx.DiGraph()
        for p, deps in zip(trans_param_defs, trans_param_dependencies):
            for d in deps:
                dept_graph.add_edge(d, p.name)
        # check that there are no loops in the dependency graph
        if not nx.is_directed_acyclic_graph(dept_graph):
            cycle = nx.find_cycle(dept_graph)
            cycle_nodes = [e[0] for e in cycle] + [cycle[0][0]]
            cycle_str = " -> ".join(cycle_nodes)
            raise Exception(f"dependency graph of transformed parameters contains cycle ({cycle_str})")
        # else, walk through the parameters in topological order.
        # if no dependencies, keep user-defined order
        top_sort = nx.lexicographical_topological_sort(dept_graph, key=lambda x: parnames.index(x))
        for rank, pn in enumerate(top_sort):
            if pn not in trans_param_names:
                continue # skip "regular" (i.e. non-transformed) parameters
            dep_pars = [params[parnames.index(dn)] for dn in dept_graph.predecessors(pn)]
            idx = parnames.index(pn)
            p = self._model_def.params[idx]
            params[idx] = parameter.param_dispatch(
                p.name,
                p.value,
                p.par_type,
                lbound=p.lbound,
                ubound=p.ubound,
                dependencies=dep_pars,
                rank=rank
            )

        # we need to have access of observations by name to compile distributions
        obs_dict = {
            ob.name: Observation(
                ob.name,
                obs_type=frontend.type_dispatch(ob.data_type),
                censored=ob.censored,
            )
            for ob in self._model_def.obs
        }
        obs = [obs_dict[ob.name] for ob in self._model_def.obs]
        ## TODO: support for more general Stan types! (dim takes care of vector)
        state = [StateVar(var.name, dim=var.dim) for var in self._model_def.state]
        dists = [
            StanDist(
                dist.name,
                obs_dict[dist.obs_name],
                *(sl.MixinExpr(x) for x in dist.params),
            )
            for dist in self._model_def.dists
        ]
        trans_state = None
        if self._model_def.trans_state is not None:
            trans_state = [
                StateVar(var.name, dim=var.dim) for var in self._model_def.trans_state
            ]
        correlations = None
        if self._model_def.correlations is not None:
            correlations = []
            for corr in self._model_def.correlations:
                corr_kwargs = {"value": corr.value, "intensity": corr.intensity}
                corr_kwargs = {k: v for k, v in corr_kwargs.items() if v is not None}
                correlations.append(Correlation(parnames=corr.params, **corr_kwargs))
        
        # return requires results
        return params, obs, dists, state, trans_state, covariates, correlations


    @property
    def model_code(self) -> str:
        """return the stan model code as a string"""
        return self._model_code

    @property
    def simulator_code(self) -> Optional[str]:
        """return the stan simulator code as a string"""
        return self._simulator_code

    @property
    def fit(self) -> CmdStanMCMC | CmdStanVB | CmdStanPathfinder:
        """
        Get access to the fitted Stan model. Make sure to call the method
        `sample`, `variational` or `pathfinder` first.

        Returns
        -------
        CmdStanMCMC | CmdStanVB | CmdStanPathfinder
            The fitted Stan model.

        """
        if self._fit is None:
            raise Exception(
                f"HbomsModel {self._model_name} has not been fit to data yet."
                "First use method one of the methods HbomsModel.sample(),"
                "HbomsModel.variational() or HbomsModel.pathfinder()."
            )
        return self._fit
    
    @property
    def prior_fit(self) -> CmdStanMCMC:
        """
        Get access to the prior predictive fit of the Stan model. Make sure
        to call the method `sample_from_prior` first.
        TODO: perhaps prior_fit is not a good name as we're not fitting to
        any data here...

        Returns
        -------
        CmdStanMCMC
            The prior predictive fit of the Stan model.

        """
        if self._prior_fit is None:
            raise Exception(
                f"HbomsModel {self._model_name} has not been sampled from the prior yet."
                "First use method HbomsModel.sample_from_prior()."
            )
        return self._prior_fit

    def stan_data_and_init(self, data: dict, n_sim: int = 100) -> tuple[dict, dict]:
        """
        Make a dictionary required to fit the model. The input is the data
        required for the model. This method adds a number of required items,
        like constants, the number of units, simulation time points. etc.
        The method also returns a dictionary with initial parameter values.

        Parameters
        ----------
        data : dict
            A dictionary with the data. Must contain at least a field "Time"
            and fields for the observations.
        n_sim : int, optional
            The number of time points for the simulation. This is used to
            determine the number of time points for the ODE system. The default
            is 100.

        Returns
        -------
        tuple[dict, dict]
            A tuple with two dictionaries. The first dictionary contains the
            data required for the Stan model. The second dictionary contains
            the initial parameter values for the Stan model.
        """
        data_dict = prepare_data(data, self._params, self._obs, self._covs, n_sim)
        _, cat_covs = group_covars(self._covs)
        num_cats = {cov.name: data_dict[cov.num_cat_var.name] for cov in cat_covs}
        num_units = data_dict["R"]
        init_dict = prepare_init(self._params, self._corr_params, num_units, num_cats)

        return data_dict, init_dict
    

    def _run(
        self, method: StanInferenceMethod, data: dict, n_sim: int, **kwargs
    ) -> None:
        """general inference method. used by 'sample' and other methods below"""
        self._fit = fit_stan_model(
            self._stan_model,
            data,
            self._params,
            self._corr_params,
            self._obs,
            self._covs,
            n_sim,
            method,
            **kwargs,
        )

    def sample(self, data: dict, n_sim: int = 100, **kwargs) -> None:
        """generate posterior samples from the Stan model using HMC"""
        self._run("sample", data, n_sim, **kwargs)

    def variational(self, data: dict, n_sim: int = 100, **kwargs) -> None:
        self._run("variational", data, n_sim, **kwargs)

    def pathfinder(self, data: dict, n_sim: int = 100, **kwargs) -> None:
        self._run("pathfinder", data, n_sim, **kwargs)

    def simulate(
        self,
        data: dict,
        num_simulations: int,
        compile_simulator: bool = True,
        output_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> list[tuple[dict, dict]]:
        """
        Simulate data using the model. Returns a list of pairs of
        simulated data sets and corresponding random parameter draws.
        The simulated data sets can be used as-is to fit the model.

        Parameters
        ----------
        data : dict
            dataset used for simulation. Requires at least a field "Time",
            but for more complex models more data is needed (e.g. covariates).
        num_simulations : int
            Determines the number of simulated datasets that will be returned.
            Should be at least 1.
        compile_simulator : bool, optional
            If False, don't compile the simulator code. This is used for debugging.
            The default is True.
        output_dir : Optional[str], optional
            Determines the directory where the Stan files are stored. If None,
            use a temporary folder. The default is None.
        seed: Optional[int], optional
            Seed passed to cmdstan to ensure reproducibility. If None, a random
            seed is used. The default is None.

        Returns
        -------
        list[tuple[dict, dict]]
            A list of pairs. Each pair contains a data set and the random
            parameters used to create the data set.

        """

        AST = gensimulator.gen_stan_model(
            self._odes,
            self._init,
            self._dists,
            self._params,
            self._corr_params,
            self._state,
            self._trans_state,
            self._transform,
            self._obs,
            self._covs,
            self._options,
        )
        if self._optimize_code:
            AST = optimize.optimize_stmt(AST)
        self._simulator_code = deparse.deparse_stmt(AST)
        if not compile_simulator:
            return []
        # else, go ahead and compile and simulate
        simulator_name = self._model_name + "_simulator"
        self._stan_simulator = compile_stan_model(
            self._simulator_code, simulator_name, self._model_dir
        )
        simulations = simulate_stan_model(
            self._stan_simulator,
            data,
            self._params,
            self._corr_params,
            self._covs,
            self._obs,
            num_simulations,
            output_dir,
            seed,
        )
        return simulations
    

    def sample_from_prior(
        self,
        data: dict,
        n_sim: int = 100,
        compile_prior_sampler: bool = True,
        **kwargs,
    ) -> None:
        """
        Sample from the prior distribution of the model. This is done
        by compiling a model that ignores the observations and only
        simulates from the prior.
        To facilitate efficient sampling from the prior,
        we force all random parameters to be non-centered.

        Parameters
        ----------
        data : dict
            A dictionary with the data. Must contain at least a field "Time"
            and fields for the observations and covariates.
        n_sim : int, optional
            The number of time points for the simulation. The default is 100.
        compile_prior_sampler : bool, optional
            If False, don't compile the prior sampling code. This is used for debugging.
            The default is True.
        **kwargs
            Additional arguments passed to cmdstanpy.

        """

        # keep track of which parameters were originally non-centered
        original_noncentered = {}
        for p in self._params:
            if p.get_type() == "random":
                original_noncentered[p.name] = p.noncentered
                p.noncentered = True
        
        AST = genmodel.gen_stan_model(
            self._odes,
            self._init,
            self._dists,
            self._params,
            self._corr_params,
            self._state,
            self._trans_state,
            self._transform,
            self._obs,
            self._covs,
            self._plugin_code,
            self._options,
            include_likelihood=False,
        )
        if self._optimize_code:
            AST = optimize.optimize_stmt(AST)
        self._prior_sampler_code = deparse.deparse_stmt(AST)
        if not compile_prior_sampler:
            return []
        # else, go ahead and compile and simulate
        prior_sampler_name = self._model_name + "_prior_sampler"
        self._stan_prior_sampler = compile_stan_model(
            self._prior_sampler_code, prior_sampler_name, self._model_dir
        )

        # next, sample from the prior
        self._prior_fit = fit_stan_model(
            self._stan_prior_sampler,
            data,
            self._params,
            self._corr_params,
            self._obs,
            self._covs,
            n_sim,
            method="sample",
            **kwargs,
        )

        # finally, reset the non-centered attributes of the parameters
        for p in self._params:
            if p.name in original_noncentered:
                p.noncentered = original_noncentered[p.name]


    def _select_state_vars(self, state_var_names: list[str] | None = None):
        ext_state = self._state + self._trans_state
        if state_var_names is None:
            plot_state = ext_state
        else:
            plot_state = [
                next(filter(lambda x: x.name == name, ext_state))
                for name in state_var_names
            ]
        return plot_state

    def _select_obs(self, obs_names: list[str] | None = None):
        if obs_names is None:
            plot_obs = self._obs
        else:
            plot_obs = [
                next(filter(lambda x: x.name == name, self._obs)) for name in obs_names
            ]
        return plot_obs

    def init_check(
        self,
        data: dict,
        state_var_names: list[str] | None = None,
        obs_names: list[str] | None = None,
        n_sim: int = 100,
        **kwargs,
    ) -> plt.Figure:
        """
        Check the initial parameter guess. Plot the given data alongside the
        trajectories based on the initial parameters.

        Parameters
        ----------
        data : dict
            A dictionary with the data. Must contain fields "Time" and fields
            for the observations.
        state_var_names : list[str] | None, optional
            Choose which trajectories to plot. If None, then plot all
            trajectories (both state and transformed state).
            The default is None.
        obs_names : list[str] | None, optional
            Choose which observations to plot. If None, then plot all
            observations. The default is None.
        n_sim : int, optional
            Number of time points used for the simulations. The default is 100.
        **kwargs
            Additional arguments passed to matplotlib.

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            a figure with a panel for each unit. Showing data and trajectories.

        """

        test_fit = fit_stan_model(
            self._stan_model,
            data,
            self._params,
            self._corr_params,
            self._obs,
            self._covs,
            n_sim,
            method="sample",
            iter_sampling=100,
            fixed_param=True,
            show_progress=False,
            chains=1,  # FIXME: there is some kind of cmdstanpy bug. report?
        )
        test_sams = test_fit.stan_variables()
        # only select certain state variables
        plot_state = self._select_state_vars(state_var_names)
        # only select certain observations
        plot_obs = self._select_obs(obs_names)

        fig = plots.plot_fits(test_sams, data, plot_state, plot_obs, **kwargs)
        return fig

    def post_pred_check(
        self,
        data: dict,
        state_var_names: Optional[list[str]] = None,
        obs_names: list[str] | None = None,
        **kwargs,
    ) -> plt.Figure:
        """plot the estimated trajectories together with the data"""
        match self._fit:
            case None:
                raise Exception(
                    "Model does not contain a fit, call the sample method first"
                )
            case CmdStanMCMC() | CmdStanPathfinder():
                sams = self._fit.stan_variables()
            case CmdStanVB():
                sams = self._fit.stan_variables(mean=False)
                # note that mean=False will be the default in the future
            case _:
                raise Exception("fit object has invalid type.")

        # only select certain state variables
        plot_state = self._select_state_vars(state_var_names)
        # only select certain observations
        plot_obs = self._select_obs(obs_names)
        fig = plots.plot_fits(sams, data, plot_state, plot_obs, **kwargs)
        return fig


    def get_simulation_times(self, data: dict, n_sim: Optional[int] = None) -> list[list[float]]:
        """
        Make a list of time points used for simulating trajectories of the model.
        This is convenient for e.g. plotting the model fits.

        Parameters
        ----------

        data : dict
            data dictionary. Must contain the key "Time"
        n_sim : Optional[int]
            number of simulation time points. If the model has been fit,
            this number can be inferred from the chain, if it is not specified
            by the user. The fefault is None

        Returns
        -------
        SimTime : list[list[float]]
            simulation time points for each unit
        
        """
        if n_sim is None:
            if self._fit is None:
                raise Exception("Model does not contain a fit. Specify the n_sim parameter manually")
            # take the first state variable (include trans_state to account for "solved" models)
            varname = (self._state + self._trans_state)[0].name
            n_sim = self._fit.stan_variable(f"{varname}_sim").shape[-1]
        return prepare_simulation_times(data, n_sim)


    def set_init(self, init_dict: dict[str, Any]) -> None:
        """
        Set initial parameter guesses to provided values.
        FIXME: this method is in development!

        Parameters
        ----------
        init_dict : dict[str, Any]
            dictionary of initial parameter values.

        """
        for p in self._params:
            val = init_dict.get(p.name)
            if val is None:
                continue
            p.value = val

        # reset the value of the parameter blocks
        # these have references to the parameters
        # and now can be updated
        for p in self._corr_params:
            p.init_value()