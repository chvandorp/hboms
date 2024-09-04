from . import genmodel, gensimulator, frontend, prior, covariate, parameter
from . import utilities as util
from . import definitions as defn
from .distribution import StanDist
from .parameter import Parameter, ParameterBlock, gen_corr_block
from .correlation import Correlation
from .covariate import Covariate, CatCovariate, group_covars
from . import plots
from .observation import Observation
from .state import StateVar
from . import stanlang as sl
from . import deparse
from . import optimize
from . import code_checks
from .transform import logit_transform_dispatch
from . import stanlexer # required for transformed parameters' dependencies

from cmdstanpy import (
    CmdStanModel,
    CmdStanMCMC,
    CmdStanVB,
    CmdStanPathfinder,
)  # type: ignore
import numpy as np
import scipy.linalg  # cholesky decomposition
import matplotlib.pyplot as plt  # type: ignore
import os
import networkx as nx
from typing import Optional, Mapping, Callable, Literal, Any


def compile_stan_model(
    stan_model: str, model_name: str, model_dir: str
) -> CmdStanModel:
    """
    Convert the source code of a Stan model into a CmdStanModel
    object. First write the string to a file. Then use
    cmdstanpy to compile the model.

    Parameters
    ----------
    stan_model : str
        String containing the source code of the Stan model.
    model_name : str
        The name of the Stan model. Used for filenames.
    model_dir : str
        Directory where to store the Stan source code and binary.

    Raises
    ------
    e
        Try to create the given directory. If it already exists,
        that's fine. If making the directory raises an exception,
        and the directory does not exist, then re-throw the exception.

    Returns
    -------
    CmdStanModel
        The compiled Stan model in a cmdstanpy wrapper.

    """

    ## check if model_dir exists and try to create it otherwise
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if not os.path.isdir(model_dir):
            raise e
    stan_file = os.path.join(model_dir, f"{model_name}.stan")
    ## check if file exists and has the same content as stan_model
    renew_file = True
    if os.path.isfile(stan_file):
        with open(stan_file, "r") as f:
            old_stan_model = f.read()
        if old_stan_model == stan_model:
            renew_file = False
    if renew_file:
        with open(stan_file, "w") as f:
            f.write(stan_model)
    sm = CmdStanModel(stan_file=stan_file, cpp_options={"STAN_THREADS": True})
    return sm


def map_cat_covars(data: dict, catcovs: list[CatCovariate]) -> tuple[dict, dict]:
    """
    auxiliary function used by prepare_data and prepare_data_simulator
    """
    # process and add categorical covariates
    cat_covariates = {cn.name: data[cn.name] for cn in catcovs}
    num_cat_names = {cn.name: cn.num_cat_var.name for cn in catcovs}
    # TODO: categories are known by CatCovariate class
    cat_mappings = {
        cn: {x: i + 1 for i, x in enumerate(util.unique(cv))}
        for cn, cv in cat_covariates.items()
    }
    # TODO: use the methods of CatCovariate class
    num_cats = {num_cat_names[cn]: len(mp) for cn, mp in cat_mappings.items()}
    mapped_cat_covariates = {
        cn.name: np.array([cat_mappings[cn.name][x] for x in cat_covariates[cn.name]])
        for cn in catcovs
    }
    return mapped_cat_covariates, num_cats


def get_shape_of_value(val):
    """
    Auxiliary function for deriving the shape of a non-scalar parameter
    from the given value.
    """
    match val:
        case list():
            return get_shape_of_value(val[0])
        case np.ndarray():
            shp = val.shape
            if len(shp) == 1:
                return shp[0]
            return list(shp)
        case float():
            return None


def prepare_data(
    data: dict,
    params: list[Parameter],
    obs: list[Observation],
    covs: list[Covariate],
    n_sim: int,
) -> dict:
    """
    Auxiliary function to create the right data dictionary accepted by cmdstanpy.
    Uses user supplied data, but also constant parameters and other required data.
    """
    Time = data["Time"]
    R = len(Time)
    N = [len(x) for x in Time]
    TimeSim = [list(np.linspace(t[0], t[-1], n_sim)) for t in Time]
    # create data dict for Stan
    stan_data = {
        "R": R,
        "N": N,
        "Time": util.apply_padding(Time),
        "NSim": n_sim,
        "TimeSim": TimeSim,
    }
    # add observations
    stan_data.update(
        {
            ob.name: util.apply_padding(
                data[ob.name], sdtype=deparse.deparse_decl(ob.obs_type)
            )  # FIXME: better solution
            for ob in obs
        }
    )
    # add censoring codes
    stan_data.update(
        {
            ob.cc_name: util.apply_padding(data[ob.cc_name], sdtype="int")
            for ob in obs
            if ob.censored
        }
    )
    # add constant parameters
    constants = {}
    for p in params:
        match (p.get_type(), p.value):
            case ("const", _):
                constants[p.name] = p.value
            case ("const_indiv", list()):
                constants[p.name] = p.value
            case ("const_indiv", float() | np.ndarray()):
                constants[p.name] = [p.value for _ in range(R)]
            case ("const_indiv", _):
                raise Exception(
                    f"invalid value type '{type(p.value)}' of const_indiv parameter"
                )
    stan_data.update(constants)

    # add shapes of non-scalar parameters
    shapes = {
        f"shape_{p.name}": get_shape_of_value(p.value)
        for p in params
        if p.space != "real"
    }
    stan_data.update(shapes)

    # add covariates
    for cov in covs:
        if cov.name not in data:
            raise Exception(f"covariate '{cov.name}' is missing from data dictionary")

    cont_covs, cat_covs = group_covars(covs)
    cont_covariates = {cov.name: data[cov.name] for cov in cont_covs}
    stan_data.update(cont_covariates)
    vv_cont_dims = {
        cov.dim_var.name: cov.dim for cov in cont_covs if cov.dim is not None
    }

    stan_data.update(vv_cont_dims)

    # use aux function to map categorical covariates
    mapped_cat_covariates, num_cats = map_cat_covars(data, cat_covs)

    # and add results to the data dict
    stan_data.update(mapped_cat_covariates)
    stan_data.update(num_cats)

    # add additional data provided in the data dict, but skip the "ID" key
    addl_keys = [k for k in data.keys() if k not in stan_data.keys() and k != "ID"]
    for k in addl_keys:
        stan_data[k] = data[k]

    return stan_data


def infer_loc(
    values: float | np.ndarray | list[float | np.ndarray], transform: Callable
) -> float | np.ndarray:
    """
    Infer a reasonable location parameter from a list of samples.
    First apply transform to the samples, then take the mean.
    If a single sample is given, then just return this value
    (after applying the transform)

    Parameters
    ----------
    values : list[float | np.ndarray]
        individual samples.
    transform : Callable
        transformation function to apply to the samples.

    Returns
    -------
    loc : float | np.ndarray
        inferred location parameter

    """
    match values:
        case list():
            tvalues = [transform(val) for val in values]
            return np.mean(tvalues, axis=0)
        case int() | float() | np.ndarray():
            return transform(values)
        case _:
            raise Exception(f"invalid value type '{type(values)}' for infer_loc")


def prepare_data_simulator(
    data: dict,
    params: list[Parameter],
    corr_params: list[ParameterBlock],
    covs: list[Covariate],
) -> dict:
    Time = data["Time"]
    R = len(Time)
    N = [len(x) for x in Time]
    # create data dict for Stan
    stan_data = {
        "R": R,
        "N": N,
        "Time": util.apply_padding(Time),
    }

    # add covariates
    for cov in covs:
        if cov.name not in data:
            raise Exception(f"covariate '{cov.name}' is missing from data dictionary")
    cont_covs, cat_covs = group_covars(covs)
    covariate_data = {cov.name: data[cov.name] for cov in covs}
    stan_data.update(covariate_data)
    vv_cont_dims = {
        cov.dim_var.name: cov.dim for cov in cont_covs if cov.dim is not None
    }
    stan_data.update(vv_cont_dims)

    # use aux function to map categorical covariates
    mapped_cat_covariates, num_cats = map_cat_covars(data, cat_covs)

    # and add results to the data dict
    stan_data.update(mapped_cat_covariates)
    stan_data.update(num_cats)

    # parameter values
    for p in params:
        match p.get_type():
            case "const":
                stan_data[p.name] = p.value
            case "fixed":
                if len(p._catcovs) == 0:
                    stan_data[p.name] = p.value
                else:
                    vals = p._cw_values.get(
                        p._catcovs[0].name, p._catcovs[0].loc_value(p.value)
                    )
                    stan_data[p.name] = vals
            case "const_indiv" | "indiv":
                match p.value:
                    case list():
                        stan_data[p.name] = p.value
                    case float() | np.ndarray():
                        stan_data[p.name] = [p.value for _ in range(R)]
                    case _:
                        raise Exception(
                            f"invalid value type '{type(p.value)}' of const_indiv parameter"
                        )
            case "random":
                transform = logit_transform_dispatch(p.lbound, p.ubound)
                loc_val = infer_loc(p.value, transform)
                if len(p._catcovs) == 0:
                    stan_data[f"loc_{p.name}"] = loc_val
                else:
                    locs = p._cw_values.get(
                        p._catcovs[0].name, p._catcovs[0].loc_value(loc_val)
                    )
                    stan_data[f"loc_{p.name}"] = np.array(locs)

                stan_data[f"scale_{p.name}"] = p.scale_value
                # scale for random levels
                if p.level is not None and p.level_type == "random":
                    stan_data[f"scale_{p.name}_{p.level.name}"] = p.level_scale_value

                # covariate weights
                for cov in p._contcovs:
                    w = p._cw_values.get(cov.name, cov.weight_value())
                    stan_data[cov.weight_name(p.name)] = w
            case "trans" | "trans_indiv":
                pass # do nothing!
            case _:
                raise Exception("invalid parameter type")

    # add shapes of non-scalar parameters
    shapes = {
        f"shape_{p.name}": get_shape_of_value(p.value)
        for p in params
        if p.space != "real"
    }
    stan_data.update(shapes)

    # correlation matrices
    for p in corr_params:
        stan_data[f"corr_{p.name}"] = p.corr_value

    return stan_data


def prepare_init(
    params: list[Parameter],
    corr_params: list[ParameterBlock],
    R: int,
    numcats: dict[str, int],
) -> dict[str, float | int | np.ndarray]:
    uncorr_params = [p for p in params if not any([(p in cp) for cp in corr_params])]

    init_dict: Mapping[str, float | int | np.ndarray] = {}

    # uncorrelated random effects, fixed parameters, indiv parameters
    for p in uncorr_params:
        value = None  # None means that we're not assigning this in the init_dict
        match p.get_type():
            case "random":
                match p.value:
                    case int() | float() | np.ndarray():
                        has_fixed_level = (
                            p.level is not None and p.level_type == "fixed"
                        )
                        num_params = numcats[p.level.name] if has_fixed_level else R
                        value = [p.value for _ in range(num_params)]
                    case list():
                        value = p.value
                    case _:
                        raise Exception(
                            f"invalid value type '{type(p.value)}' for parameter '{p.name}'"
                        )
            case "indiv":
                match p.value:
                    case float() | np.ndarray():
                        value = [p.value for _ in range(R)]
                    case list():
                        value = p.value
                    case _:
                        raise Exception(
                            f"invalid value type '{type(p.value)}' for parameter '{p.name}'"
                        )
            case "fixed":
                if len(p._catcovs) == 0:
                    value = p.value
                else:
                    value = p._cw_values.get(
                        p._catcovs[0].name, p._catcovs[0].loc_value(p.value)
                    )
            case "const" | "const_indiv" | "trans" | "trans_indiv":
                pass  # keep value equal to None
            case _:
                raise Exception(
                    f"unknown parameter type '{p.get_type()}' for parameter '{p.name}'"
                )
        # only assign the value if we found a non-const parameter
        if value is not None:
            init_dict[p.name] = value

    # location of uncorrelated random effects (with no categorical covariates)
    init_dict.update(
        {
            f"loc_{p.name}": infer_loc(
                p.value, logit_transform_dispatch(p.lbound, p.ubound)
            )
            for p in uncorr_params
            if p.get_type() == "random" and len(p._catcovs) == 0
        }
    )
    ## idem for parameters WITH categorical covariates
    ## FIXME: handle case for parameters with multiple categorical covariates
    init_dict.update(
        {
            f"loc_{p.name}": np.array(
                p._cw_values.get(
                    p._catcovs[0].name,
                    p._catcovs[0].loc_value(
                        infer_loc(p.value, logit_transform_dispatch(p.lbound, p.ubound))
                    ),
                )
            )
            for p in uncorr_params
            if p.get_type() == "random" and len(p._catcovs) > 0
        }
    )
    # scale of uncorrelated random effects
    init_dict.update(
        {
            f"scale_{p.name}": p.scale_value
            for p in uncorr_params
            if p.get_type() == "random"
        }
    )
    # correlated random effects
    init_dict.update(
        {f"block_{p.name}": np.full((R, len(p)), p.value) for p in corr_params}
    )
    # mean of correlated random effects
    # TODO: categorical covariates for correlated parameters
    # Notice that p.value for ParameterBlock p is already on the unconstraint scale!
    init_dict.update(
        {f"loc_{p.name}": infer_loc(p.value, lambda x: x) for p in corr_params}
    )
    # scale of correlated random effects
    init_dict.update({f"scale_{p.name}": p.scale_value for p in corr_params})
    # cholesky correlation matrix of correlated random effects
    # NB: Stan uses lower-triangle matrices L
    # TODO: catch errors (complex numbers etc) in decomposition
    init_dict.update(
        {
            f"chol_{p.name}": scipy.linalg.cholesky(p.corr_value, lower=True)
            for p in corr_params
        }
    )
    # set initial guesses for continuous covariate weights
    init_dict.update(
        {
            cov.weight_name(p.name): p._cw_values.get(cov.name, cov.weight_value())
            for p in params
            for cov in p._contcovs
            if p.get_type() == "random"
        }
    )

    return init_dict


StanInferenceMethod = Literal["sample", "variational", "pathfinder"]


def fit_stan_model(
    sm: CmdStanModel,
    data: dict,
    params: list[Parameter],
    corr_params: list[ParameterBlock],
    obs: list[Observation],
    covs: list[Covariate],
    n_sim: int,
    method: StanInferenceMethod,
    **kwargs,
) -> CmdStanMCMC | CmdStanVB | CmdStanPathfinder:
    """
    Interface function to cmdstanpy.

    Takes a CmdStanModel and calls the sampling method with the correct arguments
    In addition to the sampling method, we can also choose `variational`

    Parameters
    ----------
    sm : CmdStanModel
        the compiled cmdstanpy model.
    data : dict
        dictionary with the data.
    params : list[Parameter]
        list of all model parameters.
    corr_params : list[ParameterBlock]
        list of correlated parameters.
    obs : list[Observation]
        list of observations.
    covs : list[Covariate]
        list of covariates.
    n_sim : int
        number of time points for simulatio.
    method : StanInferenceMethod
        determines which cmdstanpy method is called. Must be "sample",
        "variational", "pathfinder", ...

    Returns
    -------
    CmdStanMCMC | CmdStanVB | CmdStanPathfinder
        Samples from the (approximate) posterior as a cmdstanpy object.
    """
    prep_data = prepare_data(data, params, obs, covs, n_sim)
    _, cat_covs = group_covars(covs)
    num_cats = {cov.name: prep_data[cov.num_cat_var.name] for cov in cat_covs}
    num_units = prep_data["R"]
    init_dict = prepare_init(params, corr_params, num_units, num_cats)
    match method:
        case "sample":
            sam = sm.sample(data=prep_data, inits=init_dict, **kwargs)
        case "variational":
            sam = sm.variational(data=prep_data, inits=init_dict, **kwargs)
        case "pathfinder":
            sam = sm.pathfinder(data=prep_data, inits=init_dict, **kwargs)
        case _:
            raise Exception(
                f"invalid Stan inference method {method}."
                "Choose between 'sample', 'variational', and 'pathfinder'"
            )
    return sam


def simulate_stan_model(
    sm: CmdStanModel,
    data: dict,
    params: list[Parameter],
    corr_params: list[ParameterBlock],
    covs: list[Covariate],
    obs: list[Observation],
    num_simulations: int,
    output_dir: Optional[str],
    seed: Optional[int],
) -> list[tuple[dict, dict]]:
    prep_data = prepare_data_simulator(data, params, corr_params, covs)
    sam = sm.sample(
        data=prep_data,
        output_dir=output_dir,
        fixed_param=True,
        show_progress=False,
        chains=1,
        adapt_engaged=False,
        iter_sampling=num_simulations,
        seed=seed,
    )
    N = prep_data["N"]
    simulations = [
        {
            ob.name: [
                list(sam.stan_variable(ob.name)[i, r, :Nr, ...])
                for r, Nr in enumerate(N)
            ]
            for ob in obs
        }
        for i in range(num_simulations)
    ]
    # add other data to each simulations dict for convenience
    for sim in simulations:
        sim.update(data)
    # extract simulated random effects
    random_param_draws = [
        {
            p.name: sam.stan_variable(p.name)[i, ...]
            for p in params
            if p.get_type() == "random"
        }
        for i in range(num_simulations)
    ]
    return list(zip(simulations, random_param_draws))


DEFAULT_OPTIONS = {
    "integrator": "ode_rk45",
    "rel_tol": 1e-6,
    "abs_tol": 1e-6,
    "max_num_steps": 1000000,
    "init_obs": False,
}


def complete_options(options: dict | None) -> dict:
    """
    Take user-defined options and check if they are valid.
    Add default values for any missing fields.

    Parameters
    ----------
    options : dict
        User-defined options.

    Returns
    -------
    dict
        The complemented options dictionary.

    """
    if options is None:
        return DEFAULT_OPTIONS
    # else...
    # check that keys are correct
    for k in options.keys():
        if k not in DEFAULT_OPTIONS:
            raise ValueError(f"invalid option {k} given")
    compl_options = options
    for k, v in DEFAULT_OPTIONS.items():
        if k not in options:
            compl_options[k] = v
    # check integrator
    if compl_options["integrator"] not in defn.supported_integrators:
        x = compl_options["integrator"]
        raise NotImplementedError(f"integrator {x} not supported")

    return compl_options


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

        """
        TODO: verify correctness of user input.
        TODO: move some of the "compilation" code to a separate function
        """

        # parse covariates before parameters
        covariates = None
        covar_dict = {}
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
            if p.loc_prior is not None:
                par_kwargs["loc_prior"] = prior.Prior(
                    p.loc_prior.name, p.loc_prior.params
                )
            if p.scale_prior is not None:
                par_kwargs["scale_prior"] = prior.Prior(
                    p.scale_prior.name, p.scale_prior.params
                )
            if p.prior is not None:
                par_kwargs["prior"] = prior.Prior(p.prior.name, p.prior.params)
            if p.level is not None:
                # TODO: test that the covariate exists and that it is categorical
                cov = covar_dict[p.level]
                par_kwargs["level"] = cov
                par_kwargs["level_type"] = p.level_type
                par_kwargs["level_scale"] = p.level_scale
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

    @property
    def model_code(self) -> str:
        """return the stan model code as a string"""
        return self._model_code

    @property
    def simulator_code(self) -> Optional[str]:
        """return the stan simulator code as a string"""
        return self._simulator_code

    @property
    def fit(self) -> CmdStanMCMC | CmdStanVB:
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

    def stan_data(self, data: dict, n_sim: int = 100) -> dict:
        """
        Make a dictionary required to fit the model. The input is the data
        required for the model. This method adds a number of required items,
        like constants, the number of units, simulation time points. etc.
        """
        # TODO: also return initial parameter values

        return prepare_data(data, self.params, self.obs, self.covs, n_sim)

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

    def set_init(self, init_dict: dict[str, Any]) -> None:
        """
        Set initial parameter guesses to provided values.

        Parameters
        ----------
        init_dict : dict[str, Any]
            dictionary of initial parameter values.

        """
        for p in self.params:
            val = init_dict.get(p.name)
            if val is None:
                continue
            p.value = val

            # FIXME: this does not handle correlated parameters!
