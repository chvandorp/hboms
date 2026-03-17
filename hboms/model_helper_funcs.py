"""
This module contains auxiliary functions for the HbomsModel class.
These functions are used for preparing the data and initial values for
the Stan model, and for calling the sampling methods of cmdstanpy.
"""

from . import utilities as util
from . import definitions as defn
from .parameter import Parameter, ParameterBlock
from .covariate import Covariate, CatCovariate, group_covars
from .observation import Observation
from . import deparse
from .transform import logit_transform_dispatch, IdentityTransform

from cmdstanpy import (
    CmdStanModel,
    CmdStanMCMC,
    CmdStanVB,
    CmdStanPathfinder,
)  # type: ignore
import numpy as np
import scipy.linalg  # cholesky decomposition
import os
from typing import Optional, Mapping, Callable, Literal



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


def find_cat_requiring_level_restriction(
    params: list[Parameter],
) -> list[tuple[str, str]]:
    """
    Find categorical covariates that require level restriction,
    i.e. that are used with parameters that have a fixed level.

    Parameters
    ----------
    params : list[Parameter]
        List of model parameters.

    Returns
    -------
    list[tuple[str, str]]
        List of categorical covariates that require level restriction.
    """
    cat_covs_levels = []

    def has_fixed_level(p: Parameter) -> bool:
        if p.get_type() == "random" and p.level is not None and p.level_type == "fixed":
            return True
        return False
    
    for p in params:
        if has_fixed_level(p) and len(p._catcovs) > 0:
            cat = p._catcovs[0].name
            lev = p.level.name
            cat_covs_levels.append((cat, lev))

    return util.unique(cat_covs_levels)



def restrict_cat_covar_to_level(data: dict, cov: str, level: str) -> list:
    """
    Auxiliary function to restrict categorical covariates
    to a given "level".
    This is used for parameters fith a fixed level and a categorical covariate.
    Raises an exception if the categorical covariate has multiple categories
    for the same value of the level variable.

    Parameters
    ----------
    data : dict
        Data dictionary.
    cov : str
        Name of the categorical covariate.
    level : str
        Name of the level variable.

    Example
    -------
    Suppose we have a categorical covariate "treatment" with categories
    "placebo" and "drug", and a level variable "group" with values
    "A", "B", and "C". Then this function checks that all units in group "A"
    have the same treatment category, and similarly for group "B" and "C".
    It then returns a list with the treatment category for each group.
    
    Returns
    -------
    dict[str, str]
        Dictionary mapping each level value to its corresponding category
    """
    cov_vals = data[cov]
    level_vals = data[level]

    unique_level_vals = util.unique(level_vals)
    restricted_cats = {}
    for lv in unique_level_vals:
        units = [i for i, x in enumerate(level_vals) if x == lv]
        cats = util.unique([cov_vals[i] for i in units])
        if len(cats) > 1:
            raise ValueError(
                f"categorical covariate '{cov}' has multiple categories "
                f"for level value '{lv}' of level '{level}':"
                f" {cats[0]}, {cats[1]} (and possibly more)"
            )
        restricted_cats[lv] = cats[0]
    return restricted_cats


def check_level_restrictions(data: dict, params: list[Parameter]) -> None:
    """
    Check if any categorical covariates require level restrictions,
    and apply these restrictions to the data dictionary.

    Parameters
    ----------
    data : dict
        Data dictionary.
    params : list[Parameter]
        List of model parameters.

    Raises
    ------
    ValueError
        If a categorical covariate has multiple categories for the same
        value of the level variable.
    """
    cat_covs_levels = find_cat_requiring_level_restriction(params)
    for cov, level in cat_covs_levels:
        try:
            restrict_cat_covar_to_level(data, cov, level)
        except ValueError as e:
            raise ValueError(f"Level restriction failed for covariate '{cov}' and level '{level}': {e}")


def map_cat_covars(data: dict, catcovs: list[CatCovariate]) -> tuple[dict, dict]:
    """
    auxiliary function used by prepare_data and prepare_data_simulator

    the order of the categories is used defined and stored in the CatCovariate objects.
    """
    # process and add categorical covariates
    cat_covariates = {cn.name: data[cn.name] for cn in catcovs}
    categories = {cn.name: cn.cats for cn in catcovs}
    cat_mappings = {
        cn: {x: i + 1 for i, x in enumerate(cv)}
        for cn, cv in categories.items()
    }
    # define mapping K_mycov -> number of categories for mycov
    num_cats = {cn.num_cat_var.name: cn.num_cats for cn in catcovs}
    # define mapping mycov -> list of integers representing the categories for each unit
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


def prepare_simulation_times(data: dict, n_sim: int) -> list[list[float]]:
    """
    For each unit, create a (dense) list of time points 
    on which to simulate the trajectories.
    """
    Time = data["Time"]
    TimeSim = [list(np.linspace(t[0], t[-1], n_sim)) for t in Time]
    return TimeSim


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
    # check that level restrictions are well-defined (raise error otherwise)
    check_level_restrictions(data, params)
    # notice that the covariates are restricted in the Stan mode (transformed data block)

    Time = data["Time"]
    R = len(Time)
    N = [len(x) for x in Time]
    TimeSim = prepare_simulation_times(data, n_sim)
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



def transform_values_noncentred_param(
    values: float | np.ndarray | list[float | np.ndarray], scale: float, transform: Callable
) -> float | np.ndarray | list[float | np.ndarray]:
    """
    Transform values for non-centered parameters. For lists,
    we subtract the mean and divide by the scale.
    For single values, we return zero.
    """
    match values:
        case list():
            tvalues = [transform(val) for val in values]
            loc = np.mean(tvalues, axis=0)
            zvalues = [(val - loc) / scale for val in tvalues]
            return zvalues
        case float() | np.ndarray():
            return np.zeros_like(values)  # return zero for single values
        case _:
            raise Exception(f"invalid value type '{type(values)}' for transform_values_noncentred_param")




def prepare_data_simulator(
    data: dict,
    params: list[Parameter],
    corr_params: list[ParameterBlock],
    covs: list[Covariate],
) -> dict:
    """
    Take a python dictionary with user-specified data,
    and adjust it so that it works for the simulator.
    
    The user has to specifty the time points for each unit,
    using the key "Time". This function adds the number of
    units (R), the number of time points per unit (N),
    and looks for covariates (these should also be given by the user).

    In the simulator code, non-random parameters are made constant and 
    added to the data dictionary. This also holds for locations
    and scales for random parameters.
    """
    # check that level restrictions are well-defined (raise error otherwise)
    check_level_restrictions(data, params)

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
                    print(p.name, locs, p._cw_values, p._catcovs[0].name)
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
                    case float() | np.ndarray():
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

    # location of random parameters (with no categorical covariates)
    # include correlated parameters: these also get a separate loc and scale
    init_dict.update(
        {
            f"loc_{p.name}": infer_loc(
                p.value, logit_transform_dispatch(p.lbound, p.ubound)
            )
            for p in params
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
            for p in params
            if p.get_type() == "random" and len(p._catcovs) > 0
        }
    )
    # scale of random parameters. Include correlated parameters
    init_dict.update(
        {
            f"scale_{p.name}": p.scale_value
            for p in params
            if p.get_type() == "random"
        }
    )
    # random effects for noncentered parameters
    # don't include correlated parameters here: these are handled separately
    uncorr_noncentered_params = [
        p for p in uncorr_params if p.get_type() == "random" and p.noncentered
    ]
    for p in uncorr_noncentered_params:
        val = init_dict[p.name] ## handled above: values expanded per-unit.
        zvalues = transform_values_noncentred_param(
            val, p.scale_value, logit_transform_dispatch(p.lbound, p.ubound)
        )
        init_dict[f"rand_{p.name}"] = zvalues

    # correlated random effects
    for p in corr_params:
        ## take care of the case where the parameter block has a fixed level
        num_pars = R if p.level is None else numcats[p.level.name]
        ## i.e. if level is None, then we simply have R units
        val = p.value
        ## set val to zscores for noncentered parameters
        for i, nc in enumerate(p.noncentered):
            if not nc:
                continue
            val[i] = transform_values_noncentred_param(
                val[i], p.scale_value[i], IdentityTransform()
            )
        
        # make a homogeneous block
        val = [
            np.full((num_pars,), x) if not isinstance(x, list) else np.array(x)
            for x in val
        ]

        init_dict[f"block_{p.name}"] = np.array(val).T

    # mean of correlated random effects
    # TODO: categorical covariates for correlated parameters
    # Notice that p.value for ParameterBlock p is already on the unconstrained scale!
#    init_dict.update(
#        {f"loc_{p.name}": infer_loc(p.value, lambda x: x) for p in corr_params}
#    )
    # scale of correlated random effects
#    init_dict.update({f"scale_{p.name}": p.scale_value for p in corr_params})
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


StanInferenceMethod = Literal["sample", "variational", "pathfinder"] # TODO: add more methods, incl "optimize"


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
