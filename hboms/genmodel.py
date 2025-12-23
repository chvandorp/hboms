from .parameter import Parameter, ParameterBlock
from .distribution import Distribution
from .observation import Observation
from .covariate import Covariate
from . import utilities as util
from . import stanlexer
from .state import StateVar, State
from . import stanlang as sl
from . import deparse
from . import genfunctions
from . import gencommon
from .logger import logger

from functools import reduce


def require_logitnormal_functions(
    params: list[Parameter],
    corr_params: list[ParameterBlock],
) -> bool:
    """
    Function to determine if we have to add require_logitnormal 
    lpdf and rng functions.
    Instead of determinign which specific functions are required,
    we simply return a boolean indicating whether any logitnormal
    functions are needed. In that case, we always add all versions.
    
    Parameters
    ----------
    params : list[Parameter]
        List of all parameters in the model.
    corr_params : list[ParameterBlock]
        List of correlated parameter blocks.
    
    Returns
    -------
    bool
        A boolean indicating whether any logitnormal functions are required.
    """
    def is_correlated_param(p: Parameter) -> bool:
        return any([(p in cp) for cp in corr_params])
    
    # check if any parameter has a logitnormal prior
    for p in params:
        match p.get_type():
            case "fixed" | "indiv":
                if p.prior_name == "logitnormal":
                    return True
            case "random":
                if (
                    p.loc_prior_name == "logitnormal" or 
                    p.scale_prior_name == "logitnormal" or 
                    (not is_correlated_param(p) and p.distribution == "logitnormal")
                ):
                    return True
    
    return False


def gen_functions_block(
    odes: sl.MixinStmt,
    init: sl.MixinStmt,
    dists: list[Distribution],
    params: list[Parameter],
    corr_params: list[ParameterBlock],
    state: list[StateVar],
    trans_state: list[StateVar],
    transform: sl.MixinStmt,
    obs: list[Observation],
    plugin_code: sl.MixinStmt | None,
    options: dict,
    init_parnames: list[str],
    odes_parnames: list[str],
    trans_parnames: list[str],
    rngs_parnames: list[list[str]],
    loglik_parnames: list[str]
) -> list[sl.Stmt]:
    functions_block: list[sl.Stmt] = []

    if plugin_code is not None:
        functions_block.append(sl.comment("User-provided plugin code"))
        functions_block.append(plugin_code)

    ode_function = genfunctions.gen_ode_function(
        odes, params, state, options, odes_parnames
    )
    functions_block.append(sl.comment("vector field"))
    functions_block.append(ode_function)

    init_function = genfunctions.gen_init_function(
        init, params, state, options, init_parnames
    )
    functions_block.append(sl.comment("initial condition"))
    functions_block.append(init_function)

    ## optional state transform
    if len(trans_state) > 0:
        state_trans_function = genfunctions.gen_trans_function(
            trans_state,
            transform,
            params,
            state,
            options,
            trans_parnames,
        )
        functions_block.append(sl.comment("state transform function"))
        functions_block.append(state_trans_function)

    if options["init_obs"]:
        count_init_obs_function = genfunctions.gen_count_init_obs_function()
        functions_block.append(
            sl.comment("auxiliary func for counting obs time t<=t0")
        )
        functions_block.append(count_init_obs_function)

    ivp_function = genfunctions.gen_ivp_solver_function(
        params,
        state,
        trans_state,
        options,
        init_parnames,
        odes_parnames,
        trans_parnames,
    )
    functions_block.append(sl.comment("IVP solver function"))
    functions_block.append(ivp_function)

    aux_function = genfunctions.gen_wrapper_function(
        params,
        state,
        trans_state,
        options,
        init_parnames,
        odes_parnames,
        trans_parnames,
    )
    functions_block.append(sl.comment("auxiliary function for map_rect"))
    functions_block.append(aux_function)

    loglik_function = genfunctions.gen_loglik_function(
        dists, params, state, trans_state, obs, options, loglik_parnames=loglik_parnames
    )
    functions_block.append(sl.comment("log-likelihood function"))
    functions_block.append(loglik_function)

    rng_functions = [
        genfunctions.gen_rng_function(
            dist, params, state, trans_state, rng_parnames=rng_parnames
        )
        for dist, rng_parnames in zip(dists, rngs_parnames)
    ]
    comment_rng = "rng function" + ("s" if len(rng_functions) > 1 else "")
    functions_block.append(sl.comment(comment_rng))
    functions_block += rng_functions

    # add (optional) functions for calculating transformed parameters
    functions_block += gencommon.gen_all_trans_param_functions(params)


    # add (optional) functions for logitnormal distribution if needed
    # we only need this for uncorrelated parameters
    # as correlated parameters are sampled on the unrestricted scale
    # Notice that the user can specify logitnormal distributions for
    # random, fixed, and indiv parameters.

    require_logitnormal = require_logitnormal_functions(
        params, corr_params
    )
    if require_logitnormal:
        functions_block.append(
            sl.comment("logit-normal distribution for bounded scalar parameters")
        )
        functions_block.append(genfunctions.gen_logitnormal_lpdf()) 

        functions_block.append(
            sl.comment("logit-normal distribution for bounded parameters with covariates")
        )
        functions_block.append(genfunctions.gen_vectorized_logitnormal_lpdf(array_loc=True))

        functions_block.append(
            sl.comment("logit-normal distribution for bounded parameters")
        )
        functions_block.append(genfunctions.gen_vectorized_logitnormal_lpdf(array_loc=False))  

    if require_logitnormal and options["prior_samples"]:
        functions_block.append(
            sl.comment("logit-normal random number generator")
        )
        functions_block.append(genfunctions.gen_logitnormal_rng())

    return functions_block


def gen_parameters_block(
    params: list[Parameter], 
    corr_params: list[ParameterBlock], 
    plugin_code: sl.MixinStmt | None,
    options: dict
) -> list[sl.Stmt]:
    param_block: list[sl.Stmt] = []

    uncorr_params = [p for p in params if not any([(p in cp) for cp in corr_params])]

    indiv_par_stmts = util.flatten(
        [
            p.genstmt_params()
            for p in uncorr_params
            if p.get_type() in ["random", "indiv"]
        ]
    )

    if len(indiv_par_stmts) > 0:
        param_block.append(
            sl.EmptyStmt(comment="individual parameters (and their hyper-parameters)")
        )
    param_block += indiv_par_stmts

    fixed_par_stmts = util.flatten(
        [p.genstmt_params() for p in uncorr_params if p.get_type() == "fixed"]
    )

    if len(fixed_par_stmts) > 0:
        param_block.append(sl.EmptyStmt(comment="fixed parameters"))
    param_block += fixed_par_stmts

    if len(corr_params) > 0:
        param_block.append(sl.EmptyStmt(comment="correlated individual parameters"))
    param_block += util.flatten([p.genstmt_params() for p in corr_params])

    if plugin_code is not None:
        param_block.append(sl.comment("User-provided plugin code"))
        param_block.append(plugin_code)

    return param_block


def gen_trans_param_block(
    params: list[Parameter],
    corr_params: list[ParameterBlock],
    plugin_code: sl.MixinStmt | None,
    options: dict,
    ivp_parnames: list[str],
) -> list[sl.Stmt]:
    RVar = sl.Var(sl.Int(), "R")
    rVar = sl.Var(sl.Int(), "r")

    uncorr_params = [p for p in params if not any([(p in cp) for cp in corr_params])]

    # create a list of statements
    statements: list[sl.Stmt] = []

    # define noncentered parameters from stdnorm random effects (exclude correlated parameters for now)
    noncentered_params = [p for p in uncorr_params if p.get_type() == "random" and p.noncentered]
    if len(noncentered_params) > 0:
        statements.append(sl.comment("non-centered parameterizations"))
        statements += util.flatten([p.genstmt_trans_params() for p in noncentered_params])
        

    # unpack correlated parameters (NB: make sure to do this before computing transformed parameters)
    if len(corr_params) > 0:
        statements.append(sl.EmptyStmt(comment="unpack correlated parameters"))
    statements += util.flatten([p.genstmt_trans_params() for p in corr_params])

    # compute user-defined parameter transformations. 
    # Make sure they are defined in the right order, respecting dependencies
    trans_params = [p for p in params if p.get_type() in ["trans", "trans_indiv"]]
    ranks = [p.get_rank() for p in trans_params]
    if not util.is_sorted(ranks):
        trans_params.sort(key=lambda p: p.get_rank())
        message = "Due to dependencies between transformed parameters, "
        message += "the user-defined parameter order has been modified "
        message += "in the transformed parameters block."
        logger.info(message)

    trans_par_decls = [p.genstmt_trans_params() for p in trans_params]

    if trans_par_decls:
        statements.append(sl.comment("user-defined parameter transformations"))

    for stmts in trans_par_decls:
        statements += stmts




    # random and indiv parameters can be treated the same

    random_par_objects = [
        p for p in params if gencommon.treat_as_random(p) and p.name in ivp_parnames
    ]
    random_par_vars = [p.var for p in random_par_objects]
    random_flat_dims = [var.var_type.flat_dim() for var in random_par_vars]

    def sum_dims(dims):
        return sl.izero() if len(dims) == 0 else reduce(lambda a, b: a + b, dims)

    ku = sum_dims(random_flat_dims)  # number of flat real random parameters
    uparsVar = sl.Var(sl.Array(sl.Vector(ku), (RVar,)), "upars")

    # fixed parameters

    fixed_par_objects = [
        p for p in params if gencommon.treat_as_fixed(p) and p.name in ivp_parnames
    ]
    fixed_par_vars: list[sl.Expr] = [p.var for p in fixed_par_objects]
    fixed_flat_dims = [var.var_type.flat_dim() for var in fixed_par_vars]
    kp = sum_dims(fixed_flat_dims)  # number of flat real fixed parameters
    pparVar = sl.Var(sl.Vector(kp), "ppar")

    # if all fixed parameters are scalars, we can simplify creating the ppar variable.
    all_ppars_scalar = all([p.space == "real" for p in fixed_par_objects])

    statements.append(
        sl.Decl(uparsVar, comment="prepare data structure for map_rect")
    )

    ppar_assigns: list[sl.Assign] = []

    if len(fixed_par_objects) > 0 and all_ppars_scalar:
        # declare-assign ppar, use a literal vector
        ppar_decl: sl.Decl | sl.DeclAssign = sl.DeclAssign(
            pparVar,
            sl.LiteralVector(fixed_par_vars),
            comment="assign population parameters to single vector",
        )
    if len(fixed_par_objects) > 0 and not all_ppars_scalar:
        # we have to first declare ppar, and then assign parameters to it
        ppar_decl: sl.Decl | sl.DeclAssign = sl.Decl(
            pparVar, comment="population parameters"
        )
        ppar_assigns.append(
            sl.comment("assign population parameters to population vector")
        )
        for i, p in enumerate(fixed_par_objects):
            var = p.var
            match p.space:
                case "real":
                    idx = reduce(lambda a, b: a + b, fixed_flat_dims[:i], sl.one())
                    assn = sl.Assign(pparVar.idx(idx), var)
                case "vector":
                    idx = sl.Range(
                        reduce(lambda a, b: a + b, fixed_flat_dims[:i], sl.one()),
                        reduce(lambda a, b: a + b, fixed_flat_dims[: i + 1]),
                    )
                    # ppar already is a vector, so no casting required
                    assn = sl.Assign(pparVar.idx(idx), var)
                case "matrix":
                    idx = sl.Range(
                        reduce(lambda a, b: a + b, fixed_flat_dims[:i], sl.one()),
                        reduce(lambda a, b: a + b, fixed_flat_dims[: i + 1]),
                    )
                    # cast matrix to flat vector
                    cast_var = sl.Call("to_vector", var)
                    assn = sl.Assign(pparVar.idx(idx), cast_var)
                case _:
                    raise NotImplementedError(
                        f"parameter space {p.space} not supported for fixed parameters"
                    )

            ppar_assigns.append(assn)

    if len(fixed_flat_dims) == 0:
        # there are no population parameters
        ppar_decl = sl.Decl(
            pparVar,
            comment="no population parameters required for map_rect",
        )

    statements.append(ppar_decl)


    # add unit-specific parameters
    if len(random_par_objects) > 0:
        statements.append(sl.comment("assign unit-parameters to array of vectors"))

    upar_assigns: list[sl.Assign] = []

    for i, p in enumerate(random_par_objects):
        var = sl.expandVar(
            p.var, (RVar,)
        )  ## FIXME: RVar should be different if level is not None

        match p.space:
            case "real":
                idx = reduce(lambda a, b: a + b, random_flat_dims[:i], sl.one())
                uparsVar_indexed = uparsVar.idx(sl.fullRange(), idx)
                var_indexed = (
                    var
                    if p.level is None or p.level_type == "random"
                    else var.idx(p.level.var)
                )
                assn = sl.Assign(uparsVar_indexed, var_indexed)
            case "vector":  # TODO: will this work??
                idx = sl.Range(
                    reduce(lambda a, b: a + b, random_flat_dims[:i], sl.one()),
                    reduce(lambda a, b: a + b, random_flat_dims[: i + 1]),
                )
                uparsVar_indexed = uparsVar.idx(sl.fullRange(), idx)
                var_indexed = var if p.level is None else var.idx(p.level.var)
                assn = sl.Assign(uparsVar_indexed, var_indexed)
            case "matrix":
                idx = sl.Range(
                    reduce(lambda a, b: a + b, random_flat_dims[:i], sl.one()),
                    reduce(lambda a, b: a + b, random_flat_dims[: i + 1]),
                )
                uparsVar_indexed = uparsVar.idx(rVar, idx)
                var_indexed = (
                    var.idx(rVar) if p.level is None else var.idx(p.level.var.idx(rVar))
                )
                cast_var = sl.Call("to_vector", var_indexed)
                assn = sl.ForLoop(
                    rVar,
                    sl.Range(sl.one(), RVar),
                    sl.Assign(uparsVar_indexed, cast_var),
                )
            case _:
                raise NotImplementedError(
                    f"combination of parameter type {p.get_type()} and space {p.space} not supported"
                )

        upar_assigns.append(assn)

    # add unit-parameter assignments
    statements += upar_assigns

    # add population parameter assignments
    statements += ppar_assigns

    if plugin_code is not None:
        statements.append(sl.comment("User-provided plugin code"))
        statements.append(plugin_code)

    return statements


def gen_data_block(
    params: list[Parameter],
    obs: list[Observation],
    covs: list[Covariate],
    plugin_code: sl.MixinStmt | None,
    options: dict,
) -> list[sl.Stmt]:
    declarations: list[sl.Stmt] = []

    # number of units
    RVar = sl.Var(sl.Int(lower=sl.izero()), "R")
    declarations.append(sl.Decl(RVar, comment="number of units"))

    # number of observations per unit
    NVar = sl.Var(sl.Array(sl.Int(lower=sl.izero()), (RVar,)), "N")
    declarations.append(sl.Decl(NVar, comment="number of observations per unit"))

    # array of observation times
    TimeVar = sl.Var(
        sl.Array(sl.Real(lower=sl.rzero()), (RVar, sl.Call("max", [NVar]))),
        "Time",
    )
    declarations.append(sl.Decl(TimeVar, comment="observation times"))

    declarations.append(sl.EmptyStmt(comment="observations"))
    declarations += util.flatten([ob.genstmt_data() for ob in obs])

    # number of simulation time points
    NSimVar = sl.Var(sl.Int(lower=sl.izero()), "NSim")
    declarations.append(sl.Decl(NSimVar, comment="number of simulation time points"))

    # array of simulation time points
    TimeSimDecl = sl.Decl(
        sl.Var(sl.Array(sl.Real(lower=sl.rzero()), (RVar, NSimVar)), "TimeSim"),
        comment="simulation time points",
    )
    declarations.append(TimeSimDecl)

    # non-scalar parameter shapes
    param_data_decls = util.flatten(
        [
            p.genstmt_data()
            for p in params
            if p.get_type() in ["fixed", "indiv", "random"]
        ]
    )
    if len(param_data_decls) > 0:
        declarations.append(sl.comment("non-scalar parameter shapes"))
        declarations += param_data_decls

    # constants
    const_decls = util.flatten(
        [p.genstmt_data() for p in params if p.get_type() in ["const", "const_indiv"]]
    )
    if len(const_decls) > 0:
        declarations.append(sl.EmptyStmt(comment="constants"))
        declarations += const_decls

    # covariates
    if len(covs) > 0:
        declarations.append(sl.EmptyStmt(comment="covariates"))
        declarations += util.flatten([cov.genstmt_data() for cov in covs])

    if plugin_code is not None:
        declarations.append(sl.comment("User-provided plugin code"))
        declarations.append(plugin_code)

    return declarations


def gen_trans_data_block(
    params: list[Parameter],
    plugin_code: sl.MixinStmt | None,
    options: dict,
    ivp_parnames: list[str],
) -> list[sl.Stmt]:
    """
    Generate transformed data block statements, including preparing
    rdats and idats for passing constants and shapes to map_rect.

    TODO: remove separation of declarations and definitions.

    Parameters
    ----------
    params : list[Parameter]
        List of all parameters in the model.
    plugin_code : sl.MixinStmt | None
        Optional user-provided plugin code to include in the transformed data block.
    options : dict
        Dictionary of model options.
    ivp_parnames : list[str]
        List of parameter names involved in the IVP.

    Returns
    -------
    list[sl.Stmt]
        List of statements for the transformed data block.
    """

    consts = [
        p
        for p in params
        if p.name in ivp_parnames and p.get_type() in ["const", "const_indiv"]
    ]

    const_flat_dims = [p.var.var_type.flat_dim() for p in consts]

    kc = (
        sl.izero()
        if len(const_flat_dims) == 0
        else reduce(lambda a, b: a + b, const_flat_dims)
    )

    RVar = sl.Var(sl.Int(), "R")
    rVar = sl.Var(sl.Int(), "r")
    NVar = sl.Var(sl.Array(sl.Int(), (RVar,)), "N")
    maxN = sl.Call("max", [NVar])
    TimeVar = sl.Var(sl.Array(sl.Real(), (maxN,)), "Time")
    num_rdats = maxN if len(consts) == 0 else maxN + kc  # avoid "+ 0" in code
    rdats_var = sl.Var(sl.Array(sl.Real(), (RVar, num_rdats)), "rdats")
    rdats_decl = sl.Decl(rdats_var)

    rdats_time_assign = sl.Assign(
        sl.MultiIndexOp(
            rdats_var,
            [
                sl.fullRange(),
                sl.Range(sl.one() + kc, num_rdats),
            ],
        ),
        TimeVar,
    )

    # idat contains shapes for non-scalar parameters
    shape_len_dict = {"real": 0, "vector": 1, "matrix": 2}
    shape_lens = [shape_len_dict[p.space] for p in params if p.name in ivp_parnames]

    num_idats = sl.LiteralInt(1 + sum(shape_lens))
    idats_var = sl.Var(
        sl.Array(sl.Int(), (RVar, num_idats)), "idats"
    )  # TODO: more integer data??
    idats_decl = sl.Decl(idats_var, comment="integer data")
    idats_num_time_assign = sl.Assign(
        sl.MultiIndexOp(idats_var, [sl.fullRange(), sl.one()]), NVar
    )

    # if there are random parameters with a random level, we need to define boolean matrices
    level_matrix_decls, level_matrix_asgns = gencommon.gen_level_matrices(params)

    decls: list[sl.Stmt] = [
        sl.EmptyStmt(comment="declarations"),
        rdats_decl,
        idats_decl,
    ]
    decls += level_matrix_decls

    defs: list[sl.Stmt] = [
        sl.EmptyStmt(comment="definitions"),
        rdats_time_assign,
        idats_num_time_assign,
    ]
    if len(level_matrix_decls) > 0:
        defs.append(sl.comment("construct level matrices"))
        defs += level_matrix_asgns

    # check if there are categorical covariates that require restriction
    covariate_restrictions = gencommon.gen_covariate_restrictions(params)
    if len(covariate_restrictions) > 0:
        defs.append(sl.comment("restrict categorical covariate values based on level"))
        defs += covariate_restrictions

    rdats_assigns: list[sl.Stmt] = []
    for i, p in enumerate(consts):
        var = p.var

        match p.space:
            case "real":
                idx = reduce(lambda a, b: a + b, const_flat_dims[:i], sl.one())
            case "vector" | "matrix":
                idx = sl.Range(
                    reduce(lambda a, b: a + b, const_flat_dims[:i], sl.one()),
                    reduce(lambda a, b: a + b, const_flat_dims[: i + 1]),
                )

        rdats_var_indexed_full = rdats_var.idx(sl.fullRange(), idx)
        rdats_var_indexed_r = rdats_var.idx(rVar, idx)

        match (p.get_type(), p.space):
            case ("const", "real"):
                expand_var = sl.Call("rep_array", [var, RVar])
                assn = sl.Assign(rdats_var_indexed_full, expand_var)
            case ("const", "vector" | "matrix"):
                cast_var = sl.Call("to_array_1d", var)
                expand_var = sl.Call("rep_array", [cast_var, RVar])
                assn = sl.Assign(rdats_var_indexed_full, expand_var)
            case ("const_indiv", "real"):
                assn = sl.Assign(rdats_var_indexed_full, var)
            case ("const_indiv", "vector" | "matrix"):
                # we need a for loop to handle this
                cast_var = sl.Call("to_array_1d", var.idx(rVar))
                assn = sl.ForLoop(
                    rVar,
                    sl.Range(sl.one(), RVar),
                    sl.Assign(rdats_var_indexed_r, cast_var),
                )
            case _:
                raise NotImplementedError(
                    f"combination of parameter type {p.get_type()} and space {p.space} not supported"
                )

        rdats_assigns.append(assn)

    # add a comment only if we assign values to rdats
    if len(rdats_assigns) > 0:
        rdats_assigns.insert(0, sl.comment("add constants to rdats"))

    # shapes for non-scalar parameters
    shape_assigns: list[sl.Stmt] = []
    ivp_params = [p for p in params if p.name in ivp_parnames]
    for i, par in enumerate(ivp_params):
        match par.space:
            case "real":
                pass  # no shape required
            case "vector":
                idx = sl.LiteralInt(1 + sum(shape_lens[: i + 1]))
                shp = sl.Call("rep_array", [par.shape, RVar])
                shape_assigns.append(sl.Assign(idats_var.idx(sl.fullRange(), idx), shp))
            case "matrix":
                idx = sl.Range(
                    sl.LiteralInt(1 + sum(shape_lens[:i]) + 1),
                    sl.LiteralInt(1 + sum(shape_lens[: i + 1])),
                )
                shp = sl.Call("rep_array", [par.shape, RVar])
                shape_assigns.append(sl.Assign(idats_var.idx(sl.fullRange(), idx), shp))
            case _:
                raise NotImplementedError(f"parameter space {par.space} not supported")

    # add a comment only if we add shapes to idats
    if len(shape_assigns) > 0:
        shape_assigns.insert(0, sl.comment("add shapes to idats"))

    statements = decls + defs + rdats_assigns + shape_assigns

    if plugin_code is not None:
        statements.append(sl.comment("User-provided plugin code"))
        statements.append(plugin_code)

    return statements


def gen_prior(
    params: list[Parameter], 
    corr_params: list[ParameterBlock],
    options: dict
) -> list[sl.Stmt]:
    uncorr_params = [p for p in params if not any([(p in cp) for cp in corr_params])]

    uncorr_prior_expr = util.flatten(
        [
            p.genstmt_model()
            for p in uncorr_params
            if p.get_type() not in ["const", "const_indiv"]
        ]
    )
    corr_prior_expr = util.flatten([p.genstmt_model() for p in corr_params])

    prior = uncorr_prior_expr + corr_prior_expr

    return prior



def gen_prior_samples(
    params: list[Parameter], 
    corr_params: list[ParameterBlock],
) -> list[sl.Stmt]:
    uncorr_params = [p for p in params if not any([(p in cp) for cp in corr_params])]

    uncorr_prior_samples = util.flatten([
        p.genstmt_prior_samples()
        for p in uncorr_params
        if p.get_type() not in ["const", "const_indiv"]
    ])

    corr_prior_samples = util.flatten([
        p.genstmt_prior_samples() for p in corr_params
    ])

    prior_samples = uncorr_prior_samples + corr_prior_samples

    return prior_samples



def gen_model_block(
    params: list[Parameter],
    corr_params: list[ParameterBlock],
    state_vars: list[StateVar],
    trans_state_vars: list[StateVar],
    obs: list[Observation],
    plugin_code: sl.MixinStmt | None,
    options: dict,
    loglik_parnames: list[str],
    include_likelihood: bool = True, # whether to include likelihood
) -> list[sl.Stmt]:
    state = State(state_vars + trans_state_vars)
    state_dim = state.flat_dim()
    state_var = sl.Var(sl.Vector(state_dim), "state")

    R = sl.Var(sl.Int(), "R")
    r = sl.Var(sl.Int(), "r")
    N = sl.Var(sl.Array(sl.Int(), (R,)), "N")
    maxN = sl.Call("max", [N])
    sumN = sl.Call("sum", [N])
    n = sl.Var(sl.Int(), "n")
    # determine which parameters need a unit-index
    ll_par_args: list[sl.Expr] = [
        p.expand_and_index_var(apply_idx=True)
        for p in params
        if p.name in loglik_parnames
    ]
    # observation arguments for log_lik function
    obs_list: list[sl.Expr] = [
        sl.MultiIndexOp(sl.expandVar(ob.var, (R, maxN)), [r, n]) for ob in obs
    ]
    cc_list: list[sl.Expr] = [
        sl.MultiIndexOp(sl.Var(sl.Array(sl.Int(), (R, maxN)), ob.cc_name), [r, n])
        for ob in obs
        if ob.censored
    ]
    obs_args = obs_list + cc_list
    # code for the prior
    prior = gen_prior(params, corr_params, options)
    # reslut of parallel ODE integration
    concat_res = sl.Var(sl.Vector(sumN * state_dim), "concat_res")
    # these are defined in transformed data and transformed parameters
    ppar = sl.Var(sl.Vector(sl.LiteralInt(0)), "ppar")  ## TODO: correct dimension
    upars = sl.Var(
        sl.Array(sl.Vector(sl.LiteralInt(0)), (R,)), "upars"
    )  ## TODO: correct dimension
    rdats = sl.Var(
        sl.Array(sl.Real(), (R, sl.LiteralInt(0))), "rdats"
    )  ## TODO: correct dimension
    idats = sl.Var(
        sl.Array(sl.Int(), (R, sl.LiteralInt(0))), "idats"
    )  ## TODO: correct dimension

    map_rect_helper_fun = sl.Var(
        sl.Function(concat_res.var_type, ()),  ## TODO: correct return types
        "map_rect_helper_fun",
    )

    # expression for parallel integration
    integration = sl.DeclAssign(
        concat_res,
        sl.Call("map_rect", [map_rect_helper_fun, ppar, upars, rdats, idats]),
    )
    target = sl.Var(sl.Real(), "target")
    idx = sl.Var(sl.Int(), "idx")
    sumNr = sl.Call("sum", [N.idx(sl.Range(None, r - sl.one()))])
    ## log-likelihood for loop
    inner_for_loop = sl.ForLoop(
        n,
        sl.Range(sl.one(), N.idx(r)),
        sl.Scope(
            [
                sl.EmptyStmt(comment="extract state"),
                sl.DeclAssign(
                    idx, (state_dim * sl.Par(sumNr + (n - sl.one()))) + sl.one()
                ),
                sl.DeclAssign(
                    state_var,
                    concat_res.idx(sl.Range(idx, idx + (state_dim - sl.one()))),
                ),
                sl.EmptyStmt(comment="compute likelihood of observation given state"),
                sl.AddAssign(
                    target,
                    sl.Call("loglik_fun", obs_args + [state_var] + ll_par_args),
                ),
            ]
        ),
    )

    ll_for_loop = sl.ForLoop(r, sl.Range(sl.one(), R), sl.Scope([inner_for_loop]))

    model_block: list[sl.Stmt] = []

    if include_likelihood:
        model_block = [
            sl.EmptyStmt(comment="solve ODEs in parallel"),
            integration,
            sl.EmptyStmt(comment="compute log-likelihood of observations"),
            ll_for_loop,
            sl.EmptyStmt(comment="prior"),
        ] + prior
    else:
        model_block = [
            sl.EmptyStmt(comment="likelihood is omitted..."),
            sl.EmptyStmt(comment="prior")
        ] + prior

    if plugin_code is not None:
        model_block.append(sl.comment("User-provided plugin code"))
        model_block.append(plugin_code)

    return model_block


def gen_gq_block(
    params: list[Parameter],
    corr_params: list[ParameterBlock],
    state: list[StateVar],
    trans_state: list[StateVar],
    obs: list[Observation],
    plugin_code: sl.MixinStmt | None,
    options: dict,
    ivp_parnames: list[str],
    rngs_parnames: list[list[str]],
    loglik_parnames: list[str],
) -> list[sl.Stmt]:
    # get some data
    odim = len(obs)

    ext_state = State(state + trans_state)
    estate_dim = ext_state.flat_dim()

    # commonly used variables
    R = sl.Var(sl.Int(), "R")
    NSim = sl.Var(sl.Int(), "NSim")
    N = sl.Var(sl.Array(sl.Int(), (R,)), "N")
    maxN = sl.Call("max", [N])
    sumN = sl.Call("sum", [N])
    r = sl.Var(sl.Int(), "r")
    n = sl.Var(sl.Int(), "n")
    u_sim = sl.Var(sl.Array(sl.Vector(estate_dim), (NSim,)), "u_sim")
    Nr = sl.IndexOp(N, r)
    u_sim_obs = sl.Var(sl.Array(sl.Vector(estate_dim), (Nr,)), "u_sim_obs")
    TimeSim = sl.Var(sl.Array(sl.Real(), (NSim,)), "TimeSim")
    TimeSim_r = sl.IndexOp(TimeSim, r)
    Time = sl.Var(sl.Array(sl.Real(), (R, maxN)), "Time")

    statements: list[sl.Stmt] = []  # to be returned

    # create list with correctly indexed parameters
    par_vars = [p.expand_and_index_var(apply_idx=True) for p in params]
    # select parameters used by the loglik function
    ll_par_vars = [pv for pv, p in zip(par_vars, params) if p.name in loglik_parnames]
    # select parameters used by the rng functions
    rngs_par_vars = [
        [pv for pv, p in zip(par_vars, params) if p.name in rng_parnames]
        for rng_parnames in rngs_parnames
    ]
    # select parameters used by ivp solver
    ivp_par_vars = [pv for pv, p in zip(par_vars, params) if p.name in ivp_parnames]

    # correlations between parameters
    statements += util.flatten([p.genstmt_gq() for p in corr_params])

    # declare and assign trajectories
    traj_vars = [
        sl.Var(sl.expandType(var.stan_type, (R, NSim)), var.name + "_sim")
        for var in state + trans_state
    ]
    traj_decl_lists = sl.gen_decl_lists(traj_vars)

    statements += traj_decl_lists

    traj_asgns = ext_state.gen_unpack_stmt(
        u_sim.idx(n),
        rename_func=lambda x: x + "_sim",
        transform_func=lambda x: sl.MultiIndexOp(sl.expandVar(x, (R, NSim)), [r, n]),
        declare=False,
    )

    # declare simulations
    sim_vars = [
        sl.Var(sl.expandType(ob.obs_type, (R, maxN)), ob.name + "_sim") for ob in obs
    ]

    statements += [sl.Decl(var) for var in sim_vars]

    sim_asgns = [
        sl.Assign(
            sl.MultiIndexOp(sim_vars[i], [r, n]),
            sl.Call(ob.name + "_rng", [u_sim_obs.idx(n)] + rngs_par_vars[i]),
            comment="simulate data at observation times",
        )
        for i, ob in enumerate(obs)
    ]

    # log-likelihood vector
    log_lik = sl.Var(sl.Vector(sumN * sl.LiteralInt(odim)), "log_lik")
    log_lik_decl = sl.Decl(
        log_lik, comment="vector of log-likelihoods for model comparison"
    )

    statements.append(log_lik_decl)

    loop_save_sim = sl.ForLoop(n, sl.Range(sl.one(), NSim), sl.Scope(traj_asgns))

    idx = sl.Var(sl.Int(), "idx")
    # int idx = odim * (sum(N[:r-1]) + n - 1) + 1
    sumNrn = sl.Call("sum", [N.idx(sl.Range(None, r - sl.one()))]) + (n - sl.one())
    idx_decl = sl.DeclAssign(idx, sl.LiteralInt(odim) * sl.Par(sumNrn) + sl.one())

    obs_vars = [sl.expandVar(ob.var, (R, maxN)) for ob in obs]
    obs_vars_idx: list[sl.Expr] = [sl.MultiIndexOp(var, [r, n]) for var in obs_vars]

    cc_vars = [
        sl.Var(sl.Array(sl.Int(), (R, maxN)), ob.cc_name) for ob in obs if ob.censored
    ]

    cc_vars_idx: list[sl.Expr] = [sl.MultiIndexOp(var, [r, n]) for var in cc_vars]

    llfun_args: list[sl.Expr] = (
        obs_vars_idx + cc_vars_idx + [u_sim_obs.idx(n)] + ll_par_vars
    )
    ll_asgn = sl.Assign(
        log_lik.idx(sl.Range(idx, idx + sl.LiteralInt(odim - 1))),
        sl.Call(
            "loglik_fun",
            llfun_args,
        ),
        comment="record log-likelihood of each observation",
    )

    loop_save_obs = sl.ForLoop(
        n, sl.Range(sl.one(), Nr), sl.Scope([idx_decl, ll_asgn] + sim_asgns)
    )

    Time_r_Nr = sl.MultiIndexOp(Time, [r, sl.Range(sl.one(), Nr)])
    solve_ivp_args_obs = [Time_r_Nr] + ivp_par_vars
    solve_ivp_args_sim = [TimeSim_r] + ivp_par_vars

    loop_units = sl.ForLoop(
        r,
        sl.Range(sl.one(), R),
        sl.Scope(
            [
                sl.DeclAssign(
                    u_sim,
                    sl.Call("solve_ivp", solve_ivp_args_sim),
                    comment="solve ODEs at simulation times",
                ),
                sl.DeclAssign(
                    u_sim_obs,
                    sl.Call("solve_ivp", solve_ivp_args_obs),
                    comment="solve ODEs at observation times",
                ),
                loop_save_sim,
                loop_save_obs,
            ]
        ),
    )

    statements.append(loop_units)

    if options["prior_samples"]:
        statements.append(sl.comment("generate prior samples"))
        statements += gen_prior_samples(params, corr_params)



    if plugin_code is not None:
        statements.append(sl.comment("User-provided plugin code"))
        statements.append(plugin_code)

    return statements


def gen_stan_model(
    odes: sl.MixinStmt,
    init: sl.MixinStmt,
    dists: list[Distribution],
    params: list[Parameter],
    corr_params: list[ParameterBlock],
    state: list[StateVar],
    trans_state: list[StateVar],
    transform: sl.MixinStmt,
    obs: list[Observation],
    covs: list[Covariate],
    plugin_code: dict[str, sl.MixinStmt],
    options: dict,
    include_likelihood: bool = True, # whether to include the likelihood in the model block
) -> sl.StanModel:
    # determine which parameters are required for the log-likelihood function

    # FIXME: instead of deparsing and tokenizing, find variables in an expression directly

    # TODO: there are distributions that use different parameters for RNG and loglik (e.g. multinomial)

    parnames = [p.name for p in params]

    dists_pars = [dist.params() for dist in dists]
    dists_parnames = [
        util.unique(
            util.flatten(
                [
                    stanlexer.find_used_names(deparse.deparse_expr(par), parnames)
                    for par in dist_pars
                ]
            )
        )
        for dist_pars in dists_pars
    ]

    # the loglik function takes parameters from all distributions
    loglik_parnames = util.unique(util.flatten(dists_parnames))

    # find parameters required for initial state, odes, and transform
    init_parnames = stanlexer.find_used_names(deparse.deparse_stmt(init), parnames)
    odes_parnames = stanlexer.find_used_names(deparse.deparse_stmt(odes), parnames)
    trans_parnames = stanlexer.find_used_names(
        deparse.deparse_stmt(transform), parnames
    )
    # ivp_parnames is the union of init, odes, and trans. NB: keep order as in parnames!
    ivp_parnames = [
        pn
        for pn in parnames
        if pn in set(init_parnames + odes_parnames + trans_parnames)
    ]

    model = sl.StanModel(
        gen_functions_block(
            odes,
            init,
            dists,
            params,
            corr_params,
            state,
            trans_state,
            transform,
            obs,
            plugin_code.get("functions", None),
            options,
            init_parnames,
            odes_parnames,
            trans_parnames,
            dists_parnames,
            loglik_parnames
        ),
        gen_data_block(
            params, 
            obs, 
            covs,
            plugin_code.get("data", None),
            options
        ),
        gen_trans_data_block(
            params,
            plugin_code.get("transformed data", None),
            options,
            ivp_parnames,
        ),
        gen_parameters_block(
            params, 
            corr_params, 
            plugin_code.get("parameters", None),
            options
        ),
        gen_trans_param_block(
            params, 
            corr_params,
            plugin_code.get("transformed parameters", None),
            options, 
            ivp_parnames
        ),
        gen_model_block(
            params,
            corr_params,
            state,
            trans_state,
            obs,
            plugin_code.get("model", None),
            options,
            loglik_parnames,
            include_likelihood=include_likelihood,
        ),
        gen_gq_block(
            params,
            corr_params,
            state,
            trans_state,
            obs,
            plugin_code.get("generated quantities", None),
            options,
            ivp_parnames,
            dists_parnames,
            loglik_parnames,
        ),
    )

    return model
