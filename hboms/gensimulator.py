"""
Generate code for data simulation Stan model.
Such a model can be used to generate simulated data using
the same HBOMS models used for inference.
This allows for easy model validation
"""
from typing import Optional

from .parameter import Parameter, ParameterBlock
from .covariate import Covariate
from .distribution import Distribution
from .observation import Observation
from . import utilities as util
from . import stanlexer
from .state import StateVar, State
from . import stanlang as sl
from . import deparse
from . import genfunctions, gencommon


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
    options: dict,
    init_parnames: list[str],
    odes_parnames: list[str],
    trans_parnames: list[str],
    rngs_parnames: list[list[str]],
) -> list[sl.Stmt]:
    functions_block: list[sl.Stmt] = []

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
    
    uncorr_random_params = [
        p for p in params 
        if not any([(p in cp) for cp in corr_params])
        and p.get_type() == "random"
    ]
    if any([p.distribution == "logitnormal" for p in uncorr_random_params]):
        functions_block.append(sl.comment("logit-normal rng for bounded parameters"))
        functions_block.append(genfunctions.gen_logitnormal_rng())

    return functions_block


def gen_parameters_block() -> list[sl.Stmt]:
    return [sl.comment("no parameters")]


def gen_trans_param_block() -> list[sl.Stmt]:
    return [sl.comment("no transformed parameters")]


def gen_model_block() -> list[sl.Stmt]:
    return [sl.comment("no model")]


def gen_data_block(params, corr_params, covs, options) -> list[sl.Stmt]:
    data_block: list[sl.Stmt] = []

    R = sl.Var(sl.Int(lower=sl.izero()), "R")
    N = sl.Var(sl.Array(sl.Int(lower=sl.izero()), (R,)), "N")
    maxN = sl.Call("max", [N])
    Time = sl.Var(sl.Array(sl.Real(lower=sl.rzero()), (R, maxN)), "Time")

    data_block.append(sl.Decl(R, comment="number of units"))
    data_block.append(sl.Decl(N, comment="number of observations per unit"))
    data_block.append(sl.Decl(Time, comment="observation times"))

    # covariates
    if len(covs) > 0:
        data_block.append(sl.EmptyStmt(comment="covariates"))
    data_block += util.flatten([cov.genstmt_data() for cov in covs])

    # parameters

    data_block.append(sl.comment("non-random parameters and hyper-parameters"))

    for par in params:
        data_block += par.genstmt_data_simulator()

    # correlated parameters

    data_block.append(sl.comment("hyper parameters for correlated parameters"))

    for par in corr_params:
        data_block += par.genstmt_data_simulator()

    return data_block


def gen_trans_data_block(params: list[Parameter]) -> Optional[list[sl.Stmt]]:
    # if there are random parameters with a random level, we need to define boolean matrices
    level_matrix_decls, level_matrix_asgns = gencommon.gen_level_matrices(params)

    if len(level_matrix_decls) == 0:
        return None

    stmts = [sl.comment("declare and construct level matrices")]
    return stmts + level_matrix_decls + level_matrix_asgns


def gen_gq_block(
    params,
    corr_params,
    state,
    trans_state,
    obs,
    covs,
    options,
    ivp_parnames,
    rngs_parnames,
) -> list[sl.Stmt]:
    state_obj = State(state + trans_state)
    state_dim = state_obj.flat_dim()

    R = sl.intVar("R")
    r = sl.intVar("r")
    N = sl.Var(sl.Array(sl.Int(), (R,)), "N")
    n = sl.intVar("n")
    Nr = N.idx(r)
    maxN = sl.Call("max", N)
    Time = sl.Var(sl.Array(sl.Real(), (R, maxN), data=True), "Time")
    Time_r = Time.idx(r, sl.Range(sl.one(), Nr))

    uncorr_params = [p for p in params if not any([(p in cp) for cp in corr_params])]

    # create list with correctly expanded and indexed parameters
    par_vars = [p.expand_and_index_var(apply_idx=True) for p in params]
    # select parameters used by the rng functions
    rngs_par_vars = [
        [pv for pv, p in zip(par_vars, params) if p.name in rng_parnames]
        for rng_parnames in rngs_parnames
    ]
    # select parameters used by ivp solver
    ivp_par_vars = [pv for pv, p in zip(par_vars, params) if p.name in ivp_parnames]

    gq_block: list[sl.Stmt] = []

    # declarations of simulated variables
    declarations: list[sl.Stmt] = []

    # random parameter declarations: make sure that the level is correctly handeled
    par_rng_decls = [
        sl.Decl(p.expand_and_index_var(apply_idx=False))
        for p in params
        if p.get_type() == "random"
    ]
    declarations.append(sl.comment("random parameter declarations"))
    declarations += par_rng_decls

    # declarations of individual-level transformed parameters
    trans_par_decls = [
        sl.Decl(p.expand_and_index_var(apply_idx=False))
        for p in params
        if p.get_type() == "trans_indiv"
    ]
    if trans_par_decls:
        declarations.append(sl.comment("transformed parameter declarations"))
    declarations += trans_par_decls


    sim_obs = [sl.Var(sl.expandType(ob.obs_type, (R, maxN)), ob.name) for ob in obs]

    declarations.append(sl.comment("simulated observation declarations"))
    declarations += [sl.Decl(ob) for ob in sim_obs]

    gq_block += declarations

    # random parameters with a fixed level have to be sampled before the main loop
    level_params = [
        p for p in uncorr_params
        if p.get_type() == "random" and p.level is not None and p.level_type == "fixed"
    ]
    levels = util.unique([p.level for p in level_params])
    # make sure that levels are sorted as they occur in the covariate list
    levels.sort(key=lambda x: covs.index(x))

    for level in levels:
        # collect parameter for this level
        levpars = [p for p in level_params if p.level == level]
        if len(levpars) == 0:
            continue
        # create sampling statements for selected params
        num_params = level.num_cat_var
        trans_func = lambda x: sl.expandVar(x, (num_params,)).idx(r)
        levpar_rng_stmts = [
            p.genstmt_gq_simulator(transform_func=trans_func) for p in levpars
        ]
        lev_loop_content = util.flatten(levpar_rng_stmts)
        lev_loop = sl.ForLoop(
            r, sl.Range(sl.one(), num_params), sl.Scope(lev_loop_content)
        )
        gq_block.append(
            sl.comment(f"sample random parameters with fixed level {level.name}")
        )
        gq_block.append(lev_loop)

    rand_level_params = [
        p
        for p in uncorr_params
        if p.get_type() == "random" and p.level is not None and p.level_type == "random"
    ]
    if len(rand_level_params) > 0:
        gq_block.append(sl.comment("sample parameters with a random level"))
    for p in rand_level_params:
        gq_block += p.genstmt_gq_simulator()

    # compute population-level trnasformed parameters
    trans_params = [p for p in params if p.get_type() == "trans"]
    trans_params.sort(key=lambda x: x.get_rank())
    trans_par_stmts = [p.genstmt_gq_simulator() for p in trans_params]
    if trans_params:
        gq_block.append(sl.comment("calculate transformed parameters"))
    gq_block += util.flatten(trans_par_stmts)

    # main for loop
    main_loop_content: list[sl.Stmt] = []

    # sample random parameters with no level (i.e. the unit level)
    trans_func = lambda x: sl.expandVar(x, (R,)).idx(r)
    par_rng_stmts = [
        p.genstmt_gq_simulator(transform_func=trans_func)
        for p in uncorr_params
        if p.get_type() == "random" and p.level is None
    ]
    main_loop_content.append(sl.comment("sample random parameters"))
    main_loop_content += util.flatten(par_rng_stmts)

    # sample correlated random parameters
    corr_par_rng_stmts = [
        p.genstmt_gq_simulator(transform_func=trans_func) for p in corr_params
    ]
    if corr_par_rng_stmts:
        main_loop_content.append(
            sl.EmptyStmt(comment="sample random correlated parameters")
        )
    main_loop_content += util.flatten(corr_par_rng_stmts)

    # compute individual-level trnasformed parameters
    trans_params = [p for p in params if p.get_type() == "trans_indiv"]
    trans_params.sort(key=lambda x: x.get_rank())
    trans_par_stmts = [p.genstmt_gq_simulator() for p in trans_params]
    if trans_params:
        main_loop_content.append(sl.comment("calculate transformed parameters"))
    main_loop_content += util.flatten(trans_par_stmts)

    ivp_args = [Time_r] + ivp_par_vars
    sol = sl.Var(sl.Array(sl.Vector(state_dim), [Nr]), "sol")
    solve_ivp = sl.DeclAssign(sol, sl.Call("solve_ivp", ivp_args))

    main_loop_content.append(sl.EmptyStmt(comment="solve initial value problem"))
    main_loop_content.append(solve_ivp)

    sim_asgns = [
        sl.Assign(
            sl.MultiIndexOp(sim_obs[i], [r, n]),
            sl.Call(ob.name + "_rng", [sol.idx(n)] + rngs_par_vars[i]),
            comment="simulate data at observation times",
        )
        for i, ob in enumerate(obs)
    ]

    sim_asgns_loop = sl.ForLoop(n, sl.Range(sl.one(), Nr), sl.Scope(sim_asgns))

    main_loop_content.append(sim_asgns_loop)

    main_loop = sl.ForLoop(r, sl.Range(sl.one(), R), sl.Scope(main_loop_content))

    gq_block.append(sl.EmptyStmt(comment="simulate for each unit"))
    gq_block.append(main_loop)

    return gq_block


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
    options: dict,
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
            options,
            init_parnames,
            odes_parnames,
            trans_parnames,
            dists_parnames,
        ),
        gen_data_block(params, corr_params, covs, options),
        gen_trans_data_block(params),
        gen_parameters_block(),
        None,  # gen_trans_param_block(),
        gen_model_block(),
        gen_gq_block(
            params,
            corr_params,
            state,
            trans_state,
            obs,
            covs,
            options,
            ivp_parnames,
            dists_parnames,
        ),
    )

    return model
