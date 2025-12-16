from .parameter import Parameter
from .distribution import Distribution
from .observation import Observation
from . import utilities as util
from . import stanlexer
from .state import StateVar, State
from . import stanlang as sl
from . import deparse
from . import gencommon

from functools import reduce


def find_used_names_in_stmts(stmts: list[sl.Stmt], names: list[str]) -> list[str]:
    """
    Find used names in list of statements

    TODO: move this to another module
    """
    used_names = []
    for stmt in stmts:
        code = deparse.deparse_stmt(stmt)
        used_names += stanlexer.find_used_names(code, names)
    # make unique, but keep order the same as in names
    return [name for name in names if name in used_names]


def gen_count_init_obs_function() -> sl.FuncDef:
    func_body: list[sl.Stmt] = []

    N = sl.intVar("N")
    n = sl.intVar("n")
    Time = sl.Var(sl.Array(sl.Real(), (N,)), "Time")
    t0 = sl.realVar("t0")

    func_body.append(sl.DeclAssign(N, sl.Call("num_elements", [Time])))

    func_body.append(sl.EmptyStmt(comment="count observations at time t <= t0"))

    count_tzero_obs_loop = sl.ForLoop(
        n,
        sl.Range(sl.one(), N),
        sl.IfStatement(sl.GrOp(Time.idx(n), t0), sl.Return(n - sl.one())),
    )

    func_body.append(count_tzero_obs_loop)
    func_body.append(sl.Return(N))

    func_def = sl.FuncDef(sl.Int(), "count_init_obs", [Time, t0], func_body)

    return func_def


def gen_logitnormal_lpdf() -> sl.FuncDef:
    """
    This is the simplest version of the logitnormal_lpdf function,
    Both the random variable and the location parameter are scalars.

    Returns
    -------
    sl.FuncDef
        Stan function definition for logitnormal_lpdf
    """
    x = sl.realVar("x")
    z = sl.realVar("z")
    l = sl.realVar("l")
    u = sl.realVar("u")
    mu = sl.realVar("mu")
    sigma = sl.realVar("sigma")

    func_body: list[sl.Stmt] = []
    func_body.append(sl.DeclAssign(z, sl.Par(x - l) / sl.Par(u - l)))

    neg_jac = sl.Call("log", z * sl.Par(sl.realVar(1.0) - z) * sl.Par(u - l))

    ret = sl.Return(
        sl.PCall("normal", sl.Call("logit", z), [mu, sigma], "lpdf") - neg_jac
    )

    func_body.append(ret)

    func_def = sl.FuncDef(
        sl.Real(), "logitnormal_lpdf", [x, mu, sigma, l, u], func_body
    )

    return func_def


def gen_vectorized_logitnormal_lpdf(array_loc=False) -> sl.FuncDef:
    """
    Generate a vectorized version of the logitnormal_lpdf function.

    If array_loc is True, the function will be generated with an array location
    parameter, i.e., the location parameters will be vectors of the same length
    as the input x. Otherwise, the location parameters will be a scalar.
    Stan allows function overloading, so both versions can be used in the same
    Stan program.
    """
    n = sl.intVar("n")
    x = sl.Var(sl.Array(sl.Real(), n), "x")
    nx = sl.Call("num_elements", x)
    z = sl.Var(sl.Vector(nx), "z")
    l = sl.realVar("l")
    u = sl.realVar("u")
    if array_loc:
        mu = sl.Var(sl.Array(sl.Real(), n), "mu")
    else:
        mu = sl.realVar("mu")
    sigma = sl.realVar("sigma")

    func_body: list[sl.Stmt] = []
    func_body.append(
        sl.DeclAssign(z, sl.Par(sl.Call("to_vector", x) - l) / sl.Par(u - l))
    )

    neg_jac = sl.Call(
        "log", sl.TransposeOp(z) * sl.Par(sl.realVar(1.0) - z) * sl.Par(u - l)
    )

    ret = sl.Return(
        sl.PCall("normal", sl.Call("logit", z), [mu, sigma], "lpdf") - neg_jac
    )

    func_body.append(ret)

    func_def = sl.FuncDef(
        sl.Real(), "logitnormal_lpdf", [x, mu, sigma, l, u], func_body
    )

    return func_def


def gen_logitnormal_rng() -> sl.FuncDef:
    z = sl.realVar("z")
    l = sl.realVar("l")
    u = sl.realVar("u")
    mu = sl.realVar("mu")
    sigma = sl.realVar("sigma")

    func_body: list[sl.Stmt] = []
    func_body.append(sl.DeclAssign(z, sl.Call("normal_rng", [mu, sigma])))

    ret = sl.Return(sl.Call("inv_logit", z) * sl.Par(u - l) + l)

    func_body.append(ret)

    func_def = sl.FuncDef(sl.Real(), "logitnormal_rng", [mu, sigma, l, u], func_body)

    return func_def


def gen_ivp_solver_function(
    params: list[Parameter],
    state: list[StateVar],
    trans_state: list[StateVar],
    options: dict,
    init_parnames: list[str],
    odes_parnames: list[str],
    trans_parnames: list[str],
) -> sl.FuncDef:
    # get some data
    init_obs = options["init_obs"]
    integrator = options["integrator"]
    if integrator.endswith("_tol"):
        rel_tol = sl.LiteralReal(options["rel_tol"])
        abs_tol = sl.LiteralReal(options["abs_tol"])
        max_num_steps = sl.LiteralInt(options["max_num_steps"])
        tol_pars = [rel_tol, abs_tol, max_num_steps]
    else:
        # the integrator signature does not expect any tolerance parameters.
        tol_pars = []

    ode_state = State(state)
    ext_state = State(state + trans_state)

    ostate_dim = ode_state.flat_dim()
    estate_dim = ext_state.flat_dim()

    # find parameters required for function calls
    init_params = [p for p in params if p.name in init_parnames]
    odes_params = [p for p in params if p.name in odes_parnames]
    trans_params = [p for p in params if p.name in trans_parnames]

    # common variables
    N = sl.Var(sl.Int(), "N")
    n = sl.Var(sl.Int(), "n")
    Time = sl.Var(sl.Array(sl.Real(), (N,), data=True), "Time")
    t0 = sl.LiteralReal(0.0)

    ode_fun = sl.Var(
        sl.Function(sl.Vector(ostate_dim), ()), "ode_fun"
    )  # FIXME: arguments

    # parameter declarations and list
    trans_vars = [p.var for p in trans_params]

    init_vars: list[sl.Expr] = [p.var for p in init_params]
    init_state = sl.Var(sl.Vector(ostate_dim), "init_state")
    ret_type = sl.Array(sl.Vector(estate_dim), (N,))
    sol = sl.Var(ret_type, "sol")

    odes_vars = [p.var for p in odes_params]

    prep_stmt = [
        sl.DeclAssign(
            N, sl.Call("num_elements", [Time]), comment="number of time points"
        ),
        sl.DeclAssign(
            init_state, sl.Call("gen_init", init_vars), comment="generate initial state"
        ),
        sl.Decl(sol, comment="allocate space for solution"),
    ]

    integration_stmt: list[sl.Stmt] = []
    PosTime = Time.idx(sl.Range(sl.one(), N))
    PosRange = sl.Range(sl.one(), N)
    # the data might contain observations at time T0
    if init_obs:
        NZero = sl.intVar("NZero")
        # adjust PosRange and PosTime
        PosTime = Time.idx(sl.Range(sl.one() + NZero, N))
        PosRange = sl.Range(sl.one() + NZero, N)

        integration_stmt.append(
            sl.EmptyStmt(comment="count observations at time t = t0")
        )
        count_init_obs_fun_call = sl.Call("count_init_obs", [Time, t0])
        integration_stmt.append(sl.DeclAssign(NZero, count_init_obs_fun_call))

        integration_stmt.append(
            sl.Assign(
                sl.MultiIndexOp(
                    sol, [sl.Range(sl.one(), NZero), sl.Range(sl.one(), ostate_dim)]
                ),
                sl.Call("rep_array", [init_state, NZero]),
            )
        )

    if len(state) == 0:
        # if there are no state variables, we don't have to use the ODE
        # solver, and can directly compute the transformed state.
        # This is useful for when the user has manually solved the 
        # system of ODEs.
        integration_stmt.append(sl.comment("No ODEs have to be integrated."))
    else:
        # if there are state variables, we have to use the ODE solver
        # to compute the solution.
        integration_stmt.append(
            sl.Assign(
                sl.MultiIndexOp(sol, [PosRange, sl.Range(sl.one(), ostate_dim)]),
                sl.Call(
                    integrator,
                    [ode_fun, init_state, t0, PosTime] + tol_pars + odes_vars,
                ),
                comment="solve initial value problem",
            )
        )

    # in the case that we have transformed state variables, we have to do some extra work
    if len(trans_state) == 0:
        return_stmt: list[sl.Stmt] = [sl.Return(sol, comment="return the solution")]
    else:
        trans_range = sl.Range(ostate_dim + sl.one(), estate_dim)
        sol_slice = sl.MultiIndexOp(sol, [n, sl.Range(sl.one(), ostate_dim)])

        return_stmt = [
            sl.ForLoop(
                n,
                sl.Range(sl.one(), N),
                sl.Assign(
                    sl.MultiIndexOp(sol, [n, trans_range]),
                    sl.Call(
                        "state_transform",
                        [Time.idx(n), sol_slice] + trans_vars,
                    ),
                    comment="compute transformation of the state and append to the trajectory",
                ),
            ),
            sl.Return(sol, comment="and return the result"),
        ]

    # list all parameters used by init, ode or trans
    ivp_parnames = util.unique(init_parnames + odes_parnames + trans_parnames)
    # create the Stan variables corresponding to the used parameters
    par_vars = [p.var for p in params if p.name in ivp_parnames]
    func_args = [Time] + par_vars
    func_body = prep_stmt + integration_stmt + return_stmt

    ivp_fun = sl.FuncDef(ret_type, "solve_ivp", func_args, func_body)

    return ivp_fun


def gen_wrapper_function(
    params: list[Parameter],
    state: list[StateVar],
    trans_state: list[StateVar],
    options: dict,
    init_parnames: list[str],
    odes_parnames: list[str],
    trans_parnames: list[str],
) -> sl.FuncDef:
    """
    create the 'map_rect_helper_fun' that is used for solving IVPs in parallel.
    """

    ## get some data
    ext_state = State(state + trans_state)
    estate_dim = ext_state.flat_dim()

    # determine which parameters are used in the multi-threaded part of the code
    ivp_params = [
        p
        for p in params
        if p.name in set(init_parnames + odes_parnames + trans_parnames)
    ]
    ivp_pardict = {p.name : p for p in ivp_params}

    ## make a list of parameter arguments, and choose from upar or ppar and the right index

    random_parnames = [p.name for p in ivp_params if gencommon.treat_as_random(p)]
    fixed_parnames = [p.name for p in ivp_params if gencommon.treat_as_fixed(p)]
    const_parnames = [
        p.name for p in ivp_params if p.get_type() in ["const", "const_indiv"]
    ]

    # we have to know the flattened dimmentions of the parameters (1 for scalars)

    random_flat_dims = [
        p.var.var_type.flat_dim() for p in ivp_params if p.name in random_parnames
    ]

    fixed_flat_dims = [
        p.var.var_type.flat_dim() for p in ivp_params if p.name in fixed_parnames
    ]

    const_flat_dims = [
        p.var.var_type.flat_dim() for p in ivp_params if p.name in const_parnames
    ]

    def sum_dims(dims):
        return sl.izero() if len(dims) == 0 else reduce(lambda a, b: a + b, dims)

    kr = sum_dims(random_flat_dims)  # number of flat real random parameters
    kc = sum_dims(const_flat_dims)  # number of flat real constants
    kf = sum_dims(fixed_flat_dims)  # number of flat real fixed parameters

    ki = sl.LiteralInt(1)  ## FIXME: take shapes into account

    upar = sl.Var(sl.Vector(kr), "upar")
    ppar = sl.Var(sl.Vector(kf), "ppar")
    rdat = sl.Var(sl.Array(sl.Real(), (kc,), data=True), "rdat")
    idat = sl.Var(sl.Array(sl.Int(), (ki,), data=True), "idat")

    ## we have to make sure the parameters are taken from the right arrays

    def unflatten_parameter(i, pn, flat_dims, flat_var):
        """
        aux function for filling par_arg_dict below.
        get the right slice of flat_var (ppar, rdat, ...)
        and reshape it into the correct shape.
        """
        par = ivp_pardict[pn]
        match par.space:
            case "real":
                idx = reduce(lambda a, b: a + b, flat_dims[:i], sl.one())
                return flat_var.idx(idx)
            case "vector":
                idx = sl.Range(
                    reduce(lambda a, b: a + b, flat_dims[:i], sl.one()),
                    reduce(lambda a, b: a + b, flat_dims[: i + 1]),
                )
                # TODO: ppar already is a vector, so no casting is required
                return sl.Call("to_vector", flat_var.idx(idx))
            case "matrix":
                idx = sl.Range(
                    reduce(lambda a, b: a + b, flat_dims[:i], sl.one()),
                    reduce(lambda a, b: a + b, flat_dims[: i + 1]),
                )
                tp = par.var.var_type
                return sl.Call("to_matrix", [flat_var.idx(idx), tp.n, tp.m])
            case _:
                raise Exception(f"invalid parameter space given: {par.space}")

    par_arg_dict = {}

    for i, pn in enumerate(random_parnames):
        par_arg_dict[pn] = unflatten_parameter(i, pn, random_flat_dims, upar)
    for i, pn in enumerate(fixed_parnames):
        par_arg_dict[pn] = unflatten_parameter(i, pn, fixed_flat_dims, ppar)
    for i, pn in enumerate(const_parnames):
        par_arg_dict[pn] = unflatten_parameter(i, pn, const_flat_dims, rdat)

    par_args = [par_arg_dict[p.name] for p in ivp_params]

    # idat contains shapes for non-scalar parameters
    shape_len_dict = {"real": 0, "vector": 1, "matrix": 2}
    shape_lens = [shape_len_dict[p.space] for p in ivp_params]

    par_shape_dict = {}  # shapes of vector and matrix valued parameters
    for i, par in enumerate(ivp_params):
        pn = par.name
        match par.space:
            case " real":
                pass
            case "vector":
                idx = sl.LiteralInt(1 + sum(shape_lens[: i + 1]))
                par_shape_dict[pn] = sl.DeclAssign(par.shape, idat.idx(idx))
            case "matrix":
                idx = sl.Range(
                    sl.LiteralInt(1 + sum(shape_lens[:i]) + 1),
                    sl.LiteralInt(1 + sum(shape_lens[: i + 1])),
                )
                par_shape_dict[pn] = sl.DeclAssign(par.shape, idat.idx(idx))

    N = sl.Var(sl.Int(), "N")
    n = sl.Var(sl.Int(), "n")

    res = sl.Var(sl.Vector(sl.Par(estate_dim) * N), "res")

    func_args = [ppar, upar, rdat, idat]
    sol_times = rdat.idx(sl.Range(sl.one() + kc, N + kc))

    res_range = sl.Range(
        sl.Par(n - sl.one()) * sl.Par(estate_dim) + sl.one(), n * sl.Par(estate_dim)
    )

    sol = sl.Var(sl.Array(sl.Vector(estate_dim), (N,)), "sol")

    func_body: list[sl.Stmt] = []
    func_body.append(
        sl.DeclAssign(N, idat.idx(sl.LiteralInt(1)), comment="number of time points")
    )

    # extract parameter shapes for non-scalar params
    if len(par_shape_dict) > 0:
        func_body.append(sl.comment("find shapes of non-scalar parameters"))
    for p in ivp_params:
        pn = p.name
        if pn not in par_shape_dict:
            continue  # scalar value!
        func_body.append(par_shape_dict[pn])

    ivp_par_args = [sol_times] + par_args

    func_body += [
        sl.comment("solve initial value problem"),
        sl.DeclAssign(sol, sl.Call("solve_ivp", ivp_par_args)),
        sl.comment("concatenate solution into a vector"),
        sl.Decl(res),
        sl.ForLoop(
            n,
            sl.Range(sl.one(), N),
            sl.Assign(res.idx(res_range), sol.idx(n)),
        ),
        sl.Return(res),
    ]

    aux_fun = sl.FuncDef(res.var_type, "map_rect_helper_fun", func_args, func_body)
    return aux_fun


def gen_ode_function(
    odes: sl.MixinStmt,
    params: list[Parameter],
    state: list[StateVar],
    options: dict,
    odes_parnames: list[str],
) -> sl.FuncDef:
    func_body: list[sl.Stmt] = []
    # find parameters required for function calls
    odes_params = [p for p in params if p.name in odes_parnames]
    par_args = [p.var for p in odes_params]

    t = sl.Var(sl.Real(), "t")
    ode_state = State(state)  ## FIXME: define State in Model object?
    only_scalars = ode_state.all_scalar()  # if True, we can simplify the code
    state_var = sl.Var(sl.Vector(ode_state.flat_dim()), "state")
    deriv_rename_func = lambda name: f"ddt_{name}"
    deriv = sl.Var(sl.Vector(ode_state.flat_dim()), "ddt_state")
    # deriv_literal is only valid if all variables are scalars
    deriv_literal = ode_state.gen_literal_vector(rename_func=deriv_rename_func)

    func_body.append(sl.comment("unpack the state variables"))
    func_body += ode_state.gen_unpack_stmt(state_var)

    func_body.append(sl.comment("declare derivatives of state variables"))
    func_body += ode_state.gen_decl(rename_func=deriv_rename_func)

    if not only_scalars:
        func_body.append(sl.comment("declare flat return vector"))
        func_body.append(sl.Decl(deriv))

    func_body.append(sl.comment("user-defined ODEs"))
    func_body.append(odes)

    if not only_scalars:
        func_body.append(sl.comment("combine components into flat vector"))
        func_body += ode_state.gen_flatten(deriv.name, rename_func=deriv_rename_func)
        func_body.append(sl.Return(deriv))
        return_type = deriv.var_type
    else:
        func_body.append(sl.comment("return literal vector with derivatives"))
        func_body.append(sl.Return(deriv_literal))
        return_type = deriv_literal.literal_type

    func_args = [t, state_var] + par_args
    ode_fun = sl.FuncDef(return_type, "ode_fun", func_args, func_body)
    return ode_fun


def gen_init_function(
    init: sl.MixinStmt,
    params: list[Parameter],
    state: list[StateVar],
    options: dict,
    init_parnames: list[str],
) -> sl.FuncDef:
    func_body: list[sl.Stmt] = []

    # find parameters required for function call
    init_params = [p for p in params if p.name in init_parnames]
    par_args = [p.var for p in init_params]

    ode_state = State(state)
    only_scalars = ode_state.all_scalar()  # if True, we can simplify the code
    rename_func = lambda name: f"{name}_0"
    init_state_var = sl.Var(sl.Vector(ode_state.flat_dim()), "state_0")
    # init literal is only valid in the case only_scalars == True
    init_literal = ode_state.gen_literal_vector(rename_func=rename_func)

    func_body.append(sl.comment("declare initial variables"))
    func_body += ode_state.gen_decl(rename_func=rename_func)

    if not only_scalars:
        func_body.append(
            sl.Decl(init_state_var, comment="declare flat initial state vector")
        )

    func_body.append(sl.comment("user-defined initial condition"))
    func_body.append(init)

    if not only_scalars:
        func_body.append(sl.comment("pack initial state components into flat vector"))
        func_body += ode_state.gen_flatten(init_state_var.name, rename_func=rename_func)
        func_body.append(sl.Return(init_state_var))
        return_type = init_state_var.var_type
    else:
        func_body.append(
            sl.comment("return literal vector with initival state variables")
        )
        func_body.append(sl.Return(init_literal))
        return_type = init_literal.literal_type

    init_function = sl.FuncDef(return_type, "gen_init", par_args, func_body)

    return init_function


def gen_trans_function(
    trans_state: list[StateVar],
    transform: sl.MixinStmt,
    params: list[Parameter],
    state: list[StateVar],
    options: dict,
    trans_parnames: list[str],
) -> sl.FuncDef:
    """generate stan function that creates a vector of transformed state variables"""
    func_body: list[sl.Stmt] = []
    # find parameters required for function call
    trans_params = [p for p in params if p.name in trans_parnames]

    par_args = [p.var for p in trans_params]
    ode_state = State(state)
    ostate_dim = ode_state.flat_dim()
    ostate_var = sl.Var(sl.Vector(ostate_dim), "state")

    tra_state = State(trans_state)
    only_scalars = tra_state.all_scalar()  # if True, we can simplify the code
    tstate_dim = tra_state.flat_dim()
    tstate_var = sl.Var(sl.Vector(tstate_dim), "transformed_state")
    tra_literal = tra_state.gen_literal_vector()

    # find required state variable names
    statevar_names = stanlexer.find_used_names(
        deparse.deparse_stmt(transform), [sv.name for sv in state]
    )

    time_var = sl.Var(sl.Real(), "t")

    func_body.append(sl.comment("unpack required state variables"))
    func_body += ode_state.gen_unpack_stmt(ostate_var, statevar_names=statevar_names)

    func_body.append(sl.comment("declare transformed state variables"))
    func_body += tra_state.gen_decl()

    if not only_scalars:
        func_body.append(sl.comment("declare flat return vector"))
        func_body.append(sl.Decl(tstate_var))

    func_body.append(sl.comment("user-defined transformation code"))
    func_body.append(transform)

    if not only_scalars:
        func_body.append(sl.comment("combine components into flat vector"))
        func_body += tra_state.gen_flatten(tstate_var.name)
        func_body.append(sl.Return(tstate_var))
        return_type = tstate_var.var_type
    else:
        func_body.append(sl.comment("return literal vector with transformed variables"))
        func_body.append(sl.Return(tra_literal))
        return_type = tra_literal.literal_type

    trans_func = sl.FuncDef(
        return_type, "state_transform", [time_var, ostate_var] + par_args, func_body
    )

    return trans_func


def gen_loglik_function(
    dists: list[Distribution],
    params: list[Parameter],
    state_vars: list[StateVar],
    trans_state_vars: list[StateVar],
    obs: list[Observation],
    options: dict,
    loglik_parnames: list[str],
) -> sl.FuncDef:
    func_body: list[sl.Stmt] = []
    # list of names of observations (and optional censor codes)
    obs_vars = [sl.Var(ob.obs_type, ob.name) for ob in obs]
    cc_vars = [sl.Var(sl.Int(), ob.cc_name) for ob in obs if ob.censored]

    obs_args = obs_vars + cc_vars

    # list of parameters
    loglik_params = [p for p in params if p.name in loglik_parnames]
    par_args = [p.var for p in loglik_params]

    # make sure that state and transformed state variables are defined
    state = State(state_vars + trans_state_vars)
    state_var = sl.Var(sl.Vector(state.flat_dim()), "state")

    # create loglik statements and analyze
    loglik_stmts = [dist.genstmt_loglik() for dist in dists]
    sv_names = [sv.name for sv in state_vars + trans_state_vars]
    used_sv_names = find_used_names_in_stmts(loglik_stmts, sv_names)

    func_body.append(sl.comment("unpack required state variables"))
    func_body += state.gen_unpack_stmt(state_var, statevar_names=used_sv_names)

    func_body.append(sl.comment("declare log-lik variables"))
    # list of names to be returned in a vector
    loglik_vars = [sl.Var(sl.Real(), f"ll_{ob.name}") for ob in obs]
    # declare individual log-lik terms (and set to zero be default!)
    func_body += [sl.DeclAssign(lv, sl.rzero()) for lv in loglik_vars]

    func_body.append(sl.comment("user-defined log-likelihood"))
    func_body += loglik_stmts

    func_body.append(sl.Return(sl.LiteralVector(loglik_vars)))

    ll_func = sl.FuncDef(
        sl.Vector(sl.LiteralInt(len(obs))),  ## return type
        "loglik_fun",  ## function name
        obs_args + [state_var] + par_args,  ## arguments
        func_body,
    )

    return ll_func


def gen_rng_function(
    dist: Distribution,
    params: list[Parameter],
    state_vars: list[StateVar],
    trans_state_vars: list[StateVar],
    rng_parnames: list[str],
) -> sl.FuncDef:
    func_body: list[sl.Stmt] = []

    obs_type = dist.obs.obs_type
    obs_name = dist.obs.name
    obs_var = sl.Var(obs_type, obs_name)

    rng_params = [p for p in params if p.name in rng_parnames]
    par_args = [p.var for p in rng_params]

    # make sure that state and transformed state variables are defined
    state = State(state_vars + trans_state_vars)
    state_var = sl.Var(sl.Vector(state.flat_dim()), "state")

    rng_stmt = dist.genstmt_rng()
    sv_names = [sv.name for sv in state_vars + trans_state_vars]
    used_sv_names = find_used_names_in_stmts([rng_stmt], sv_names)

    func_body.append(sl.EmptyStmt(comment="unpack required state variables"))
    func_body += state.gen_unpack_stmt(state_var, statevar_names=used_sv_names)

    func_body += [
        sl.EmptyStmt(comment="declare variable to-be-returned"),
        sl.Decl(obs_var),
        sl.EmptyStmt(comment="user-defined sampler"),
        rng_stmt,
        sl.Return(obs_var),
    ]

    rng_func = sl.FuncDef(
        obs_type,  ## function's return type
        f"{obs_name}_rng",  ## function's name
        [state_var] + par_args,  ## function's arguments
        func_body,  ## function's body
    )
    return rng_func
