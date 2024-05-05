from . import stanlang as sl
from . import utilities as util
from .parameter import Parameter
from . import stanlexer, deparse



def gen_level_matrices(params: list[Parameter]) -> tuple[list[sl.Stmt], list[sl.Stmt]]:
    RVar = sl.intVar("R")
    # if there are random parameters with a random level, we need to define boolean matrices
    random_levels = util.unique(
        [
            p.level
            for p in params
            if p.get_type() == "random"
            and p.level is not None
            and p.level_type == "random"
        ]
    )
    level_matrices = [
        sl.Var(sl.Matrix(RVar, RVar), f"level_{lev.name}") for lev in random_levels
    ]
    level_matrix_decls = [sl.Decl(lev_mat) for lev_mat in level_matrices]
    level_matrix_defs = []
    r1Var, r2Var = sl.intVar("r1"), sl.intVar("r2")
    for lev, lev_mat in zip(random_levels, level_matrices):
        level_matrix_defs.append(
            sl.Assign(
                lev_mat.idx(r1Var, r2Var),
                sl.Par(sl.EqOp(lev.var.idx(r1Var), lev.var.idx(r2Var))),
            )
        )
    level_inner_loop = sl.ForLoop(
        r2Var, sl.Range(sl.one(), RVar), sl.Scope(level_matrix_defs)
    )
    level_outer_loop = sl.ForLoop(r1Var, sl.Range(sl.one(), RVar), level_inner_loop)

    return level_matrix_decls, [level_outer_loop]



def expand_and_index_param(p, apply_idx=False):
    """
    Helper function for correctly expanding and indexing individual parameters
    Used in gen_model_block (for log-likelihood function parameters)
    and gen_gq_block (for integrating ODEs and computing LLs)
    """
    R = sl.intVar("R")
    r = sl.intVar("r")
    match p.get_type():
        case "random":
            has_fixed_level = p.level is not None and p.level_type == "fixed"
            idx = p.level.var.idx(r) if has_fixed_level else r
            num_params = p.level.num_cat_var if has_fixed_level else R
            if apply_idx:
                return sl.expandVar(p.var, (num_params,)).idx(idx)
            return sl.expandVar(p.var, (num_params,))
        case "const_indiv" | "indiv":
            if apply_idx:
                return sl.expandVar(p.var, (R,)).idx(r)
            return sl.expandVar(p.var, (R,))
        case "fixed":
            if p.level is None:
                return p.var
            idx = p.level.var.idx(r)
            num_params = p.level.num_cat_var
            if apply_idx:
                return sl.expandVar(p.var, num_params).idx(idx)
            return sl.expandVar(p.var, num_params)
        case _:
            return p.var



def find_varnames(exprs: list[sl.Expr], names: list[str]):
    """
    Auxilary function used in gen_stan_model in the genmodel and gensimulator
    modules. It takes a list of stanlang sxpressios and returns the names
    of all used variables.
    """
    
    # FIXME: instead of deparsing and tokenizing, find variables in an expression directly

    unique_parnames = util.unique(
        util.flatten(
            [
                stanlexer.find_used_names(deparse.deparse_expr(expr), names)
                for expr in exprs
            ]
        )
    )
    return unique_parnames
