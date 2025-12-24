from . import stanlang as sl
from . import utilities as util
from .parameter import Parameter


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
    if len(level_matrices) == 0:
        return level_matrix_decls, level_matrix_defs
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


def gen_covariate_restrictions(params: list[Parameter]) -> list[sl.Stmt]:
    """
    Find parameters with covariates and a fixed level
    and generate statements to restrict covariate values based on the level.
    Avoid adding duplicates...

    TODO: this is only for categorical covariates right now.
    but would also make sense for continuous covariates!!
    """
    stmts: list[sl.Stmt] = []
    pairs_added = set()
    random_params = [p for p in params if p.get_type() == "random"]
    for p in random_params:
        if p._catcovs and p.level is not None and p.level_type == "fixed":
            for cov in p._catcovs:
                if (p.level.name, cov.name) not in pairs_added:
                    pairs_added.add((p.level.name, cov.name))
                    stmts += cov.genstmt_transformed_data(p.level)
    return stmts



def treat_as_random(p):
    """
    rules for how to group types of parameters based on their indexing requirements
    """
    if p.get_type() in ["random", "indiv", "trans_indiv"]:
        return True
    if p.get_type() == "fixed" and p.has_covs():
        return True
    return False


def treat_as_fixed(p):
    if p.get_type() == "trans" or (p.get_type() == "fixed" and not p.has_covs()):
        return True
    return False


def gen_all_trans_param_functions(params: list[Parameter]) -> list[sl.FuncDef]:
    """
    Get all transformed parameters from the parameter list and make function 
    definitions. Include a comment line.
    """
    functions_block: list[sl.Stmt] = []
    trans_params = [par for par in params if par.get_type() in ["trans", "trans_indiv"]]
    trans_par_functions = util.flatten([
        par.genstmt_functions() for par in trans_params
    ])
    comment_trans_par = "parameter transform" + ("s" if len(trans_par_functions) > 1 else "")
    if trans_par_functions:
        functions_block.append(sl.comment(comment_trans_par))    
    functions_block += trans_par_functions
    return functions_block
