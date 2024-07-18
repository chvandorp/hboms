from . import stanlang as sl
from . import utilities as util


def find_variables_expr(expr: sl.Expr) -> list[sl.Var]:
    match expr:
        case sl.LiteralInt(val):
            return []
        case sl.LiteralReal(val):
            return []
        case sl.LiteralComplex(val):
            return []
        case sl.LiteralString(val):
            return []
        case sl.LiteralRowVector(elts):
            return util.flatten([find_variables_expr(elt) for elt in elts])
        case sl.LiteralVector(elts):
            return util.flatten([find_variables_expr(elt) for elt in elts])
        case sl.LiteralArray(elts):
            return util.flatten([find_variables_expr(elt) for elt in elts])
        case sl.Var(var_type, name):
            return [sl.Var(var_type, name)] + find_variables_type(var_type)
        case sl.Range(start, stop):
            return find_variables_expr(start) + find_variables_expr(stop)
        case sl.TransposeOp(val):
            return find_variables_expr(val)
        case sl.Negate(val):
            return find_variables_expr(val)
        case sl.MulOp(left, right):
            return find_variables_expr(left) + find_variables_expr(right)
        case sl.AddOp(left, right):
            return find_variables_expr(left) + find_variables_expr(right)
        case sl.SubOp(left, right):
            return find_variables_expr(left) + find_variables_expr(right)
        case sl.EltMulOp(left, right):
            return find_variables_expr(left) + find_variables_expr(right)
        case sl.DivOp(left, right):
            return find_variables_expr(left) + find_variables_expr(right)
        case sl.EltDivOp(left, right):
            return find_variables_expr(left) + find_variables_expr(right)
        case sl.ModuloOp(dividend, divisor):
            return find_variables_expr(dividend) + find_variables_expr(divisor)
        case sl.IDivOp(left, right):
            return find_variables_expr(left) + find_variables_expr(right)
        case sl.PowOp(base, exponent):
            return find_variables_expr(base) + find_variables_expr(exponent)
        case sl.EltPowOp(base, exponent):
            return find_variables_expr(base) + find_variables_expr(exponent)
        case sl.MultiIndexOp(var, indices):
            return find_variables_expr(var) + util.flatten(
                [find_variables_expr(index) for index in indices]
            )
        case sl.IndexOp(var, index):
            return find_variables_expr(var) + find_variables_expr(index)
        case sl.TupleIndexOp(var, index):
            return find_variables_expr(var) + find_variables_expr(index)
        case sl.EqOp(left, right):
            return find_variables_expr(left) + find_variables_expr(right)
        case sl.NeqOp(left, right):
            return find_variables_expr(left) + find_variables_expr(right)
        case sl.LeOp(left, right):
            return find_variables_expr(left) + find_variables_expr(right)
        case sl.GrOp(left, right):
            return find_variables_expr(left) + find_variables_expr(right)
        case sl.LeEqOp(left, right):
            return find_variables_expr(left) + find_variables_expr(right)
        case sl.GrEqOp(left, right):
            return find_variables_expr(left) + find_variables_expr(right)
        case sl.TernaryOp(cond, expr_if_true, expr_if_false):
            return (
                find_variables_expr(cond)
                + find_variables_expr(expr_if_true)
                + find_variables_expr(expr_if_false)
            )
        case sl.LogicOrOp(left, right):
            return find_variables_expr(left) + find_variables_expr(right)
        case sl.LogicAndOp(left, right):
            return find_variables_expr(left) + find_variables_expr(right)
        case sl.Par(content):
            return find_variables_expr(content)
        case sl.Call(func_name, arguments):
            return util.flatten([find_variables_expr(arg) for arg in arguments])
        case sl.PCall(dist_name, obs, parameters, suffix):
            return find_variables_expr(obs) + util.flatten(
                [find_variables_expr(par) for par in parameters]
            )
        case sl.MixinExpr(code):  # TODO can't find variables in a MixinExpr?
            return []
        case _:
            raise Exception("could not find variables in expression", expr)


def find_variables_type(typ: sl.Type) -> list[sl.Var]:
    """
    Find variables in Stan types (bounds and dimensions)
    """
    match typ:
        case sl.BoundedType(lower=lower, upper=upper):
            # optmize the optional bounds
            lower_vars = find_variables_expr(lower) if lower is not None else []
            upper_vars = find_variables_expr(upper) if upper is not None else []
            # match typ again for all bounded types
            match typ:
                case sl.Real():
                    dim_vars = []
                case sl.Int():
                    dim_vars = []
                case sl.Vector(n):
                    dim_vars = find_variables_expr(n)
                case sl.RowVector(n):
                    dim_vars = find_variables_expr(n)
                case sl.Matrix(n, m):
                    dim_vars = find_variables_expr(n) + find_variables_expr(m)
                case _:
                    raise Exception("could not find variables in bounded type", str(typ))
            return lower_vars + upper_vars + dim_vars
        case sl.Complex():
            return []
        case sl.ComplexVector(n):
            return find_variables_expr(n)
        case sl.ComplexRowVector(n):
            return find_variables_expr(n)
        case sl.ComplexMatrix(n, m):
            return find_variables_expr(n) + find_variables_expr(m)
        case sl.Simplex(n):
            return find_variables_expr(n)
        case sl.CholeskyFactorCorr(n):
            return find_variables_expr(n)
        case sl.Array(base, shape):
            shape_vars = [v for n in shape for v in find_variables_expr(n)]
            base_vars = find_variables_type(base)
            return base_vars + shape_vars
        case sl.Tuple(types):
            tup_vars = [v for tp in types for v in find_variables_type(tp)]
            return tup_vars
        case sl.Function(return_type, argument_types):
            arg_vars = [v for tp in argument_types for v in find_variables_type(tp)]
            ret_vars = find_variables_type(return_type)
            return arg_vars + ret_vars
        case _:
            raise Exception("could not find variables in type", str(typ))


def find_variables_stmt(stmt: sl.Stmt) -> list[sl.Var]:
    """
    Find variables in Stan statements
    """
    match stmt:
        case sl.EmptyStmt():
            return []
        case sl.ExpressionStmt(expression):
            return find_variables_expr(expression)
        case sl.MixinStmt(code): # TODO can't find variables in a MixinExpr?
            return [] 
        case sl.Decl(var):
            return find_variables_expr(var)
        case sl.DeclList(var_type, names):
            return find_variables_type(var_type)
        case sl.DeclAssign(var, rhs):
            return find_variables_expr(var) + find_variables_expr(rhs)
        case sl.Assign(lhs, rhs):
            find_variables_expr(lhs) + find_variables_expr(rhs)
        case sl.AddAssign(lhs, rhs):
            find_variables_expr(lhs) + find_variables_expr(rhs)
        case sl.Sample(lhs, rhs):
            find_variables_expr(lhs) + find_variables_expr(rhs)
        case sl.Return(ret):
            find_variables_expr(ret)
        case sl.Break():
            return []
        case sl.Scope(content, indentation):
            return [v for s in content for v in find_variables_stmt(s)]
        case sl.FuncDef(return_type, name, arguments, func_body):
            return_type_vars = find_variables_type(return_type)
            argument_vars = [v for e in arguments for v in find_variables_expr(e)]
            # make use of Scope to turn list[Stmt] into a Stmt
            func_body_vars = find_variables_stmt(sl.Scope(func_body))
            return return_type_vars + argument_vars + func_body_vars
        case sl.ForLoop(index, sequence, content):
            index_vars = find_variables_expr(index)
            seq_vars = find_variables_expr(sequence)
            content_vars = find_variables_stmt(content)
            return index_vars + seq_vars + content_vars
        case sl.IfStatement(condition, content):
            cond_vars = find_variables_expr(condition)
            content_vars = find_variables_stmt(content)
            return cond_vars + content_vars
        case sl.IfElseStatement(condition, content_if_true, content_if_false):
            cond_vars = find_variables_expr(condition)
            content_true_vars = find_variables_stmt(content_if_true)
            content_false_vars = find_variables_stmt(content_if_false)
            return cond_vars + content_true_vars + content_false_vars
        case sl.StanModel(funcs, data, tdata, params, tparams, model, gq):
            blocks = [funcs, data, tdata, params, tparams, model, gq]
            # make use of Scope to turn list[Stmt] into a Stmt
            block_scopes = [sl.Scope(bc) for bc in blocks if bc is not None]
            return [v for bs in block_scopes for v in find_variables_stmt(bs)]
        case _:
            raise Exception("could not find variables in statement " + str(stmt))
