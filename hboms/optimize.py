import operator
from typing import Literal

from . import stanlang as sl


def optimize_stmt(stmt: sl.Stmt) -> sl.Stmt:
    """
    Simplify Stan statements, and the expressions within those statements
    """
    match stmt:
        case sl.EmptyStmt():
            return sl.EmptyStmt(comment=stmt.comment)
        case sl.ExpressionStmt(expression):
            return sl.ExpressionStmt(optimize_expr(expression), comment=stmt.comment)
        case sl.MixinStmt(code):
            return sl.MixinStmt(code, comment=stmt.comment)
        case sl.Decl(var):
            opt_var = optimize_expr(var)  # FIXME: how to make sure this passes mypy?
            return sl.Decl(opt_var, comment=stmt.comment)
        case sl.DeclList(var_type, names):
            return sl.DeclList(optimize_type(var_type), names)
        case sl.DeclAssign(var, rhs):
            return sl.DeclAssign(
                optimize_expr(var), optimize_expr(rhs), comment=stmt.comment
            )
        case sl.Assign(lhs, rhs):
            return sl.Assign(
                optimize_expr(lhs), optimize_expr(rhs), comment=stmt.comment
            )
        case sl.AddAssign(lhs, rhs):
            return sl.AddAssign(
                optimize_expr(lhs), optimize_expr(rhs), comment=stmt.comment
            )
        case sl.Sample(lhs, rhs):
            return sl.Sample(
                optimize_expr(lhs), optimize_expr(rhs), comment=stmt.comment
            )
        case sl.Return(val):
            return sl.Return(optimize_expr(val), comment=stmt.comment)
        case sl.Break():
            return sl.Break(comment=stmt.comment)
        case sl.Scope(content, indentation):
            opt_content = optimize_stmt_list(content)
            return sl.Scope(opt_content, indentation)
        case sl.FuncDef(return_type, name, arguments, func_body):
            opt_args = optimize_expr_list(arguments)
            opt_body = optimize_stmt_list(func_body)
            return sl.FuncDef(
                optimize_type(return_type),
                name,
                opt_args,
                opt_body,
                comment=stmt.comment,
            )
        case sl.ForLoop(index, sequence, content):
            return sl.ForLoop(
                optimize_expr(index),
                optimize_expr(sequence),
                optimize_stmt(content),
                comment=stmt.comment,
            )
        case sl.IfStatement(condition, content):
            return sl.IfStatement(
                optimize_expr(condition), optimize_stmt(content), comment=stmt.comment
            )
        case sl.IfElseStatement(condition, content_if_true, content_if_false):
            return sl.IfElseStatement(
                optimize_expr(condition),
                optimize_stmt(content_if_true),
                optimize_stmt(content_if_false),
                comment=stmt.comment,
            )
        case sl.StanModel(
            functions,
            data,
            transformed_data,
            parameters,
            transformed_parameters,
            model,
            generated_quantities,
        ):
            return sl.StanModel(
                optimize_optional_stmt_list(functions),
                optimize_stmt_list(data),
                optimize_optional_stmt_list(transformed_data),
                optimize_stmt_list(parameters),
                optimize_optional_stmt_list(transformed_parameters),
                optimize_stmt_list(model),
                optimize_optional_stmt_list(generated_quantities),
                comment=stmt.comment,
            )
        case _:
            raise Exception("could not optimize stmt " + str(stmt))


def optimize_stmt_list(statements: list[sl.Stmt]) -> list[sl.Stmt]:
    return [optimize_stmt(stmt) for stmt in statements]


def optimize_optional_stmt_list(
    statements: list[sl.Stmt] | None,
) -> list[sl.Stmt] | None:
    if statements is None:
        return None
    # else...
    return optimize_stmt_list(statements)


def optimize_type(typ: sl.Type) -> sl.Type:
    """
    Simplify Stan types, and the expressions in the bounds and dimensions
    """
    match typ:
        case sl.BoundedType(lower=lower, upper=upper):
            # optmize the optional bounds
            if lower is not None:
                lower = optimize_expr(lower)
            if upper is not None:
                upper = optimize_expr(upper)
            # match typ again for all bounded types
            match typ:
                case sl.Real():
                    return sl.Real(lower=lower, upper=upper, data=typ.data)
                case sl.Int():
                    return sl.Int(lower=lower, upper=upper, data=typ.data)
                case sl.Vector(n):
                    return sl.Vector(
                        optimize_expr(n), lower=lower, upper=upper, data=typ.data
                    )
                case sl.RowVector(n):
                    return sl.RowVector(
                        optimize_expr(n), lower=lower, upper=upper, data=typ.data
                    )
                case sl.Matrix(n, m):
                    return sl.Matrix(
                        optimize_expr(n),
                        optimize_expr(m),
                        lower=lower,
                        upper=upper,
                        data=typ.data,
                    )
                case _:
                    raise Exception("could not optimize bounded type", str(typ))
        case sl.Complex():
            return sl.Complex(data=typ.data)
        case sl.ComplexVector(n):
            return sl.ComplexVector(optimize_expr(n), data=typ.data)
        case sl.ComplexRowVector(n):
            return sl.ComplexRowVector(optimize_expr(n), data=typ.data)
        case sl.ComplexMatrix(n, m):
            return sl.ComplexMatrix(optimize_expr(n), optimize_expr(m), data=typ.data)
        case sl.Simplex(n):
            return sl.Simplex(optimize_expr(n), data=typ.data)
        case sl.CholeskyFactorCorr(n):
            return sl.CholeskyFactorCorr(optimize_expr(n), data=typ.data)
        case sl.Array(base, shape):
            opt_shape = tuple(optimize_expr(n) for n in shape)
            return sl.Array(optimize_type(base), opt_shape, data=typ.data)
        case sl.Tuple(types):
            opt_types = tuple(optimize_type(tp) for tp in types)
            return sl.Tuple(opt_types)
        case sl.Function(return_type, argument_types):
            opt_argument_types = tuple(optimize_type(t) for t in argument_types)
            return sl.Function(
                optimize_type(return_type), opt_argument_types, data=typ.data
            )
        case sl.UnresolvedType():
            return sl.UnresolvedType()
        case _:
            raise Exception("could not optimize type " + str(typ))


def optimize_expr(expr: sl.Expr) -> sl.Expr:
    """
    Function to simplify some aspects of the expresion tree.
    For instance arithmatic with literal ints and reals

    TODO: - optimize cases like expr + 0, expr * 1, expr * 0, etc.
          - optimize power expressions like 2^3
          - remove unnecessary parentheses using operator precedence
          - optimize literal comparison, equalities
    """
    match expr:
        case sl.LiteralInt(val):
            return sl.LiteralInt(val)
        case sl.LiteralReal(val):
            return sl.LiteralReal(val)
        case sl.LiteralComplex(val):
            return sl.LiteralComplex(val)
        case sl.LiteralString(val):
            return sl.LiteralString(val)
        case sl.LiteralRowVector(elts):
            return sl.LiteralRowVector([optimize_expr(elt) for elt in elts])
        case sl.LiteralVector(elts):
            return sl.LiteralVector([optimize_expr(elt) for elt in elts])
        case sl.LiteralArray(elts):
            return sl.LiteralArray([optimize_expr(elt) for elt in elts])
        case sl.Var(var_type, name):
            return sl.Var(optimize_type(var_type), name)
        case sl.Range(start, stop):
            opt_start = optimize_expr(start) if start is not None else None
            opt_stop = optimize_expr(stop) if stop is not None else None
            return sl.Range(opt_start, opt_stop)
        case sl.TransposeOp(val):
            return optimize_transpose_op(val)
        case sl.Negate(val):
            return optimize_negate(val)
        case sl.MulOp(left, right):
            return optimize_bin_op("mul", left, right)
        case sl.AddOp(left, right):
            return optimize_bin_op("add", left, right)
        case sl.SubOp(left, right):
            return optimize_bin_op("sub", left, right)
        case sl.EltMulOp(left, right):
            return sl.EltMulOp(optimize_expr(left), optimize_expr(right))
        case sl.DivOp(left, right):
            return optimize_div_op(left, right)
        case sl.EltDivOp(left, right):
            return sl.EltDivOp(optimize_expr(left), optimize_expr(right))
        case sl.ModuloOp(dividend, divisor):
            return optimize_ibin_op("mod", dividend, divisor)
        case sl.IDivOp(left, right):
            return optimize_ibin_op("idiv", left, right)
        case sl.PowOp(base, exponent):
            return optimize_pow_op(base, exponent)
        case sl.EltPowOp(base, exponent):
            return sl.EltPowOp(optimize_expr(base), optimize_expr(exponent))
        case sl.MultiIndexOp(var, indices):
            return optimize_multi_index_op(var, indices)
        case sl.IndexOp(var, index):
            return optimize_multi_index_op(var, [index])
        case sl.TupleIndexOp(var, index):
            return sl.TupleIndexOp(optimize_expr(var), optimize_expr(index))
        case sl.EqOp(left, right):
            return sl.EqOp(optimize_expr(left), optimize_expr(right))
        case sl.NeqOp(left, right):
            return sl.NeqOp(optimize_expr(left), optimize_expr(right))
        case sl.LeOp(left, right):
            return sl.LeOp(optimize_expr(left), optimize_expr(right))
        case sl.GrOp(left, right):
            return sl.GrOp(optimize_expr(left), optimize_expr(right))
        case sl.LeEqOp(left, right):
            return sl.LeEqOp(optimize_expr(left), optimize_expr(right))
        case sl.GrEqOp(left, right):
            return sl.GrEqOp(optimize_expr(left), optimize_expr(right))
        case sl.TernaryOp(cond, expr_if_true, expr_if_false):
            return sl.TernaryOp(
                optimize_expr(cond),
                optimize_expr(expr_if_true),
                optimize_expr(expr_if_false),
            )
        case sl.LogicOrOp(left, right):
            return sl.LogicOrOp(optimize_expr(left), optimize_expr(right))
        case sl.LogicAndOp(left, right):
            return sl.LogicAndOp(optimize_expr(left), optimize_expr(right))
        case sl.Par(content):
            opt_content = optimize_expr(content)
            return opt_content if is_atomic_expr(opt_content) else sl.Par(opt_content)
        case sl.Call(func_name, arguments):
            opt_arguments = optimize_expr_list(arguments)
            return sl.Call(func_name, opt_arguments)
        case sl.PCall(dist_name, obs, parameters, suffix):
            opt_params = optimize_expr_list(parameters)
            return sl.PCall(dist_name, optimize_expr(obs), opt_params, suffix)
        case sl.MixinExpr(code):
            return sl.MixinExpr(code)
        case _:
            raise Exception("expresion can not be optimized " + str(expr))


def optimize_expr_list(expressions: list[sl.Expr]) -> list[sl.Expr]:
    return [optimize_expr(expr) for expr in expressions]


def optimize_bin_op(
    op: Literal["mul", "add", "sub"], left: sl.Expr, right: sl.Expr
) -> sl.Expr:
    pyop_dispatcher = {"mul": operator.mul, "add": operator.add, "sub": operator.sub}
    stanop_dispatcher = {"mul": sl.MulOp, "add": sl.AddOp, "sub": sl.SubOp}
    match (optimize_expr(left), optimize_expr(right)):
        case (sl.LiteralInt(val=a), sl.LiteralInt(val=b)):
            return sl.LiteralInt(pyop_dispatcher[op](a, b))
        case (
            sl.LiteralInt(val=a) | sl.LiteralReal(val=a),
            sl.LiteralInt(val=b) | sl.LiteralReal(val=b),
        ):
            return sl.LiteralReal(pyop_dispatcher[op](a, b))
        case (
            sl.LiteralInt(val=a) | sl.LiteralReal(val=a) | sl.LiteralComplex(val=a),
            sl.LiteralInt(val=b) | sl.LiteralReal(val=b) | sl.LiteralComplex(val=b),
        ):
            return sl.LiteralComplex(pyop_dispatcher[op](a, b))
        case (other_left, other_right):
            return stanop_dispatcher[op](other_left, other_right)
        case _:
            raise Exception("invalid case")


def optimize_div_op(left: sl.Expr, right: sl.Expr) -> sl.Expr:
    match (optimize_expr(left), optimize_expr(right)):
        case (
            sl.LiteralInt(val=a) | sl.LiteralReal(val=a),
            sl.LiteralInt(val=b) | sl.LiteralReal(val=b),
        ):
            return sl.LiteralReal(a / b)
        case (
            sl.LiteralInt(val=a) | sl.LiteralReal(val=a) | sl.LiteralComplex(a),
            sl.LiteralInt(val=b) | sl.LiteralReal(val=b) | sl.LiteralComplex(b),
        ):
            return sl.LiteralComplex(a / b)
        case (other_left, other_right):
            return sl.DivOp(other_left, other_right)
        case _:
            raise Exception("invalid case")


def optimize_ibin_op(
    op: Literal["idiv", "mod"], left: sl.Expr, right: sl.Expr
) -> sl.Expr:
    pyop_dispatcher = {"idiv": operator.floordiv, "mod": operator.mod}
    stanop_dispatcher = {"idiv": sl.IDivOp, "mod": sl.ModuloOp}
    match (optimize_expr(left), optimize_expr(right)):
        case (sl.LiteralInt(val=a), sl.LiteralInt(val=b)):
            return sl.LiteralInt(pyop_dispatcher[op](a, b))
        case (other_left, other_right):
            return stanop_dispatcher[op](other_left, other_right)
        case _:
            raise Exception("invalid case")


def optimize_pow_op(base: sl.Expr, exponent: sl.Expr) -> sl.Expr:
    # TODO: optimize literal values!
    match (optimize_expr(base), optimize_expr(exponent)):
        case (other_base, other_exponent):
            return sl.PowOp(other_base, other_exponent)
        case _:
            raise Exception("invalid case")


# FIXME: this is not always correct: does not work for multiple range indicess
def optimize_multi_index_op(var: sl.Expr, indices: list[sl.Expr]) -> sl.Expr:
    match optimize_expr(var):
        case sl.MultiIndexOp(base_var, base_indices):
            fin_idxs = base_indices + indices
            fin_var = base_var
        case sl.IndexOp(base_var, base_index):
            fin_idxs = [base_index] + indices
            fin_var = base_var
        case other:
            fin_idxs = indices
            fin_var = other
    # optimize indices
    fin_idxs = [optimize_expr(idx) for idx in fin_idxs]
    if len(fin_idxs) == 0:
        return fin_var
    elif len(fin_idxs) == 1:
        return sl.IndexOp(fin_var, fin_idxs[0])
    else:
        return sl.MultiIndexOp(fin_var, fin_idxs)


def optimize_transpose_op(val: sl.Expr) -> sl.Expr:
    """
    transpose is it's own inverse: x'' = x
    """
    match optimize_expr(val):
        case sl.TransposeOp(ival):
            return ival
        case other:
            return sl.TransposeOp(other)


def optimize_negate(val: sl.Expr) -> sl.Expr:
    """
    negate is its own inverse
    """
    match optimize_expr(val):
        case sl.Negate(ival):
            return ival
        case other:
            return sl.Negate(other)


def is_atomic_expr(expr: sl.Expr) -> bool:
    """
    test if an expression is "atomic".
    Atomic expressions are literals (but not complex literals), variables,
    and grouped expressions.
    """
    match expr:
        case sl.LiteralInt(val):
            return True
        case sl.LiteralReal(val):
            return True
        case sl.LiteralComplex(val):
            return False
        case sl.LiteralString(val):
            return True
        case sl.LiteralRowVector(elts):
            return True
        case sl.LiteralVector(elts):
            return True
        case sl.LiteralArray(elts):
            return True
        case sl.Var(var_type, name):
            return True
        case sl.Range(start, stop):
            return False
        case sl.TransposeOp(val):
            return False
        case sl.Negate(val):
            return False
        case sl.MulOp(left, right):
            return False
        case sl.EltMulOp(left, right):
            return False
        case sl.AddOp(left, right):
            return False
        case sl.SubOp(left, right):
            return False
        case sl.DivOp(left, right):
            return False
        case sl.EltDivOp(left, right):
            return False
        case sl.ModuloOp(dividend, divisor):
            return False
        case sl.IDivOp(left, right):
            return False
        case sl.PowOp(left, right):
            return False
        case sl.EltPowOp(left, right):
            return False
        case sl.MultiIndexOp(var, indices):
            return False
        case sl.IndexOp(var, index):
            return False
        case sl.EqOp(left, right):
            return False
        case sl.NeqOp(left, right):
            return False
        case sl.LeOp(left, right):
            return False
        case sl.GrOp(left, right):
            return False
        case sl.LeEqOp(left, right):
            return False
        case sl.GrEqOp(left, right):
            return False
        case sl.TernaryOp(cond, expr_if_true, expr_if_false):
            return False
        case sl.LogicOrOp(left, right):
            return False
        case sl.LogicAndOp(left, right):
            return False
        case sl.Par(content):
            return True
        case sl.Call(func_name, arguments):
            return False
        case sl.PCall(dist_name, obs, parameters, suffix):
            return False
        case sl.MixinExpr(code):
            return False
        case _:
            raise Exception("invalid expression type " + str(expr))
