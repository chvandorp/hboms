from . import utilities as util
from . import stanlang as sl

from typing import Optional


def deparse_func_arg(typ: sl.Type) -> str:
    modifier = "data " if typ.data else ""
    match typ:
        case sl.Real():
            return modifier + "real"
        case sl.Complex():
            return modifier + "complex"
        case sl.Int():
            return modifier + "int"
        case sl.RowVector(n):
            return modifier + "row_vector"
        case sl.Vector(n):
            return modifier + "vector"
        case sl.Matrix(n, m):
            return modifier + "matrix"
        case sl.ComplexRowVector(n):
            return modifier + "complex_row_vector"
        case sl.ComplexVector(n):
            return modifier + "complex_vector"
        case sl.ComplexMatrix(n, m):
            return modifier + "complex_matrix"
        case sl.Array(base_type, shape):
            return modifier + f"array[] {deparse_func_arg(base_type)}"
        case sl.Tuple(types):
            type_list = ", ".join([deparse_func_arg(tp) for tp in types])
            return modifier + f"tuple({type_list})"
        case sl.Simplex(n):
            return modifier + "simplex"
        case sl.CholeskyFactorCorr(n):
            return modifier + "matrix"
        case sl.Function(argument_types):
            raise Exception(
                "functions are not allowed to be arguments of user-defined functions"
            )
        case _:
            raise Exception("unable to parse type (func_arg)")


def deparse_chevrons(lower: Optional[sl.Expr], upper: Optional[sl.Expr]) -> str:
    """
    the chevrons function constructs a string for the declaration of a bounded variable.

    Example: `<lower=0, upper=1>`
    """
    bounds_strs = []
    if lower is not None:
        bounds_strs.append(f"lower={deparse_expr(lower)}")
    if upper is not None:
        bounds_strs.append(f"upper={deparse_expr(upper)}")
    if len(bounds_strs) > 0:
        return "<" + ", ".join(bounds_strs) + ">"
    else:
        return ""


def deparse_decl(typ: sl.Type) -> str:
    match typ:
        case sl.Real(lower=lower, upper=upper):
            return "real" + deparse_chevrons(lower, upper)
        case sl.Complex():
            return "complex"
        case sl.Int(lower=lower, upper=upper):
            return "int" + deparse_chevrons(lower, upper)
        case sl.RowVector(n, lower=lower, upper=upper):
            chstr = deparse_chevrons(lower, upper)
            return f"row_vector{chstr}[{deparse_expr(n)}]"
        case sl.Vector(n, lower=lower, upper=upper):
            chstr = deparse_chevrons(lower, upper)
            return f"vector{chstr}[{deparse_expr(n)}]"
        case sl.Matrix(n, m, lower=lower, upper=upper):
            chstr = deparse_chevrons(lower, upper)
            return f"matrix{chstr}[{deparse_expr(n)}, {deparse_expr(m)}]"
        case sl.ComplexRowVector(n):
            return f"complex_row_vector[{deparse_expr(n)}]"
        case sl.ComplexVector(n):
            return f"complex_vector[{deparse_expr(n)}]"
        case sl.ComplexMatrix(n, m):
            return f"complex_matrix[{deparse_expr(n)}, {deparse_expr(m)}]"
        case sl.Array(base_type, shape):
            shape_str = ", ".join([deparse_expr(x) for x in shape])
            return f"array[{shape_str}] {deparse_decl(base_type)}"
        case sl.Tuple(types):
            type_list = ", ".join([deparse_decl(tp) for tp in types])
            return f"tuple({type_list})"
        case sl.Simplex(n):
            return f"simplex[{deparse_expr(n)}]"
        case sl.CholeskyFactorCorr(n):
            return f"cholesky_factor_corr[{deparse_expr(n)}]"
        case sl.Function(return_type, argument_types):
            raise Exception(
                "functions are not allowed to be arguments of user-defined functions"
            )
        case _:
            raise Exception("unable to parse type (func_arg)")


def deparse_expr(expr: sl.Expr) -> str:
    """
    transform a Expr into Stan code
    """
    match expr:
        case sl.LiteralInt(val):
            code_str = str(val)

        case sl.LiteralReal(val):
            code_str = str(float(val))

        case sl.LiteralComplex(val):
            op = "+" if val.imag >= 0 else "-"
            code_str = (
                f"{val.real}{op}{abs(val.imag)}i"  # stan uses i instead of pythons j
            )

        case sl.LiteralRowVector(elts):
            code_str = "[" + ", ".join([deparse_expr(x) for x in elts]) + "]"

        case sl.LiteralVector(elts):
            code_str = "[" + ", ".join([deparse_expr(x) for x in elts]) + "]'"

        case sl.LiteralArray(elts):
            code_str = "{" + ", ".join([deparse_expr(x) for x in elts]) + "}"

        case sl.LiteralString(val):
            code_str = '"' + val + '"'

        case sl.Var(var_type, name):
            code_str = name

        case sl.Range(start, stop):
            dep_start = deparse_expr(start) if start is not None else ""
            dep_stop = deparse_expr(stop) if stop is not None else ""
            code_str = f"{dep_start}:{dep_stop}"

        case sl.TransposeOp(val):
            code_str = f"{deparse_expr(val)}'"

        case sl.IndexOp(var, index):
            code_str = f"{deparse_expr(var)}[{deparse_expr(index)}]"

        case sl.MultiIndexOp(var, indices):
            idx_list = ", ".join([deparse_expr(idx) for idx in indices])
            code_str = f"{deparse_expr(var)}[{idx_list}]"

        case sl.TupleIndexOp(var, index):
            code_str = f"{deparse_expr(var)}.{deparse_expr(index)}"

        case sl.LogicOrOp(left, right):
            code_str = f"{deparse_expr(left)} || {deparse_expr(right)}"

        case sl.LogicAndOp(left, right):
            code_str = f"{deparse_expr(left)} && {deparse_expr(right)}"

        case sl.Negate(val):
            code_str = f"-{deparse_expr(val)}"

        case sl.MulOp(left, right):
            code_str = f"{deparse_expr(left)} * {deparse_expr(right)}"

        case sl.EltMulOp(left, right):
            code_str = f"{deparse_expr(left)} .* {deparse_expr(right)}"

        case sl.AddOp(left, right):
            code_str = f"{deparse_expr(left)} + {deparse_expr(right)}"

        case sl.SubOp(left, right):
            code_str = f"{deparse_expr(left)} - {deparse_expr(right)}"

        case sl.DivOp(num, den):
            code_str = f"{deparse_expr(num)} / {deparse_expr(den)}"

        case sl.EltDivOp(num, den):
            code_str = f"{deparse_expr(num)} ./ {deparse_expr(den)}"

        case sl.ModuloOp(dividend, divisor):
            code_str = f"{deparse_expr(dividend)} % {deparse_expr(divisor)}"

        case sl.IDivOp(num, den):
            code_str = f"{deparse_expr(num)} %/% {deparse_expr(den)}"

        case sl.PowOp(base, exponent):
            code_str = f"{deparse_expr(base)}^{deparse_expr(exponent)}"

        case sl.EltPowOp(base, exponent):
            code_str = f"{deparse_expr(base)}.^{deparse_expr(exponent)}"

        case sl.EqOp(left, right):
            code_str = f"{deparse_expr(left)} == {deparse_expr(right)}"

        case sl.NeqOp(left, right):
            code_str = f"{deparse_expr(left)} != {deparse_expr(right)}"

        case sl.LeOp(left, right):
            code_str = f"{deparse_expr(left)} < {deparse_expr(right)}"

        case sl.GrOp(left, right):
            code_str = f"{deparse_expr(left)} > {deparse_expr(right)}"

        case sl.LeEqOp(left, right):
            code_str = f"{deparse_expr(left)} <= {deparse_expr(right)}"

        case sl.GrEqOp(left, right):
            code_str = f"{deparse_expr(left)} >= {deparse_expr(right)}"

        case sl.TernaryOp(cond, expr_if_true, expr_if_false):
            code_str = f"{deparse_expr(cond)} ? {deparse_expr(expr_if_true)} : {deparse_expr(expr_if_false)}"

        case sl.Par(content):
            code_str = f"({deparse_expr(content)})"

        case sl.Call(func_name, arguments):
            arg_list = ", ".join([deparse_expr(arg) for arg in arguments])
            code_str = f"{func_name}({arg_list})"

        case sl.PCall(dist_name, obs, params, suffix):
            ## FIXME: what if there are no parameters?? Add test
            par_list = ", ".join([deparse_expr(par) for par in params])
            func_name = dist_name + "_" + suffix
            code_str = f"{func_name}({deparse_expr(obs)} | {par_list})"

        case sl.MixinExpr(code):
            code_str = code.strip()

        case _:
            raise Exception("unable to deparse expression " + str(expr))

    # and return result
    return code_str


def deparse_stmt(stmt: sl.Stmt) -> str:
    """
    transform a Stmt into Stan code
    """
    match stmt:
        case sl.EmptyStmt():
            code_str = ""

        case sl.ExpressionStmt(expression):
            code_str = deparse_expr(expression) + ";"

        case sl.MixinStmt(code):
            code_str = code.strip()

        case sl.Decl(sl.Var(var_type, name)):
            tp_str = deparse_decl(var_type)
            code_str = f"{tp_str} {name};"

        case sl.DeclList(var_type, names):
            tp_str = deparse_decl(var_type)
            code_str = f"{tp_str} " + ", ".join(names) + ";"

        case sl.DeclAssign(sl.Var(var_type, name), rhs):
            tp_str = deparse_decl(var_type)
            code_str = f"{tp_str} {name} = {deparse_expr(rhs)};"

        case sl.Assign(lhs, rhs):
            code_str = f"{deparse_expr(lhs)} = {deparse_expr(rhs)};"

        case sl.AddAssign(lhs, rhs):
            code_str = f"{deparse_expr(lhs)} += {deparse_expr(rhs)};"

        case sl.Sample(lhs, rhs):
            code_str = f"{deparse_expr(lhs)} ~ {deparse_expr(rhs)};"

        case sl.Return(ret):
            code_str = f"return {deparse_expr(ret)};"

        case sl.Break():
            code_str = "break;"

        case sl.Scope(content, indentation):
            content_str = "\n".join([deparse_stmt(stmt) for stmt in content])
            code_str = f"{{\n{util.indent(content_str, indentation)}\n}}"

        case sl.FuncDef(return_type, name, arguments, func_body):
            arg_list = ", ".join(
                [f"{deparse_func_arg(arg.var_type)} {arg.name}" for arg in arguments]
            )
            body = sl.Scope(func_body)
            func_str = f"{deparse_func_arg(return_type)} {name}({arg_list}) {deparse_stmt(body)}"
            code_str = func_str

        case sl.ForLoop(index, sequence, content):
            forloop_str = f"for ( {deparse_expr(index)} in {deparse_expr(sequence)} ) {deparse_stmt(content)}"
            code_str = forloop_str

        case sl.IfStatement(condition, content):
            code_str = f"if ( {deparse_expr(condition)} ) {deparse_stmt(content)}"

        case sl.IfElseStatement(condition, content_if_true, content_if_false):
            cond_str = deparse_expr(condition)
            cont_true = deparse_stmt(content_if_true)
            cont_false = deparse_stmt(content_if_false)
            # different formatting for scoped and unscoped content
            sep = " " if isinstance(content_if_true, sl.Scope) else "\n"
            code_str = f"if ( {cond_str} ) {cont_true}{sep}else {cont_false}"

        case sl.StanModel(funcs, data, tdata, params, tparams, model, gq):
            blocks = [funcs, data, tdata, params, tparams, model, gq]
            block_scopes = [sl.Scope(bc) for bc in blocks if bc is not None]
            block_names = [
                bn for bn, bc in zip(sl.model_block_names, blocks) if bc is not None
            ]
            block_strs = [
                f"{bn} {deparse_stmt(sc)}"
                for bn, sc in zip(block_names, block_scopes)
                if sc is not None
            ]
            code_str = "\n\n".join(block_strs)
            return code_str

        case _:
            raise Exception("unable to deparse statement " + str(stmt))

    # add optional comments to code_str
    spacing = " " if len(code_str) > 0 else ""
    for comment in stmt.comment:
        code_str += spacing + f"/* {comment} */"
        spacing = "\n"
    # and return result
    return code_str
