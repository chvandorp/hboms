import ply.yacc as yacc
from . import stanlang as sl
from .stanlexer import tokens, lexer


# statement


def p_declaration_list(p):
    """
    declaration_list : declaration
                     | declaration declaration_list
    """
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]
    else:
        raise SyntaxError("error in declaration list")


# TODO: improve handling of long an line comments.


def p_declaration_longcomment(p):
    """declaration : LONGCOMMENT"""
    p[0] = sl.comment(p[1][2:-2].strip())


def p_declaration_comment(p):
    """declaration : declaration COMMENT"""
    p[0] = p[1]
    p[1].comment.append(p[2][2:].strip())


def p_declaration_conditional(p):
    """
    declaration : statement
                | unmatched
    """
    p[0] = p[1]


def p_declaration_var(p):
    """
    declaration : type ID SEMICOLON
    """
    p[0] = sl.Decl(sl.Var(p[1], p[2]))


def p_id_list(p):
    """
    id_list : ID COMMA ID
            | ID COMMA id_list
    """
    match p[3]:
        case str():
            p[0] = [p[1], p[3]]
        case list():
            p[0] = [p[1]] + p[3]
        case _:
            raise SyntaxError("invalid type of ID or ID list")


def p_declaration_decl_list(p):
    """
    declaration : type id_list SEMICOLON
    """
    p[0] = sl.DeclList(p[1], p[2])


def p_declaration_assign(p):
    """
    declaration : type ID ASSIGN expression SEMICOLON
    """
    p[0] = sl.DeclAssign(sl.Var(p[1], p[2]), p[4])


def p_statement_expr(p):
    """
    statement : expression SEMICOLON
    """
    p[0] = sl.ExpressionStmt(p[1])


def p_statement_assign(p):
    """
    statement : expression ASSIGN expression SEMICOLON
    """
    p[0] = sl.Assign(p[1], p[3])


def p_statement_add_assign(p):
    """
    statement : expression ADDASSIGN expression SEMICOLON
    """
    p[0] = sl.AddAssign(p[1], p[3])


def p_statement_sample(p):
    """
    statement : expression TILDE expression SEMICOLON
    """
    # TODO: make sure p[3] is a sl.Call object
    p[0] = sl.Sample(p[1], p[3])


def p_statement_block(p):
    """
    statement : LBRACE declaration_list RBRACE
    """
    p[0] = sl.Scope(p[2])


def p_statement_if(p):
    """
    unmatched : IF LPAREN expression RPAREN statement
              | IF LPAREN expression RPAREN unmatched
    """
    p[0] = sl.IfStatement(p[3], p[5])


def p_statement_if_else(p):
    """
    statement : IF LPAREN expression RPAREN statement ELSE statement
    unmatched : IF LPAREN expression RPAREN statement ELSE unmatched
    """
    p[0] = sl.IfElseStatement(p[3], p[5], p[7])


def p_statement_for(p):
    """
    statement : FOR LPAREN ID IN range RPAREN statement
    unmatched : FOR LPAREN ID IN range RPAREN unmatched
    """
    index_var = sl.Var(sl.UnresolvedType(), p[3])
    p[0] = sl.ForLoop(index_var, p[5], p[7])


def p_statement_return(p):
    """
    statement : RETURN expression SEMICOLON
    """
    p[0] = sl.Return(p[2])


def p_statement_break(p):
    """
    statement : BREAK SEMICOLON
    """
    p[0] = sl.Break()


# range


def p_range(p):
    "range : expression COLON expression"
    p[0] = sl.Range(p[1], p[3])


def p_range_expression(p):
    "range : expression"
    p[0] = p[1]


# expressions

"""
We currently have the following hierarchy of expressions:
    expression
    disjunction
    conjunction
    equality
    comparison
    arithmatic
    term
    factor
    unary
    base
    right_unary
    primary
TODO: make this fully compatible with Stan
"""

# conditional operator a ? b : c


def p_expression_ternary(p):
    "expression : disjunction QUESTIONMARK expression COLON expression"
    p[0] = sl.TernaryOp(p[1], p[3], p[5])


# logical operators


def p_expression_disjunction(p):
    "expression : disjunction"
    p[0] = p[1]


def p_disjunction_or(p):
    "disjunction : disjunction OR conjunction"
    p[0] = sl.LogicOrOp(p[1], p[3])


def p_disjunction_conjunction(p):
    "disjunction : conjunction"
    p[0] = p[1]


def p_conjunction_and(p):
    "conjunction : conjunction AND equality"
    p[0] = sl.LogicAndOp(p[1], p[3])


def p_conjunction_equality(p):
    "conjunction : equality"
    p[0] = p[1]


def p_equality_equals(p):
    "equality : equality EQUALS comparison"
    p[0] = sl.EqOp(p[1], p[3])


def p_equality_notequals(p):
    "equality : equality NOTEQUALS comparison"
    p[0] = sl.EqOp(p[1], p[3])


def p_equality_comparison(p):
    "equality : comparison"
    p[0] = p[1]


def p_comparison_less(p):
    "comparison : comparison LESS arithmatic"
    p[0] = sl.LeOp(p[1], p[3])


def p_comparison_lessequals(p):
    "comparison : comparison LESSEQUALS arithmatic"
    p[0] = sl.LeEqOp(p[1], p[3])


def p_comparison_greater(p):
    "comparison : comparison GREATER arithmatic"
    p[0] = sl.GrOp(p[1], p[3])


def p_comparison_greaterequals(p):
    "comparison : comparison GREATEREQUALS arithmatic"
    p[0] = sl.GrEqOp(p[1], p[3])


def p_comparison_arithmatic(p):
    "comparison : arithmatic"
    p[0] = p[1]


# arithmatic operators


def p_arithmatic_plus(p):
    "arithmatic : arithmatic PLUS term"
    p[0] = sl.AddOp(p[1], p[3])


def p_arithmatic_minus(p):
    "arithmatic : arithmatic MINUS term"
    p[0] = sl.SubOp(p[1], p[3])


# term


def p_arithmatic_term(p):
    "arithmatic : term"
    p[0] = p[1]


def p_term_times(p):
    "term : term TIMES factor"
    p[0] = sl.MulOp(p[1], p[3])


def p_term_dottimes(p):
    "term : term DOTTIMES factor"
    p[0] = sl.EltMulOp(p[1], p[3])


def p_term_div(p):
    "term : term DIVIDE factor"
    p[0] = sl.DivOp(p[1], p[3])


def p_term_dotdiv(p):
    "term : term DOTDIVIDE factor"
    p[0] = sl.EltDivOp(p[1], p[3])


def p_term_modulo(p):
    "term : term MODULUS factor"
    p[0] = sl.ModuloOp(p[1], p[3])


# FIXME: idiv has a higher prescedence than div


def p_term_idiv(p):
    "term : term IDIVIDE factor"
    p[0] = sl.IDivOp(p[1], p[3])


# factor


def p_term_factor(p):
    "term : factor"
    p[0] = p[1]


def p_factor_unary(p):
    "factor : unary"
    p[0] = p[1]


# unary


def p_unary_exponent(p):
    "unary : base"
    p[0] = p[1]


def p_unary_negate(p):
    "unary : MINUS unary"
    p[0] = sl.Negate(p[2])


# exponent and base


def p_unary_power(p):
    "unary : base POWER unary"
    p[0] = sl.PowOp(p[1], p[3])


def p_unary_dotpower(p):
    "unary : base DOTPOWER unary"
    p[0] = sl.EltPowOp(p[1], p[3])


# right-unary


def p_base_rightunary(p):
    """
    base : rightunary
    """
    p[0] = p[1]


def p_rightunary_transpose(p):
    "rightunary : rightunary TRANSPOSE"
    p[0] = sl.TransposeOp(p[1])


def p_rightunary_indexing(p):
    "rightunary : rightunary LBRACKET range RBRACKET"
    p[0] = sl.IndexOp(p[1], p[3])


def p_range_list(p):
    """
    range_list : range COMMA range
               | range COMMA range_list
    """
    match p[3]:
        case sl.Expr():
            p[0] = [p[1], p[3]]
        case list():
            p[0] = [p[1]] + p[3]
        case _:
            raise SyntaxError("invalid type of range or range list")


def p_rightunary_multiindexing(p):
    "rightunary : rightunary LBRACKET range_list RBRACKET"
    p[0] = sl.MultiIndexOp(p[1], p[3])


def p_rightunary_primary(p):
    "rightunary : primary"
    p[0] = p[1]


# primary


def p_primary_par(p):
    "primary : LPAREN expression RPAREN"
    p[0] = sl.Par(p[2])


def p_primary_real(p):
    "primary : REAL"
    p[0] = sl.LiteralReal(p[1])


def p_primary_imag(p):
    "primary : IMAG"
    p[0] = sl.LiteralComplex(complex(0.0, p[1])) # real part is always zero


def p_primary_int(p):
    "primary : INT"
    p[0] = sl.LiteralInt(p[1])


def p_primary_string(p):
    "primary : STRING"
    p[0] = sl.LiteralString(p[1])


def p_primary_id(p):
    "primary : ID"
    p[0] = sl.Var(sl.UnresolvedType(), p[1])


def p_primary_call(p):
    """
    primary : ID LPAREN RPAREN
            | ID LPAREN expression RPAREN
            | ID LPAREN expression_list RPAREN
    """
    if len(p) == 5:
        match p[3]:
            case sl.Expr():
                p[0] = sl.Call(p[1], [p[3]])
            case list():
                p[0] = sl.Call(p[1], p[3])
            case _:
                raise SyntaxError("invalid type of argument list in call")
    elif len(p) == 4:
        p[0] = sl.Call(p[1], [])


def p_primary_pcall(p):
    """
    primary : ID LPAREN expression PIPE expression RPAREN
            | ID LPAREN expression PIPE expression_list RPAREN
    """
    name_parts = p[1].split("_")
    if len(name_parts) < 2:
        raise SyntaxError("invalid name format in pcall")
    distname, suffix = "_".join(name_parts[:-1]), name_parts[-1]
    # TODO test that suffix is a valid suffix (lpdf, lcdf, ...)
    match p[5]:
        case sl.Expr():
            p[0] = sl.PCall(distname, p[3], [p[5]], suffix)
        case list():
            p[0] = sl.PCall(distname, p[3], p[5], suffix)
        case _:
            raise SyntaxError("invalid type of argument list in call")


def p_primary_literalrowvector(p):
    """
    primary : LBRACKET expression_list RBRACKET
            | LBRACKET expression RBRACKET
            | LBRACKET RBRACKET
    """
    if len(p) == 4:
        match p[2]:
            case sl.Expr():
                p[0] = sl.LiteralRowVector([p[2]])
            case list():
                p[0] = sl.LiteralRowVector(p[2])
    elif len(p) == 3:
        p[0] = sl.LiteralRowVector([])


def p_primary_literalarray(p):
    """
    primary : LBRACE expression_list RBRACE
            | LBRACE expression RBRACE
            | LBRACE RBRACE
    """
    if len(p) == 4:
        match p[2]:
            case sl.Expr():
                p[0] = sl.LiteralArray([p[2]])
            case list():
                p[0] = sl.LiteralArray(p[2])
    elif len(p) == 3:
        p[0] = sl.LiteralArray([])


def p_expression_list(p):
    """
    expression_list : expression COMMA expression
                    | expression COMMA expression_list
    """
    match p[3]:
        case sl.Expr():
            p[0] = [p[1], p[3]]
        case list():
            p[0] = [p[1]] + p[3]
        case _:
            raise SyntaxError("invalid type of expression or expression list")


def p_lbound(p):
    "lbound : LOWER ASSIGN arithmatic"
    # lbound expressions can not contain logical operators.
    p[0] = ("lower", p[3])


def p_ubound(p):
    "ubound : UPPER ASSIGN arithmatic"
    # ubound expressions can not contain logical operators.
    p[0] = ("upper", p[3])


def p_chevrons(p):
    """
    chevrons : LESS lbound GREATER
             | LESS ubound GREATER
             | LESS lbound COMMA ubound GREATER
    """
    if len(p) == 4:
        bound_list = [p[2]]
    elif len(p) == 6:
        bound_list = [p[2], p[4]]
    else:
        raise SyntaxError("unable to parse chevrons")
    p[0] = dict(bound_list)


def p_type_prim_scalar(p):
    """
    type_prim_scalar : TYPE_REAL
                     | TYPE_INT
    """
    p[0] = p[1]


def p_type_scalar(p):
    """
    type_scalar : type_prim_scalar
                | type_prim_scalar chevrons
    """
    if len(p) == 2:
        chevrons = {}
    elif len(p) == 3:
        chevrons = p[2]

    type_dispatch = {"real": sl.Real, "int": sl.Int}

    p[0] = type_dispatch[p[1]](**chevrons)


def p_type_prim_vector(p):
    """
    type_prim_vector : TYPE_VECTOR
                     | TYPE_ROW_VECTOR
    """
    p[0] = p[1]


def p_type_vector(p):
    """
    type_vector : type_prim_vector LBRACKET expression RBRACKET
                | type_prim_vector chevrons LBRACKET expression RBRACKET
    """
    if len(p) == 5:
        chevrons = {}
        n = p[3]
    elif len(p) == 6:
        chevrons = p[2]
        n = p[4]

    type_dispatch = {"vector": sl.Vector, "row_vector": sl.RowVector}

    p[0] = type_dispatch[p[1]](n, **chevrons)


def p_type_matrix(p):
    """
    type_matrix : TYPE_MATRIX LBRACKET expression COMMA expression RBRACKET
                | TYPE_MATRIX chevrons LBRACKET expression COMMA expression RBRACKET
    """
    if len(p) == 7:
        chevrons = {}
        n, m = p[3], p[5]
    elif len(p) == 8:
        chevrons = p[2]
        n, m = p[4], p[6]
    else:
        raise SyntaxError("unable to parse matrix type")

    p[0] = sl.Matrix(n, m, **chevrons)


def p_type_array(p):
    """
    type_array : TYPE_ARRAY LBRACKET expression RBRACKET type
               | TYPE_ARRAY LBRACKET expression_list RBRACKET type
    """
    if isinstance(p[3], list):
        p[0] = sl.Array(p[5], tuple(p[3]))
    else:
        p[0] = sl.Array(p[5], (p[3],))



def p_type_list(p):
    """
    type_list : type COMMA type
              | type COMMA type_list
    """
    match p[3]:
        case sl.Type():
            p[0] = [p[1], p[3]]
        case list():
            p[0] = [p[1]] + p[3]
        case _:
            raise SyntaxError("invalid type of type or type list")



def p_type_tuple(p):
    """
    type_tuple : TYPE_TUPLE LPAREN type_list RPAREN
    """
    p[0] = sl.Tuple(tuple(p[3]))
    
    

def p_type(p):
    """
    type : type_scalar
         | type_vector
         | type_matrix
         | type_array
         | type_tuple
    """
    p[0] = p[1]


# Error rule for syntax errors
def p_error(p):
    if p is None:
        print("Syntax error at end of input")
    else:
        print(f"Syntax error at '{p.value}' on line {p.lineno}")


# Build the parser
parser = yacc.yacc()
