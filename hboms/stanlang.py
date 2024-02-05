"""
a model of the Stan programming language
"""
from dataclasses import dataclass, field
from typing import Optional, ClassVar, Sequence, Literal
import typing  # don't import Tuple, as we need this name
from abc import ABC, abstractmethod
from functools import reduce


@dataclass
class Expr(ABC):
    """abstract base class for Stan expressions"""

    def __mul__(self, other):
        return MulOp(self, other)

    def __add__(self, other):
        return AddOp(self, other)

    def __sub__(self, other):
        return SubOp(self, other)

    def __truediv__(self, other):
        return DivOp(self, other)

    def __floordiv__(self, other):
        return IDivOp(self, other)

    def __xor__(self, other):
        return PowOp(self, other)

    def __mod__(self, other):
        return ModuloOp(self, other)

    def idx(self, other, *others):
        """
        Implementing __getitem__ is too confusing.
        """
        if len(others) == 0:
            return IndexOp(self, other)
        else:
            return MultiIndexOp(self, [other] + list(others))

    def tupidx(self, other):
        return TupleIndexOp(self, other)


@dataclass  # type: ignore[misc]
class Type(ABC):
    """
    abstract base class for a Stan type.
    data keyword is added to function definitions
    TODO: mypy bug
    """

    data: bool = field(kw_only=True, default=False)

    @abstractmethod
    def flat_dim(self) -> Expr:
        raise NotImplementedError("flat_dim is an abstract method")

    @abstractmethod
    def is_discrete(self) -> bool:
        raise NotImplementedError("is_discrete is an abstract method")


@dataclass  # type: ignore[misc]
class Stmt(ABC):
    comment: Optional[str | list[str]] = field(kw_only=True, default=None)

    def __post_init__(self):
        match self.comment:
            case None:
                self.comment = []
            case str():
                self.comment = [self.comment]
            case _:
                pass


@dataclass  # type: ignore[misc]
class BoundedType(Type):
    lower: Optional[Expr] = field(kw_only=True, default=None)
    upper: Optional[Expr] = field(kw_only=True, default=None)


"""
derived Stan types
"""


@dataclass
class UnresolvedType(Type):
    def flat_dim(self) -> Expr:
        raise Exception("trying to get flat dim of unresolved type")

    def is_discrete(self) -> bool:
        raise Exception("trying to determine if an unresolved type is discrete")


@dataclass
class Real(BoundedType):
    def flat_dim(self) -> Expr:
        return LiteralInt(1)

    def is_discrete(self) -> bool:
        return False


@dataclass
class Complex(Type):
    def flat_dim(self) -> Expr:
        return LiteralInt(2)  # dimension over the reals

    def is_discrete(self) -> bool:
        return False


@dataclass
class Int(BoundedType):
    def flat_dim(self) -> Expr:
        return LiteralInt(1)

    def is_discrete(self) -> bool:
        return True


@dataclass
class Vector(BoundedType):
    n: Expr

    def flat_dim(self) -> Expr:
        return self.n

    def is_discrete(self) -> bool:
        return False


@dataclass
class ComplexVector(Type):
    n: Expr

    def flat_dim(self) -> Expr:
        return LiteralInt(2) * Par(self.n)

    def is_discrete(self) -> bool:
        return False


@dataclass
class RowVector(BoundedType):
    n: Expr

    def flat_dim(self) -> Expr:
        return self.n

    def is_discrete(self) -> bool:
        return False


@dataclass
class ComplexRowVector(Type):
    n: Expr

    def flat_dim(self) -> Expr:
        return LiteralInt(2) * Par(self.n)

    def is_discrete(self) -> bool:
        return False


@dataclass
class Matrix(BoundedType):
    n: Expr
    m: Expr

    def flat_dim(self) -> Expr:
        return Par(self.n) * Par(self.m)

    def is_discrete(self) -> bool:
        return False


@dataclass
class ComplexMatrix(Type):
    n: Expr
    m: Expr

    def flat_dim(self) -> Expr:
        return LiteralInt(2) * Par(self.n) * Par(self.m)

    def is_discrete(self) -> bool:
        return False


@dataclass
class Simplex(Type):
    n: Expr

    def flat_dim(self) -> Expr:
        return self.n

    def is_discrete(self) -> bool:
        return False


@dataclass
class CholeskyFactorCorr(Type):
    n: Expr

    def flat_dim(self) -> Expr:
        d = IDivOp(MulOp(self.n, self.n + LiteralInt(1)), LiteralInt(2))
        return d

    def is_discrete(self) -> bool:
        return False


@dataclass
class Array(Type):
    base: Type
    shape: Expr | tuple[Expr, ...]

    def __post_init__(self) -> None:
        if isinstance(self.shape, Expr):
            self.shape = (self.shape,)

    def flat_dim(self) -> Expr:
        return reduce(lambda a, b: MulOp(a, b), (self.base.flat_dim(), *self.shape))

    def is_discrete(self) -> bool:
        return self.base.is_discrete()


@dataclass
class Tuple(Type):
    types: tuple[Type, ...]

    def flat_dim(self) -> Expr:
        return reduce(lambda a, b: AddOp(a, b), [tp.flat_dim() for tp in self.types])

    def is_discrete(self) -> bool:
        # FIXME: what is is_discrete used for? Is this a good definition for tuple?
        return all([tp.is_discrete() for tp in self.types])


@dataclass
class Function(Type):
    return_type: Type
    argument_types: tuple[Type, ...]

    def flat_dim(self) -> Expr:
        return self.return_type.flat_dim()

    def is_discrete(self) -> bool:
        return self.return_type.is_discrete()


"""
Stan expressions
"""


@dataclass
class LiteralInt(Expr):
    val: int
    literal_type: ClassVar[Type] = Int()


@dataclass
class LiteralReal(Expr):
    val: float
    literal_type: ClassVar[Type] = Real()


@dataclass
class LiteralComplex(Expr):
    val: complex
    literal_type: ClassVar[Type] = Complex()


@dataclass
class LiteralString(Expr):
    val: str


# FIXME: literal vectors could also be complex, and we can even construct matrices!


@dataclass
class LiteralRowVector(Expr):
    elts: Sequence[Expr]

    def __post_init__(self) -> None:
        self.literal_type = RowVector(LiteralInt(len(self.elts)))


@dataclass
class LiteralVector(Expr):
    elts: Sequence[Expr]

    def __post_init__(self) -> None:
        self.literal_type = Vector(LiteralInt(len(self.elts)))


@dataclass
class LiteralArray(Expr):
    elts: Sequence[Expr]

    def __post_init__(self) -> None:
        self.literal_type = Array(UnresolvedType(), (LiteralInt(len(self.elts)),))


@dataclass
class Var(Expr):
    """
    a Stan variable

    Example: `real x`
    """

    var_type: Type
    name: str


@dataclass
class Range(Expr):
    """
    a range defined with the : operator
    If start (stop) is None, then the bound is left empty,
    indicating the first (last) element in list

    Example: `1:10`, `:3`, `2:`, `:`
    """

    start: Optional[Expr]
    stop: Optional[Expr]


@dataclass
class TransposeOp(Expr):
    """
    transpose of a vector, row_vector or matrix

    Example: `A'`
    """

    val: Expr


@dataclass
class IndexOp(Expr):
    """
    indexing of a Stan variable

    Example: `x[1]`
    """

    var: Expr
    index: Expr


@dataclass
class MultiIndexOp(Expr):
    """
    indexing of a Stan variable

    Example: `x[1, 2:3]`
    """

    var: Expr
    indices: list[Expr]


@dataclass
class TupleIndexOp(Expr):
    """
    access to element of a tuple

    Example: `x.1`, `tup.2`, `val.3`
    """

    var: Expr
    index: LiteralInt


@dataclass
class Negate(Expr):
    """
    Unary minus operator

    Example: `-x`
    """

    val: Expr


@dataclass
class MulOp(Expr):
    """
    multiplication operator

    Example: `2 * 3`
    """

    left: Expr
    right: Expr


@dataclass
class EltMulOp(Expr):
    """
    element-wise multiplication operator

    Example: `[1,2,3] .* [4,5,6] == [4,10,18]`
    """

    left: Expr
    right: Expr


@dataclass
class AddOp(Expr):
    """
    addition operator

    Example: `2 + 3`
    """

    left: Expr
    right: Expr


@dataclass
class SubOp(Expr):
    """
    addition operator

    Example: `2 - 3`
    """

    left: Expr
    right: Expr


@dataclass
class DivOp(Expr):
    """
    division operator

    Example `1 / 2`
    """

    num: Expr
    den: Expr


@dataclass
class ModuloOp(Expr):
    """
    modulus operator

    Example `a % n`
    """

    dividend: Expr
    divisor: Expr


@dataclass
class IDivOp(Expr):
    """
    integer division operator

    Example `5 %/% 2`
    """

    num: Expr
    den: Expr


@dataclass
class PowOp(Expr):
    """
    Raise an expression to a power

    Example: `x^y`
    """

    base: Expr
    exponent: Expr


@dataclass
class EltPowOp(Expr):
    """
    Raise an expression to a power, elementwise

    Example: `[1,2,3].^[2,3,1]`
    """

    base: Expr
    exponent: Expr


@dataclass
class EqOp(Expr):
    """
    equality operator

    Example: `2 == 3`
    """

    left: Expr
    right: Expr


@dataclass
class NeqOp(Expr):
    """
    non-equality operator

    Example: `2 != 3`
    """

    left: Expr
    right: Expr


@dataclass
class LeOp(Expr):
    """
    less than operator

    Example: `2 < 3`
    """

    left: Expr
    right: Expr


@dataclass
class LeEqOp(Expr):
    """
    less than or equal operator

    Example: `2 <= 3`
    """

    left: Expr
    right: Expr


@dataclass
class GrOp(Expr):
    """
    greater than operator

    Example: `2 > 3`
    """

    left: Expr
    right: Expr


@dataclass
class GrEqOp(Expr):
    """
    greater than or equal operator

    Example: `2 >= 3`
    """

    left: Expr
    right: Expr


@dataclass
class TernaryOp(Expr):
    """
    the ternary operator (conditional expression)

    Example: `x == y ? a : b`
    """

    cond: Expr
    expr_if_true: Expr
    expr_if_false: Expr


@dataclass
class LogicOrOp(Expr):
    """
    logical or operator
    """

    left: Expr
    right: Expr


@dataclass
class LogicAndOp(Expr):
    """
    logical and operator
    """

    left: Expr
    right: Expr


@dataclass
class Par(Expr):
    """
    parentheses

    Example: `(1 + 2) * 3`
    """

    content: Expr


@dataclass
class Call(Expr):
    """
    object representing a function call

    Example `log_sum_exp(a, b)`
    """

    func_name: str
    arguments: Expr | list[Expr]

    def __post_init__(self) -> None:
        if isinstance(self.arguments, Expr):
            self.arguments = [self.arguments]


DistSuffix = Literal["lpdf", "lpmf", "lcdf", "lccdf"]


@dataclass
class PCall(Expr):
    """
    object representing a function call

    Example `normal_lpdf(a | b, c)`
    """

    dist_name: str
    obs: Expr
    parameters: Expr | list[Expr]
    suffix: DistSuffix

    def __post_init__(self) -> None:
        if isinstance(self.parameters, Expr):
            self.parameters = [self.parameters]


@dataclass
class MixinExpr(Expr):
    """
    string mixin, allows for arbitrary stan code snippets
    """

    code: str


"""
Statements in the Stan language
"""


@dataclass
class EmptyStmt(Stmt):
    """
    Empty Stan statement
    """

    pass


@dataclass
class ExpressionStmt(Stmt):
    """
    Make statements from expressions
    """

    expression: Expr


@dataclass
class MixinStmt(Stmt):
    """
    string mixin, allows for arbitrary stan code snippets
    """

    code: str


@dataclass
class Decl(Stmt):
    """
    object representing an uninitialized declaration

    Example: `real x;`, `matrix[2,3] M;`
    """

    var: Var


@dataclass
class DeclList(Stmt):
    """
    object representing multiple declarations of the same type

    Example: `real<lower=0> x, y, z;`
    """

    var_type: Type
    names: list[str]


@dataclass
class DeclAssign(Stmt):
    """
    object representing a declaration-assignment

    Example: `real x = sum(y);`
    """

    var: Var
    rhs: Expr


@dataclass
class Assign(Stmt):
    """
    object representing an assignment

    Example: `x = sum(y);`
    """

    lhs: Expr
    rhs: Expr


@dataclass
class AddAssign(Stmt):
    """
    addition-assignment

    Example `target += normal_lpdf(x | 0, 1);`
    """

    lhs: Expr
    rhs: Expr


@dataclass
class Sample(Stmt):
    """
    object representing a sampling statement

    Example: `x ~ normal(0, 1);`
    """

    lhs: Expr
    rhs: Call


@dataclass
class Return(Stmt):
    """
    return statement
    """

    ret: Expr


@dataclass
class Break(Stmt):
    """
    break statement
    """

    pass


@dataclass
class Scope(Stmt):
    """
    scoped statements
    """

    content: list[Stmt]
    indentation: int = 4


@dataclass
class FuncDef(Stmt):
    """
    object representing a Stan function definition
    """

    return_type: Type
    name: str
    arguments: list[Var]
    func_body: list[Stmt]


@dataclass
class ForLoop(Stmt):
    """
    a for loop in Stan.

    Example: `for ( i in 1:10 ) { u[i] ~ normal(0, 1); }`
    """

    index: Var
    sequence: Expr
    content: Stmt


@dataclass
class IfStatement(Stmt):
    """
    Stan if statement

    Example: `if ( c == 0 ) { n = n + 1; }`
    """

    condition: Expr
    content: Stmt


@dataclass
class IfElseStatement(Stmt):
    """
    Stan if-else statement

    Example:
    ```
    if ( c == 0 ) {
        n = n + 1;
    } else {
        n = n - 1;
    }
    ```
    """

    condition: Expr
    content_if_true: Stmt
    content_if_false: Stmt


## Definition of a general Stan model

model_block_names = [
    "functions",
    "data",
    "transformed data",
    "parameters",
    "transformed parameters",
    "model",
    "generated quantities",
]


@dataclass
class StanModel(Stmt):
    """
    object representing a Stan model.
    """

    functions: list[Stmt] | None
    data: list[Stmt]
    transformed_data: list[Stmt] | None
    parameters: list[Stmt]
    transformed_parameters: list[Stmt] | None
    model: list[Stmt]
    generated_quantities: list[Stmt] | None


# some convenience functions for generating often-used values/patterns


def one():
    return LiteralInt(1)


def izero():
    return LiteralInt(0)


def rzero():
    return LiteralReal(0.0)


def intVar(name: str):
    return Var(Int(), name)


def realVar(name: str):
    return Var(Real(), name)


def expandType(base: Type, shape: tuple[Expr, ...]) -> Array:
    """
    Make an Array type with base type `base`.
    This takes care of the case where `base` itself is an Array
    as expressions like `array[3] array[2] real` are illegal in Stan.
    New dimensions are added on the left.
    """
    match base:
        case Array(base_base, base_shape):
            return Array(base_base, shape + base_shape)
        case _:  # catch-all
            return Array(base, shape)


def expandVar(var: Var, shape: tuple[Expr, ...]) -> Var:
    """
    expand the type of a variable
    """
    return Var(expandType(var.var_type, shape), var.name)


def expandVecType(tp: Type, dim: Expr) -> Type:
    match tp:
        case Real():
            return RowVector(dim, data=tp.data, lower=tp.lower, upper=tp.upper)
        case Vector(n):
            return Matrix(n, dim, data=tp.data, lower=tp.lower, upper=tp.upper)
        case _:
            raise Exception("invalid Type in expandVecType. tp must be Real or Vector")


def expandVecVar(var: Var, dim: Expr) -> Var:
    """
    expand a scalar to a row_vector, and a vector to a matrix
    """
    tp = expandVecType(var.var_type, dim)
    return Var(tp, var.name)


def fullRange() -> Range:
    return Range(None, None)


def comment(line: str) -> Stmt:
    """
    short-hand for an empty statement with a comment.
    Used to add comment lines to code blocks.
    """
    return EmptyStmt(comment=line)


def gen_decl_lists(variables: Var) -> list[DeclList]:
    """
    Sort variables by type and create declaration lists for each unique type
    """
    types = [var.var_type for var in variables]
    utypes = []
    for tp in types:
        if tp not in utypes:
            utypes.append(tp)
    decl_lists = [
        DeclList(tp, [var.name for var in variables if var.var_type == tp])
        for tp in utypes
    ]
    return decl_lists


def square(x: Expr) -> Expr:
    return PowOp(x, LiteralInt(2))
