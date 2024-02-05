from typing import Optional, Callable

from . import stanlang as sl


class StateVar:
    """A class representing a state variable in the system of ODEs

    Args:
        name (str): the name of the state variable. The same name should
        be used in the ODE model definition.

    Kwargs:
        stan_type (sl.Type): The Stan type of the state variable. By default,
        this is a `sl.Real` (which is a `real` in Stan).
        dim (int): If the user specifies a dimension, the type of the
        state variable becomes `sl.Vector` with the given dimension.
    """

    def __init__(
        self, name: str, stan_type: sl.Type = sl.Real(), dim: Optional[int] = None
    ) -> None:
        self._name = name
        self._stan_type: sl.Type = (
            sl.Vector(sl.LiteralInt(dim)) if dim is not None else stan_type
        )

    def is_scalar(self) -> bool:
        return isinstance(self.stan_type, sl.Real)

    @property
    def name(self) -> str:
        return self._name

    @property
    def stan_type(self) -> sl.Type:
        return self._stan_type

    def var(self, rename_func: Callable = lambda x: x) -> sl.Var:
        return sl.Var(self._stan_type, rename_func(self.name))

    def gen_decl(self, rename_func: Callable = lambda x: x) -> sl.Decl:
        return sl.Decl(self.var(rename_func))

    def gen_decl_assign(
        self, rvalue: sl.Expr, rename_func: Callable = lambda x: x
    ) -> sl.DeclAssign:
        return sl.DeclAssign(self.var(rename_func), rvalue)


class State:
    """The class State is a wrapper around list[StateVar].

    It provides code generation functions and indexing functions
    """

    def __init__(self, state_vars: list[StateVar]) -> None:
        self._state_vars = state_vars
        dims = [x.stan_type.flat_dim() for x in state_vars]
        self._dims = dims
        self._idxs = [
            (
                sum(dims[:i], start=sl.LiteralInt(1)),
                sum(dims[: i + 1], start=sl.LiteralInt(0)),
            )
            for i in range(len(dims))
        ]

    def gen_unpack_stmt(
        self,
        vec: sl.Var,
        rename_func: Callable = lambda x: x,
        transform_func: Callable = lambda x: x,
        declare: bool = True,
        statevar_names: Optional[list[str]] = None,
    ) -> list[sl.Stmt]:
        """
        Generate code to unpack a state vector.

        Parameters
        ----------
        vec : sl.Var
            Stan variable corresponding to the state vector.
        rename_func : Callable, optional
            Function to rename state variables. The default is lambda x: x.
        transform_func : Callable, optional
            Function to manipulate state variables. The default is lambda x: x.
        declare : bool, optional
            Should the state variables be declared?. The default is True.
        statevar_names : Optional[list[str]], optional
            List of state variable names that should be unpacked.
            None means unpack all state variables. The default is None.

        Raises
        ------
        Exception
            Currently only sl.Real and sl.Vector
            are the only supported state variable types

        Returns
        -------
        statements : list[sl.Stmt]
            A list of statements that unpack a flat state vector into
            individual state variables.
        """
        statements: list[sl.Stmt] = []

        # TODO: currently this only works for vectors and reals!!
        for i, x in enumerate(self._state_vars):
            # skip state variables that are not in statevar_names
            if statevar_names is not None and x.name not in statevar_names:
                continue
            # get the right slice of vec
            match x.stan_type:
                case sl.Real():
                    vec_slice = vec.idx(self._idxs[i][1])
                case sl.Vector():
                    vec_slice = vec.idx(sl.Range(*self._idxs[i]))
                case _:
                    raise Exception("unsupported stan type")
            x_renamed = sl.Var(x.stan_type, rename_func(x.name))
            if declare:
                stmt: sl.Stmt = sl.DeclAssign(transform_func(x_renamed), vec_slice)
            else:
                stmt: sl.Stmt = sl.Assign(transform_func(x_renamed), vec_slice)
            statements.append(stmt)
        return statements

    def all_scalar(self, statevar_names: Optional[list[str]] = None) -> bool:
        """
        Determine if all state variables are scalars.
        If this is the case, the code can be significantly simplified.
        For instance by returning a literal vector.

        Parameters
        ----------
        statevar_names : Optional[list[str]], optional
            A list of state variable names that you want include for the test.
            If this is None, all state variables are included.
            The default is None.

        Returns
        -------
        bool
            True if all state variables are scalars, False if at least one is not.

        """

        if statevar_names is None:
            statevar_names = [x.name for x in self._state_vars]
        is_scalar = [
            x.is_scalar() for x in self._state_vars if x.name in statevar_names
        ]
        return all(is_scalar)

    def gen_decl(self, rename_func: Callable = lambda x: x) -> list[sl.Decl]:
        """declare stan variables containing e.g. the derivative or initial state"""
        declarations = [x.gen_decl(rename_func=rename_func) for x in self._state_vars]
        decl_lists = sl.gen_decl_lists([decl.var for decl in declarations])
        # return declarations
        return decl_lists

    def gen_flatten(
        self, vec_name: str, rename_func: Callable = lambda x: x
    ) -> list[sl.Stmt]:
        """copy the state variables to a flat array or vector"""
        assignments: list[sl.Stmt] = []
        vec = sl.Var(
            sl.Vector(self.flat_dim()), vec_name
        )  ## FIXME: does this make sense?
        for i, x in enumerate(self._state_vars):
            var = sl.Var(x.stan_type, rename_func(x.name))
            match x.stan_type:
                case sl.Real():
                    stmt = sl.Assign(sl.IndexOp(vec, self._idxs[i][1]), var)
                    assignments.append(stmt)
                case sl.Vector():
                    ran = sl.Range(self._idxs[i][0], self._idxs[i][1])
                    stmt = sl.Assign(sl.IndexOp(vec, ran), var)
                    assignments.append(stmt)
                case _:
                    raise Exception("unsupported stan type")

        # TODO: currently this only works for vectors and reals!!
        return assignments

    def gen_literal_vector(self, rename_func: Callable = lambda x: x) -> sl.Expr:
        vec = sl.LiteralVector([x.var(rename_func) for x in self._state_vars])
        return vec

    def flat_dim(self) -> sl.Expr:
        dims = [x.stan_type.flat_dim() for x in self._state_vars]
        return sum(dims, start=sl.LiteralInt(0))
