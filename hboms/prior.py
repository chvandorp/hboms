from typing import Optional, Callable

from . import stanlang as sl


class Prior:
    def __init__(self, name: str, params: list[float], transform: Callable = lambda x: x) -> None:
        """
        transform is added for default priors on matrix-valued parameters.
        The default student-t distributions does not accept matrix values,
        so we transform it with to_vector. In this case we use 
        ```
        transform = lambda x: sl.Call("to_vector", x)
        ```
        TODO: make this more general and add jacobian corrections!
        TODO: parameters could also be model parameters instead of literals
        """
        self._name = name
        self._params = params
        self._transform = transform

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> list[float]:
        return self._params

    def gen_sampling_stmt(self, par: sl.Expr) -> list[sl.Stmt]:
        """
        Generate "sampling statement" from the prior distribution for the given parameter expression.
        """
        lits: list[sl.Expr] = [sl.LiteralReal(x) for x in self._params]
        trans_par = self._transform(par)
        stmts: list[sl.Stmt] = [sl.Sample(trans_par, sl.Call(self._name, lits))]
        return stmts
    
    def gen_rng_expr(self) -> sl.Expr:
        """
        Generate an expression that generates a sample from the prior distribution.        
        """
        lits: list[sl.Expr] = [sl.LiteralReal(x) for x in self._params]
        ## FIXME: apply transform? Add inverse transform with jacobian correction?
        expr: sl.Expr = sl.Call(f"{self._name}_rng", lits)
        return expr