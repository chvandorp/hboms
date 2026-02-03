from typing import Callable
from abc import ABC

from . import stanlang as sl

class Prior(ABC):
    def gen_sampling_stmt(self, par: sl.Expr) -> list[sl.Stmt]:
        """
        Generate "sampling statement" from the prior distribution for the given parameter expression.
        """
        raise NotImplementedError("gen_sampling_stmt method not implemented in base Prior class.")


class AbsContPrior(Prior):
    def __init__(self, name: str, params: list[float], transform: Callable = lambda x: x) -> None:
        """
        Absolute continuous prior distribution (w.r.t. Lebesgue measure).

        `transform` is added for default priors on matrix-valued parameters.
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
    

class DiracDeltaPrior(Prior):
    def __init__(self, param: str | float) -> None:
        """
        A Dirac delta prior that fixes a parameter to a specific value.

        Parameters:
        -----------
        param: str | float
            The fixed value or the name of the parameter to which
            this prior is anchored.
        """
        self._param = param

    @property
    def name(self) -> str:
        return "dirac_delta"

    @property
    def param(self) -> str | float:
        return self._param
    
    @property
    def var(self) -> sl.Stmt:
        match self._param:
            case str():
                ## FIXME: resolve the type during parsing of the hboms model
                return sl.Var(sl.UnresolvedType(), self._param)
            case float():
                return sl.LiteralReal(self._param)
            
    def gen_defining_stmt(self, par: sl.Expr) -> list[sl.Stmt]:
        """
        Generate assignment statement to set the parameter equal to the fixed value.
        """
        decl = sl.DeclAssign(par, self.var)
        return [decl]
    
    def gen_sampling_stmt(self, par: sl.Expr) -> list[sl.Stmt]:
        """
        Dirac delta priors do not generate sampling statements.
        """
        return []
    