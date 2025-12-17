from typing import Optional, Literal
from . import stanlang as sl
import numpy as np
from abc import ABC, abstractmethod


class Covariate(ABC):
    def __init__(self, name: str) -> None:
        self._name = name

    def __eq__(self, other) -> bool:
        return self._name == other._name

    def __le__(self, other) -> bool:
        return self._name < other._name

    def __hash__(self) -> int:
        return hash(self._name)

    @abstractmethod
    def get_type(self):
        return "undefined"

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def stan_type(self) -> sl.Type:
        pass

    @property
    def var(self) -> sl.Var:
        return sl.Var(self.stan_type, self.name)

    @abstractmethod
    def genstmt_data(self) -> list[sl.Decl]:
        pass


class ContCovariate(Covariate):
    def __init__(self, name: str, dim: Optional[int] = None) -> None:
        super().__init__(name)
        # dim is None means that the covariate is a scalar
        self._dim = dim

    def get_type(self):
        return "cont"

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    @property
    def dim_var(self) -> sl.Var:
        return sl.Var(sl.Int(lower=sl.one()), f"K_{self.name}")

    @property
    def stan_type(self) -> sl.Type:
        if self._dim is None:
            return sl.Real()
        return sl.Vector(self.dim_var)

    def weight_name(self, param_name: str) -> str:
        return f"weight_{self.name}_{param_name}"

    def weight_var(self, param_name: str) -> sl.Var:
        return sl.Var(self.stan_type, self.weight_name(param_name))

    def weight_value(self, value=0.0) -> float | np.ndarray:
        if self._dim == None:
            return value
        # else...
        return np.full(self._dim, value)

    def genstmt_data(self) -> list[sl.Decl]:
        RVar = sl.intVar("R")
        decls: list[sl.Decl] = []
        if self.dim is not None:
            decls.append(
                sl.Decl(
                    self.dim_var,
                    comment=f"dimension of vector-valued covariate {self.name}",
                )
            )
        decls.append(sl.Decl(sl.expandVar(self.var, (RVar,))))
        return decls


class CatCovariate(Covariate):
    def __init__(self, name: str, categories: list[str]) -> None:
        """
        To define a categorical covariate, we need its name and a
        list of categories.

        Parameters
        ----------
        name : str
            The name of the covariate.
        categories : list[str]
            a list with the name of each category.

        """
        super().__init__(name)
        self._cats = categories
        self._num_cats = len(categories)

    def get_type(self):
        return "cat"

    @property
    def stan_type(self) -> sl.Type:
        return sl.Int(lower=sl.one(), upper=self.num_cat_var)

    @property
    def num_cats(self) -> int:
        return self._num_cats

    @property
    def cats(self) -> list[str]:
        return self._cats
    
    @property
    def level_matrix_var(self) -> sl.Var:
        """
        This is used for when the covariate is used as for a random level.
        The matrix is of size (R x R), where R is the number of units.
        It defines which units are in the same category.
        
        Returns
        -------
        sl.Var
            A Stan variable representing the level matrix.
        """
        R = sl.intVar("R")
        return sl.Var(sl.Matrix(R, R), f"level_{self.name}")

    @property
    def num_cat_var(self) -> sl.Var:
        return sl.Var(sl.Int(lower=sl.one()), f"K_{self.name}")

    def loc_value(self, value) -> np.ndarray:
        return np.full(self._num_cats, value)

    def genstmt_data(self) -> list[sl.Decl]:
        RVar = sl.intVar("R")
        decls: list[sl.Decl] = []

        decls.append(
            sl.Decl(self.num_cat_var, comment=f"number of categories for {self.name}")
        )
        decls.append(sl.Decl(sl.expandVar(self.var, (RVar,))))

        return decls


def group_covars(
    covs: list[Covariate],
) -> tuple[list[ContCovariate], list[CatCovariate]]:
    cont_covs = [cov for cov in covs if isinstance(cov, ContCovariate)]
    cat_covs = [cov for cov in covs if isinstance(cov, CatCovariate)]
    return cont_covs, cat_covs


CovariateType = Literal["cont", "cat"]

def covar_dispatch(name: str, cov_type: CovariateType, **kwargs) -> Covariate:
    match cov_type:
        case "cont":
            return ContCovariate(name, **kwargs)
        case "cat":
            return CatCovariate(name, **kwargs)
        case _:
            raise Exception(f"invalid covariate type '{cov_type}'")
