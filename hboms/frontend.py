"""
Objects and functions used for HBOMS model definition.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from . import utilities as util
from . import stanlang as sl


@dataclass
class StanPrior:
    name: str
    params: list[float]


@dataclass
class Parameter:
    name: str
    value: float | list[float]
    par_type: str
    scale: Optional[float] = None
    covariates: Optional[list[str]] = None
    cw_values: Optional[dict[str, float | list[float]]] = None
    space: Optional[str] = None
    lbound: Optional[float] = 0.0
    ubound: Optional[float] = None
    prior: Optional[StanPrior] = None
    loc_prior: Optional[StanPrior] = None
    scale_prior: Optional[StanPrior] = None
    level: Optional[str] = None
    level_type: Optional[str] = None
    level_scale: Optional[float] = None
    level_scale_prior: Optional[StanPrior] = None
    noncentered: Optional[bool] = None

    def __str__(self) -> str:
        return print_parameter(self)


@dataclass
class StanDist:
    name: str
    obs_name: str
    params: list[str]


@dataclass
class Variable:
    name: str
    dim: Optional[int] = None


@dataclass
class Observation:
    name: str
    data_type: str = "real"
    censored: bool = False


@dataclass
class Covariate:
    name: str
    cov_type: str = "cont"
    categories: Optional[list[str]] = None
    dim: Optional[int] = None


@dataclass
class Correlation:
    params: list[str]
    value: Optional[np.ndarray] = None
    intensity: Optional[float] = None


@dataclass
class HbomsModelDef:
    name: str
    state: list[Variable]
    odes: str
    init: str
    params: list[Parameter]
    obs: list[Observation]
    dists: list[StanDist]
    trans_state: Optional[list[Variable]] = None
    transform: Optional[str] = None
    covariates: Optional[list[Covariate]] = None
    correlations: Optional[list[Correlation]] = None

    def __str__(self):
        return print_model(self)


def type_dispatch(tp: str) -> sl.Type:
    # TODO: parse other data types
    match tp:
        case "int":
            return sl.Int()
        case "real":
            return sl.Real()
        case _:
            return NotImplementedError(f"cannot convert type {tp}")


def print_model(model: HbomsModelDef) -> str:
    n = 4
    s = "Model(\n"
    # name
    s += util.indent(f"name = '{model.name}',", n) + "\n"
    # state
    s += util.indent("state = [", n) + "\n"
    for var in model.state:
        s += util.indent(str(var) + ",", 2 * n) + "\n"
    s += util.indent("],", n)
    # odes
    s += "\n"
    s += util.indent('odes = """', n) + "\n"
    s += model.odes.strip() + "\n"
    s += '""",\n'
    # init
    s += util.indent('init = """', n) + "\n"
    s += model.init.strip() + "\n"
    s += '""",\n'
    # params
    s += util.indent("params = [", n) + "\n"
    for par in model.params:
        s += util.indent(str(par) + ",", 2 * n) + "\n"
    s += util.indent("],", n) + "\n"
    # obs
    s += util.indent("obs = [", n) + "\n"
    for ob in model.obs:
        s += util.indent(str(ob) + ",", 2 * n) + "\n"
    s += util.indent("],", n) + "\n"
    # dists
    s += util.indent("dists = [", n) + "\n"
    for dist in model.dists:
        s += util.indent(str(dist) + ",", 2 * n) + "\n"
    s += util.indent("],", n) + "\n"
    # trans_state
    if model.trans_state is not None:
        s += util.indent("trans_state = [", n) + "\n"
        for var in model.trans_state:
            s += util.indent(str(var) + ",", 2 * n) + "\n"
        s += util.indent("],", n) + "\n"
    if model.transform is not None:
        s += util.indent('transform = """', n) + "\n"
        s += model.transform.strip() + "\n"
        s += '""",\n'
    if model.covariates is not None:
        s += util.indent("covariates = [", n) + "\n"
        for cov in model.covariates:
            s += util.indent(str(cov) + ",", 2 * n) + "\n"
        s += util.indent("],", n) + "\n"
    if model.correlations is not None:
        s += util.indent("correlations = [", n) + "\n"
        for corr in model.correlations:
            s += util.indent(str(corr) + ",", 2 * n) + "\n"
        s += util.indent("],", n) + "\n"
    s += ")"
    return s


def print_parameter(par):
    s = f"Parameter(name = {par.name}, value = {par.value}, par_type = {par.par_type}"
    if par.scale is not None:
        s += f", scale = {par.scale}"
    if par.covariates is not None:
        s += f", covariates = {par.covariates}"
    if par.space is not None:
        s += f", space = {par.space}"
    if par.lbound is not None:
        s += f", lbound = {par.lbound}"
    if par.ubound is not None:
        s += f", ubound = {par.ubound}"

    s += ")"

    return s
