"""
Objects and functions used for HBOMS model definition.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from . import utilities as util
from . import stanlang as sl


@dataclass
class StanPrior:
    """
    Representation of a prior distribution in Stan.

    Notice that the name should correspond to a valid Stan distribution,
    or a user-defined distribution in the Stan model code.

    Example:
        Standard normal prior:
        `prior = StanPrior(name="normal", params=[0, 1])`
        Exponential prior with rate 2:
        `prior = StanPrior(name="exponential", params=[2])`

    Attributes:
        name (str): Name of the Stan distribution.
        params (list[float]): Parameters of the distribution.

    """
    name: str
    params: list[float]


@dataclass
class DiracDeltaPrior:
    """
    Representation of a Dirac delta prior (point mass) for a parameter.

    This can be used to fix e.g. a location or scale parameter to a specific value:
    possibly the location or scale of another parameter.

    Example:
        Fixing a parameter to a constant value:
        `prior = DiracDeltaPrior(param=5.0)`
    
    Example:
        Anchoring a parameter to another parameter named "alpha":
        `prior = DiracDeltaPrior(param="alpha")`
    
    Example:
        Let a be a random parameter with scale `scale_a`, and fix b's scale to a's scale:
        ```
        param_a = Parameter(name="a", value=0.0, par_type="random")
        ddp = DiracDeltaPrior(param="scale_a")
        param_b = Parameter(name="b", value=1.0, par_type="random", scale_prior=ddp)
        ```

    Attributes:
        param (str | float): The fixed value or the name of the parameter to which
                             this prior is anchored.
    """
    param: str | float


@dataclass
class Parameter:
    """
    Representation of a model parameter.

    Example:
        A random parameter "k" with initial value 0.1, scale 0.05, lower bound 0.0, upper bound 1.0,
        affected by covariates "age" and "weight", with a normal prior on its location and
        an exponential prior on its scale:
        ```
        prior_loc = StanPrior(name="normal", params=[0.0, 1.0])
        prior_scale = StanPrior(name="exponential", params=[1.0])
        param_k = Parameter(
            name="k",
            value=0.1,
            par_type="random",
            scale=0.05,
            covariates=["age", "weight"],
            lbound=0.0,
            ubound=1.0,
            loc_prior=prior_loc,
            scale_prior=prior_scale
        )
        ```

    Attributes:
        name (str): Name of the parameter.
        value (float | list[float]): Initial value(s) of the parameter.
        par_type (str): Type of the parameter (e.g., "fixed", "random").
        scale (Optional[float]): Initial scale of the parameter (for random parameters).
        covariates (Optional[list[str]]): List of covariate names affecting the parameter.
        cw_values (Optional[dict[str, float | list[float]]]): Covariate-wise initial values for the parameter.
        space (Optional[str]): Parameter space (e.g., "real ", "vector").
        lbound (Optional[float]): Lower bound of the parameter.
        ubound (Optional[float]): Upper bound of the parameter.
        prior (Optional[StanPrior]): Prior distribution for the parameter (for fixed, indiv).
        loc_prior (Optional[StanPrior]): Prior for the location of the parameter (for random).
        scale_prior (Optional[StanPrior]): Prior for the scale of the parameter (for random).
        level (Optional[str]): Hierarchical level of the parameter (for hierarchical parameters).
        level_type (Optional[str]): Type of the hierarchical level (e.g., "fixed", "random").
        level_scale (Optional[float]): Initial scale of the hierarchical level (for random levels).
        level_scale_prior (Optional[StanPrior]): Prior for the scale of the hierarchical level (for random levels).
        noncentered (Optional[bool]): Whether to use non-centered parameterization (for random levels).

    """
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
    """
    Representation of an observation distribution in Stan.

    Example:
        A normal distribution for observation "y" with parameters "mu" and "sigma":
        `dist = StanDist(name="normal", obs_name="y", params=["mu", "sigma"])`

    Attributes:
        name (str): Name of the Stan distribution.
        obs_name (str): Name of the observation variable.
        params (list[str]): List of parameter names for the distribution.
    """
    name: str
    obs_name: str
    params: list[str]


@dataclass
class Variable:
    """
    Representation of a model variable (state or transformed state).
    This can be a scalar or vector-valued variable, but is always real-valued,
    or has real-valued components.

    Example:
        A state variable "S" representing susceptible individuals:
        `var = Variable(name="S")`

    Example:
        A vector-valued variable "velocity" with dimension 3:
        `var = Variable(name="velocity", dim=3)`

    Attributes:
        name (str): Name of the variable.
        dim (Optional[int]): Dimension of the variable (for vector-valued variables).
    """
    name: str
    dim: Optional[int] = None


@dataclass
class Observation:
    """
    Representation of an observed variable. These are typically linked to
    the model state via StanDist objects.

    Example:
        A real-valued observation "y" that is not censored:
        `obs = Observation(name="y")`

    Example:
        A left-censored viral load observation "VL" of type "real":
        `obs = Observation(name="VL", data_type="real", censored=True)`

    Example:
        A count variable "cases" of type "int":
        `obs = Observation(name="cases", data_type="int")`

    Attributes:
        name (str): Name of the observation variable.
        data_type (str): Data type of the observation (e.g., "real", "int").
        censored (bool): Whether the observation is censored.
    """
    name: str
    data_type: str = "real"
    censored: bool = False


@dataclass
class Covariate:
    """
    Representation of a covariate variable.

    Example:
        A continuous covariate "age":
        `cov = Covariate(name="age", cov_type="cont")`

    Example:
        A categorical covariate "treatment" with categories "placebo" and "drug":
        `cov = Covariate(name="treatment", cov_type="cat", categories=["placebo", "drug"])`

    Example:
        A vector-valued covariate "biomarkers" with dimension 5:
        `cov = Covariate(name="biomarkers", cov_type="cont", dim=5)`

    Attributes:
        name (str): Name of the covariate.
        cov_type (str): Type of the covariate ("cont" for continuous, "cat" for categorical).
        categories (Optional[list[str]]): Categories for categorical covariates.
        dim (Optional[int]): Dimension of the covariate (for vector-valued covariates).
    """
    name: str
    cov_type: str = "cont"
    categories: Optional[list[str]] = None
    dim: Optional[int] = None


@dataclass
class Correlation:
    """
    Representation of a correlation structure among parameters.
    The intensity attribute can be used to specify the a priori strength of the correlation.
    The intenisty is used in the prior definition for the correlation matrix,
    e.g., in a LKJ prior.

    Example:
        A correlation structure among parameters "alpha", "beta", and "gamma" with a specified correlation matrix:
        ```corr = Correlation(
            params=["alpha", "beta", "gamma"],
            value=np.array([[1.0, 0.5, 0.3],
                            [0.5, 1.0, 0.2],
                            [0.3, 0.2, 1.0]]),
            intensity=1.0
        )```

    Attributes:
        params (list[str]): List of parameter names involved in the correlation.
        value (Optional[np.ndarray]): Initial correlation matrix, identity matrix if None.
        intensity (Optional[float]): Intensity of the correlation structure (used for the prior).
    """
    params: list[str]
    value: Optional[np.ndarray] = None
    intensity: Optional[float] = None


@dataclass
class HbomsModelDef:
    """
    Representation of an HBOMS model definition.
    This is currently only used internally to pass model definitions around,
    and "compile" them into Stan code. Users typically define models
    using the `hboms.HbomsModel` class.
    """

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
