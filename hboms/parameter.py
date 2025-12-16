from typing import List, Optional, Sequence, Callable, Literal
from abc import ABC, abstractmethod
import numpy as np
from functools import reduce
import copy

from . import utilities as util
from . import stanlang as sl
from .correlation import Correlation
from .covariate import Covariate, ContCovariate, CatCovariate, group_covars
from .prior import Prior
from .transform import constr_to_unconstr_float


def select_default_rand_param_dist(
    lbound: Optional[float], ubound: Optional[float]
) -> str:
    ## FIXME: better default prior system

    ## FIXME: non-zero lower bounds
    ## FIXME: (non-zero) upper bounds
    match (lbound, ubound):
        case (None, None):
            return "normal"
        case (float(), None) | (None, float()):
            return "lognormal"
        case (float(), float()):
            return "logitnormal"
        case _:
            # all other cases get (truncated) normal (FIXME!)
            return "normal"


def domain_transform(
    val: sl.Expr, lbound: Optional[float], ubound: Optional[float], array: bool = False,
    loc: Optional[sl.Expr] = None, scale: Optional[sl.Expr] = None
) -> sl.Expr:
    """
    Generate expressions for transforming unconstraint values to a bounded interval.

    Parameters
    ----------
    val : sl.Expr
        the unconstrained value to-be transformed.
    lbound : Optional[float]
        lower bound of the transformed value.
    ubound : Optional[float]
        upper bound of the transformed value.
    array : bool
        is val of type array? In that case we have to convert to array and back.
        The default is False
    loc : Optional[sl.Expr]
        optional location parameter for the transformation.
        used for e.g. noncentered parameters.
    scale : Optional[sl.Expr]
        optional scale parameter for the transformation.
        used for e.g. noncentered parameters.

    Raises
    ------
    Exception
        Raise an exception if the lbound and ubound types are invalid.

    Returns
    -------
    sl.Expr
        Stan expression for the transformed value.
    """

    def to_vec(x):
        return sl.Call("to_vector", [x]) if array else x

    def to_array(x):
        return sl.Call("to_array_1d", [x]) if array else x


    to_vec_applied = True
    match (loc, scale):
        case (None, None):
            # no linear transformation
            to_vec_applied = False
        case (sl.Expr(), None):
            # apply a linear transformation to the value
            val = loc + to_vec(val)
        case (None, sl.Expr()):
            # apply a linear transformation to the value
            val = to_vec(val) * scale
        case (sl.Expr(), sl.Expr()):
            # apply a linear transformation to the value
            val = loc + scale * to_vec(val)
        case _:
            raise Exception("invalid loc and scale parameters for domain transformation")

    if to_vec_applied:
        # do not apply to_vector twice..."""
        to_vec = lambda x: x

    match (lbound, ubound):
        case (None, None):
            return to_array(val) if to_vec_applied else val
        case (float(), None):
            exp_val = sl.Call("exp", [val])
            if lbound == 0.0:
                return to_array(exp_val) if to_vec_applied else exp_val
            # we have to convert to vector to translate
            exp_vec_val = to_vec(exp_val)
            translated_val = to_array(sl.LiteralReal(lbound) + exp_vec_val)
            return translated_val
        case (None, float()):
            vec_val = to_vec(val)
            exp_vec_val = sl.Call("exp", [sl.Negate(vec_val)])
            if ubound == 0.0:
                return to_array(sl.Negate(exp_vec_val))
            translated_val = sl.LiteralReal(ubound) - exp_vec_val
            return to_array(translated_val)
        case (float(), float()):
            expit_val = sl.Call("inv_logit", [val])
            lb = sl.LiteralReal(lbound)
            ub = sl.LiteralReal(ubound)
            if lbound == 0.0 and ubound == 1.0:
                return to_array(expit_val) if to_vec_applied else expit_val
            vec_expit_val = to_vec(expit_val)
            if lbound == 0.0:
                return to_array(ub * vec_expit_val)
            scaled_val = lb + sl.Par(ub - lb) * vec_expit_val
            return to_array(scaled_val)
        case _:
            raise Exception("invalid parameter bounds")


ParamSpace = Literal["real", "vector", "matrix", "rugged_vector", "spline"]

LevelType = Literal["fixed", "random"]


class Parameter(ABC):
    def __init__(
        self,
        name: str,
        value: float | list[float],  # TODO: extend types
        covariates: Optional[list[Covariate]] = None,
        cw_values: Optional[dict[str, float | list[float]]] = None,
        lbound: Optional[float] = 0.0,
        ubound: Optional[float] = None,
        space: ParamSpace = "real",
    ) -> None:
        self._name = name
        self._value = value
        covariates = [] if covariates is None else covariates
        self._contcovs, self._catcovs = group_covars(covariates)
        if len(self._catcovs) > 1:
            raise NotImplementedError(
                "parameters can currently only have a single categorical covariate"
            )
        # the user might (unintentionally) provide integer bounds.
        # Correct this without warning. Take care of the case where the bound
        # are None (+/- inf)
        def maybe_float(x):
            return None if x is None else float(x)

        self._lbound, self._ubound = maybe_float(lbound), maybe_float(ubound)
        self._cw_values = {} if cw_values is None else cw_values

        # determine if the var should get a data keyword
        data_kw = self.get_type() in ["const", "const_indiv"]

        # set parameter space
        self._space = space

        stan_type_kwargs = {
            "lower": None if self._lbound is None else sl.LiteralReal(self._lbound),
            "upper": None if self._ubound is None else sl.LiteralReal(self._ubound),
            "data": data_kw,
        }

        match space:
            case "real":
                self._shape = None
                self._stan_type = sl.Real(**stan_type_kwargs)
            case "vector":
                self._shape = sl.Var(sl.Int(), f"shape_{name}")
                self._stan_type = sl.Vector(self._shape, **stan_type_kwargs)
            case "matrix":
                self._shape = sl.Var(
                    sl.Array(sl.Int(), sl.LiteralInt(2)), f"shape_{name}"
                )
                self._stan_type = sl.Matrix(
                    self._shape.idx(sl.LiteralInt(1)),
                    self._shape.idx(sl.LiteralInt(2)),
                    **stan_type_kwargs,
                )
            case "rugged_vector" | "spline":
                raise NotImplementedError(
                    f"parameter space '{space}' not yet implemented"
                )
            case _:
                raise Exception(f"invalid parameter space given: '{space}'")
        self._var = sl.Var(self._stan_type, self._name)

    @abstractmethod
    def get_type(self) -> str:
        return "undefined"

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> float | list[float]:
        return self._value

    @value.setter
    def value(self, value: float | list[float]) -> None:
        self._value = value

    @property
    def lbound(self) -> Optional[float]:
        return self._lbound

    @property
    def ubound(self) -> Optional[float]:
        return self._ubound

    @property
    def space(self) -> ParamSpace:
        return self._space

    @property
    def shape(self) -> Optional[sl.Var]:
        return self._shape

    @property
    def var(self) -> sl.Var:
        return self._var

    @abstractmethod
    def expand_and_index_var(self, apply_idx=False) -> sl.Expr:
        """
        Implement for all parameter types below.
        This method should get the right expansion (e.g. `array[R]`)
        for the type and index (e.g. `theta[r]`) for the variable.
        Population-level parameters should just return `self.var`
        """
        return self._var

    def has_covs(self) -> bool:
        """
        Return True if the parameter has any covariates
        (categorical or continuous). Else return False.
        """
        return self._catcovs or self._contcovs

    def is_pop_level(self) -> bool:
        """
        return True if the parameter is always the same for all units,
        and therefore does not require indexing.
        """
        match self.get_type():
            case "fixed":
                return not self.has_covs()
            case "const" | "trans":
                return True
            case _: # includes random, indiv, const_indiv, trans_indiv, ...
                return False
    
    def is_const(self):
        """
        return True if the parameter is a constant (i.e. defined in the `data` block)
        """
        return self.get_type() in ["const", "const_indiv"]

    def genstmt_data(self) -> list[sl.Decl]:
        return []

    def genstmt_params(self) -> list[sl.Decl]:
        return []

    def genstmt_trans_params(self) -> list[sl.Stmt]:
        return []

    def genstmt_model(self) -> list[sl.Stmt]:
        return []

    def genstmt_data_simulator(self) -> list[sl.Decl]:
        return []
    
    def genstmt_gq_simulator(self, transform_func: Callable = lambda x: x) -> List[sl.Stmt]:
        return []
    
    def genstmt_prior_samples(self) -> List[sl.Stmt]:
        return []



## TODO a FixedParameter can have continuous covariates. How to implement?
class FixedParameter(Parameter):
    def __init__(
        self,
        name: str,
        value: float | list[float],
        covariates: Optional[list[Covariate]] = None,
        cw_values: Optional[dict[str, float | list[float]]] = None,
        lbound: Optional[float] = 0.0,
        ubound: Optional[float] = None,
        space: ParamSpace = "real",
        prior: Optional[Prior] = None,
    ) -> None:
        super().__init__(
            name,
            value,
            covariates=covariates,
            cw_values=cw_values,
            lbound=lbound,
            ubound=ubound,
            space=space,
        )

        # warn against the use of covariates: they corrently don't work
        if self._contcovs:
            raise NotImplementedError(
                "continuous covariates are not yet implemented for fixed parameters"
            )

        if prior is None:  # default prior
            match self._space:
                case "real" | "vector":
                    # default student-t will work fine for scalars and vectors
                    self._prior = Prior("student_t", [3.0, 0.0, 2.5])
                case "matrix":
                    # but matrix variables are not allowed. Add a transform!
                    self._prior = Prior(
                        "student_t",
                        [3.0, 0.0, 2.5],
                        transform=lambda x: sl.Call("to_vector", x),
                    )
                case _:
                    raise NotImplementedError(
                        "default priors for space '{self._space}' are not yet implemented"
                    )

        else:  # user-defined prior
            self._prior = prior

    @property
    def level(self) -> Optional[CatCovariate]:
        return self._catcovs[0] if len(self._catcovs) > 0 else None

    @property
    def level_type(self) -> LevelType:
        return "fixed"
    
    @property
    def prior_name(self) -> str:
        return self._prior.name

    def get_type(this):
        return "fixed"

    def expand_and_index_var(self, apply_idx=False) -> sl.Expr:
        if self.level is None:
            return self.var
        idx = self.level.var.idx(sl.intVar("r"))
        num_params = self.level.num_cat_var
        if apply_idx:
            return sl.expandVar(self.var, num_params).idx(idx)
        return sl.expandVar(self.var, num_params)

    def genstmt_data(self):
        decls: list[sl.Decl] = []
        match self.space:
            case "real":
                pass  # real values have no shape: keep it simple
            case "vector" | "matrix":
                decls.append(sl.Decl(self._shape))
            case _:
                raise Exception(f"invalid parameter space: {self.space}")
        return decls

    def genstmt_params(self) -> list[sl.Decl]:
        if len(self._catcovs) == 0:
            var = self.var
        else:  # TODO: multiple catcovs
            numcats = self._catcovs[0].num_cat_var
            var = sl.expandVar(self.var, (numcats,))
        decl = sl.Decl(var)
        return [decl]

    def genstmt_model(self) -> list[sl.Stmt]:
        sam = self._prior.gen_sampling_stmt(self._var)
        return sam

    def genstmt_data_simulator(self) -> list[sl.Decl]:
        # add data declarations (shapes) and parameter declarations
        return self.genstmt_data() + self.genstmt_params()
    
    def genstmt_prior_samples(self) -> List[sl.Stmt]:
        rng_expr = self._prior.gen_rng_expr()
        # create declaration and assignment statement
        # the variable name is `prior_sample_{parameter_name}`
        return [sl.DeclAssign(sl.Var(self.var.var_type, f"prior_sample_{self._name}"), rng_expr)]


class RandomParameter(Parameter):
    def __init__(
        self,
        name: str,
        value: float | list[float],  ## TODO: extend types (np.ndarray)
        covariates: Optional[list[Covariate]] = None,
        cw_values: Optional[dict[str, float | list[float]]] = None,
        lbound: Optional[float] = 0.0,
        ubound: Optional[float] = None,
        space: ParamSpace = "real",
        distribution: Optional[str] = None,
        scale: float = 1.0,  # TODO: extend types (np.ndarray)
        loc_prior: Optional[Prior] = None,
        scale_prior: Optional[Prior] = None,
        level: Optional[CatCovariate] = None,
        level_type: LevelType = "fixed",
        level_scale: float = 1.0,
        level_scale_prior : Optional[Prior] = None,
        noncentered: bool = False,
    ) -> None:
        super().__init__(
            name,
            value,
            covariates=covariates,
            cw_values=cw_values,
            lbound=lbound,
            ubound=ubound,
            space=space,
        )

        if distribution is None:
            # Don't use lbound and ubound as they could be non-floats!
            self._distribution = select_default_rand_param_dist(self._lbound, self._ubound)
        else:
            ## TODO: check that domain is compatible with distribution
            self._distribution = distribution

        if self._distribution == "logitnormal" and self.space != "real":
            raise NotImplementedError(
                "currently bounded parameters can only be scalars"
            )

        # determine Type of the loc and scale
        match self.space:
            case "real":
                loc_type = sl.Real()
                scale_type = sl.Real(lower=sl.rzero())
            case "vector":
                loc_type = sl.Vector(self.shape)
                scale_type = sl.Vector(self.shape, lower=sl.rzero())
            case "matrix":
                n, m = (self.shape.idx(sl.LiteralInt(i)) for i in [1, 2])
                loc_type = sl.Matrix(n, m)
                scale_type = sl.Matrix(n, m, lower=sl.rzero())
            case _:
                raise NotImplementedError("invalid space for random parameter")

        if len(self._catcovs) == 0:
            self._loc = sl.Var(loc_type, f"loc_{self._name}")
        else:  # TODO: multiple catcovs
            numcats = self._catcovs[0].num_cat_var
            self._loc = sl.Var(sl.Array(loc_type, (numcats,)), f"loc_{self._name}")

        self._scale_value = scale
        self._scale = sl.Var(scale_type, f"scale_{self._name}")
        # TODO: how does the space influence self._cws?
        self._cws = [cov.weight_var(self._name) for cov in self._contcovs]

        def_hyp_prior = Prior("student_t", [3.0, 0.0, 2.5])
        self._loc_prior = def_hyp_prior if loc_prior is None else loc_prior
        self._scale_prior = def_hyp_prior if scale_prior is None else scale_prior

        # store the level at which we have random effects. None means at the unit level
        self._level = level
        self._level_type = level_type
        if level is not None and level_type == "random" and self.space != "real":
            raise NotImplementedError(
                "parameters with group-specific random effects can only be scalars"
            )
        if level is not None:
            self._level_scale = sl.Var(
                sl.Real(lower=sl.rzero()), f"scale_{self.name}_{level.name}"
            )
        else:
            self._level_scale = None
        self._level_scale_value = level_scale
        # set the prior for level_scale to the default or the user-given prior
        # TODO: we could choose the make the default equal to the scale_prior.
        self._level_scale_prior = def_hyp_prior if level_scale_prior is None else level_scale_prior

        # at this point parameters with a fixed level can't have a categorical covariate
        # because we would have to "reduce" the covariate to the level (instead of the unit)
        if level is not None and level_type == "fixed" and self._catcovs:
            raise NotImplementedError(
                "parameters with a fixed level can't have a categorical covariate"
            )

        # centered or non-centered parameterization
        self._noncentered = noncentered

        # non-centered parameters are still very limited
        if self._noncentered:
            if self.space != "real":
                raise NotImplementedError(
                    "non-centered parameterization is only implemented for scalars"
                )
            if self.level is not None and self.level_type == "random":
                raise NotImplementedError(
                    "non-centered parameterization is not implemented for random levels"
                )
            if self._contcovs:
                raise NotImplementedError(
                    "non-centered parameterization is not implemented for continuous covariates"
                )


    @property
    def scale_value(self) -> float:
        return self._scale_value

    @property
    def loc(self) -> sl.Var:
        return self._loc

    @property
    def scale(self) -> sl.Var:
        return self._scale

    @property
    def distribution(self) -> str:
        return self._distribution

    @property
    def level(self) -> Optional[CatCovariate]:
        return self._level

    @property
    def level_type(self) -> LevelType:
        return self._level_type

    @property
    def level_scale(self) -> Optional[CatCovariate]:
        return self._level_scale

    @property
    def level_scale_value(self) -> float:
        return self._level_scale_value

    @property
    def noncentered(self) -> bool:
        return self._noncentered

    @property
    def loc_prior_name(self) -> str:
        return self._loc_prior.name
    
    @property
    def scale_prior_name(self) -> str:
        return self._scale_prior.name

    def get_type(this) -> str:
        return "random"

    def expand_and_index_var(self, apply_idx=False) -> sl.Expr:
        R, r = sl.intVar("R"), sl.intVar("r")
        has_fixed_level = self.level is not None and self.level_type == "fixed"
        idx = self.level.var.idx(r) if has_fixed_level else r
        num_params = self.level.num_cat_var if has_fixed_level else R
        if apply_idx:
            return sl.expandVar(self.var, (num_params,)).idx(idx)
        return sl.expandVar(self.var, (num_params,))

    def genstmt_data(self) -> list[sl.Stmt]:
        decls: list[sl.Decl] = []
        match self.space:
            case "real":
                pass  # real values have no shape: keep it simple
            case "vector" | "matrix":
                decls.append(sl.Decl(self._shape))
            case _:
                raise Exception(f"invalid parameter space: {self.space}")
        return decls

    def genstmt_params(self) -> list[sl.Decl]:
        R = sl.Var(sl.Int(), "R")
        # by default, each unit has it's own random effect,
        # but the random effect can also be specified on a group level
        # which is defined by a categorical covariate
        one_par_per_unit = self.level is None or self._level_type == "random"
        num_pars = R if one_par_per_unit else self.level.num_cat_var
        # expand the parameter to the right shape
        # choose between centered and non-centered parameterization
        if self._noncentered:
            # create a new variable for the random effect
            rand_effect_type = copy.deepcopy(self._stan_type)
            # the random effect is unrestricted: set lbound and ubound to None
            # without deepcopy this would modify self._stan_type as well!
            rand_effect_type.lower, rand_effect_type.upper = None, None
            rand_effect = sl.Var(rand_effect_type, f"rand_{self._name}")
            array = sl.expandVar(rand_effect, (num_pars,))
        else:
            array = sl.expandVar(self.var, (num_pars,))
        decls = [sl.Decl(x) for x in [array, self._loc, self._scale] + self._cws]
        # declare an additional scale if the parameter has additional group-specific random effects
        if self.level is not None and self.level_type == "random":
            decls.append(sl.Decl(self.level_scale))
        return decls

    def genstmt_trans_params(self) -> List[sl.Stmt]:
        # if the parameter is non-centered, we have to transform it
        # otherwise we can just sample the parameter directly and don't need to transform
        if not self._noncentered:
            return []
        # else...
        R, r = sl.intVar("R"), sl.intVar("r") ## TODO: levels!
        one_par_per_unit = self.level is None or self._level_type == "random"
        num_pars = R if one_par_per_unit else self._level.num_cat_var
        array = sl.expandVar(self.var, (num_pars,))
        rand_effect = sl.Var(self._stan_type, f"rand_{self._name}")
        rand_effect_array = sl.expandVar(rand_effect, (num_pars,))
        # allow for categorical covariates by selecting the correct loc value
        loc = self._loc 
        if self._catcovs:
            cat = self._catcovs[0].var.idx(r) # the category for unit r
            loc = loc.idx(cat) # the loc for that category
        scale = self._scale
        # linear transformation of the random effect
        lin_trans_rand_effect = loc + scale * rand_effect_array.idx(r)
        decl = sl.Decl(array)
        # use a for loop to fill the array with transformed values
        for_loop = sl.ForLoop(
            r,
            sl.Range(sl.one(), num_pars),
            sl.Assign(
                array.idx(r),
                domain_transform(lin_trans_rand_effect, self._lbound, self._ubound),
            ),
        )
        return [decl, for_loop]


    def genstmt_hyper_prior(self) -> list[sl.Stmt]:
        """
        Generate a list of statements for the hyper prior of the random parameter.
        This includes the loc and scale parameters, as well as the covariate weights.

        This method is used in genstmt_model, and also in the genstmt_model 
        method of the ParameterBlock class.
        """
        priors: list[sl.Stmt] = []
        priors += self._loc_prior.gen_sampling_stmt(self._loc)
        priors += self._scale_prior.gen_sampling_stmt(self._scale)
        # add hyper prior for covariate weights (TODO: customize)
        hyper_prior = Prior("student_t", [3.0, 0.0, 2.5])
        for x in self._cws:
            priors += hyper_prior.gen_sampling_stmt(x)
        # add prior for level scale parameter
        if self.level is not None and self.level_type == "random":
            priors += self._level_scale_prior.gen_sampling_stmt(self._level_scale)
        return priors


    def genstmt_model(self) -> list[sl.Stmt]:
        R = sl.Var(sl.Int(), "R")
        unit_index = sl.Var(sl.Int(), "r")
        unit_range = sl.Range(sl.one(), R)

        covs = [sl.expandVar(cn.var, (R,)) for cn in self._contcovs]
        # we have to transpose vector-valued weights: weight' * cov = scalar
        cwts = [
            sl.TransposeOp(cw) if cov.dim is not None else cw
            for cov, cw in zip(self._contcovs, self._cws)
        ]

        # list of expressions to-be-returned
        priors: list[sl.Stmt] = []

        # in case of non-centered parameterization, we just sample the random effect
        # FIXME: this needs to be more general!
        if self._noncentered:
            one_par_per_unit = self.level is None or self._level_type == "random"
            num_pars = R if one_par_per_unit else self._level.num_cat_var
            rand_effect = sl.Var(self._stan_type, f"rand_{self._name}")
            rand_array = sl.expandVar(rand_effect, (num_pars,))
            priors.append(sl.Sample(rand_array, sl.Call("std_normal", [])))
            priors += self._loc_prior.gen_sampling_stmt(self._loc)
            priors += self._scale_prior.gen_sampling_stmt(self._scale)
            return priors

        # select the correct loc from the loc array in the case of categorical covariates
        if len(self._contcovs) == 0:
            if len(self._catcovs) == 0:
                loc = self._loc
            elif len(self._catcovs) > 0:
                catcov_var = sl.expandVar(self._catcovs[0].var, (R,))
                loc = self._loc.idx(catcov_var)

            # take care of the parameter space.
            # TODO: simplify and apply to other cases below!
            match self.space:
                case "real":
                    if self.level is not None and self.level_type == "random":
                        if len(self._catcovs) > 0:
                            # loc already has the right dimension, but needs to be a vector
                            mu = sl.Call("to_vector", loc)
                        else:  # just repeat the shared loc parameter
                            mu = sl.Call("rep_vector", [self.loc, R])
                        level_mat = sl.Var(sl.Matrix(R, R), f"level_{self.level.name}")
                        eye = sl.Call("identity_matrix", R)
                        sigma = (
                            sl.square(self.scale) * eye
                            + sl.square(self.level_scale) * level_mat
                        )
                        vec_var = sl.Call("to_vector", self.var)
                        # translate the distribution to a multivariate version
                        match self._distribution:
                            case "normal":
                                prior_stmts = [
                                    sl.Sample(
                                        vec_var, sl.Call("multi_normal", [mu, sigma])
                                    )
                                ]
                            case "lognormal":
                                jac_correction = sl.AddAssign(
                                    sl.realVar("target"),
                                    sl.Negate(sl.Call("log", vec_var)),
                                )
                                prior_stmts = [
                                    sl.Sample(
                                        sl.Call("log", vec_var),
                                        sl.Call("multi_normal", [mu, sigma]),
                                    ),
                                    jac_correction,
                                ]
                            case _:
                                raise NotImplementedError(
                                    f"random effects distribution {self._distribution} not implemented for random level"
                                )
                    else:  # no random level
                        dist_pars = [loc, self._scale]
                        sam_var = self.var # the variable to-be-sampled
                        if self._distribution == "logitnormal":
                            # logitnormal is defined in the functions block
                            # and requires lower and upper bounds as additional parameters
                            bound_pars = [
                                sl.LiteralReal(self._lbound),
                                sl.LiteralReal(self._ubound),
                            ]
                            dist_pars += bound_pars
                            
                        elif self._distribution == "lognormal":
                            to_vec = lambda x: sl.Call("to_vector", x)
                            # apply a linear transformation prior to sample statement
                            if self._lbound is not None and self._lbound != 0.0:
                                sam_var = to_vec(self.var) - sl.LiteralReal(self._lbound)
                            if self._ubound is not None and self._ubound == 0.0:
                                sam_var = sl.Negate(to_vec(self.var)) ## FIXME: should we take the reciprocal?
                            if self._ubound is not None and self._ubound != 0.0:
                                sam_var = sl.LiteralReal(self._ubound) - to_vec(sam_var)
                            
                        prior_stmts = [
                            sl.Sample(
                                sam_var,
                                sl.Call(self._distribution, dist_pars),
                            )
                        ]

                    priors += prior_stmts
                case "vector":
                    var = sl.expandVar(self.var, (R,)).idx(unit_index)
                    for_loop = sl.ForLoop(
                        unit_index,
                        unit_range,
                        sl.Sample(
                            var,
                            sl.Call(self._distribution, [loc, self._scale]),
                        ),
                    )
                    priors.append(for_loop)
                case "matrix":
                    var = sl.expandVar(self.var, (R,)).idx(unit_index)
                    to_vec = lambda x: sl.Call("to_vector", x)
                    for_loop = sl.ForLoop(
                        unit_index,
                        unit_range,
                        sl.Sample(
                            to_vec(var),
                            sl.Call(
                                self._distribution, [to_vec(loc), to_vec(self._scale)]
                            ),
                        ),
                    )
                    priors.append(for_loop)
                case _:
                    raise NotImplementedError(
                        f"parameter space {self.space} not implemented for random parameter"
                    )

        else:  # meaning that we have continuous covariates!
            # we have to use a for loop (do we?)

            # FIXME: implement this for levels, and refactor the code!!

            loc_terms: list[sl.Expr] = []

            # select the correct loc from the loc array in the case of categorical covariates
            if len(self._catcovs) == 0:
                loc = self._loc
            elif len(self._catcovs) == 1:
                catcov_var = sl.expandVar(self._catcovs[0].var, (R,))
                # in the for loop, the categories have to be indexed with the unit's index
                indexed_catcov_var = catcov_var.idx(unit_index)
                loc = self._loc.idx(indexed_catcov_var)
            else:
                raise NotImplementedError(
                    "parameters can currently only have a single categorical covariate"
                )

            loc_terms.append(loc)
            loc_terms += [cwt * cov.idx(unit_index) for cov, cwt in zip(covs, cwts)]
            loc_sum = reduce(lambda a, b: a + b, loc_terms)
            var = sl.expandVar(self.var, (R,)).idx(unit_index)

            for_loop = sl.ForLoop(
                unit_index,
                unit_range,
                sl.Sample(
                    var,
                    sl.Call(self._distribution, [loc_sum, self._scale]),
                ),
            )

            priors.append(for_loop)

        # add hyper priors for loc, scale, and covariate weights
        priors += self.genstmt_hyper_prior()
 
        return priors

    def genstmt_data_simulator(self) -> list[sl.Decl]:
        decls = [sl.Decl(x) for x in [self._loc, self._scale] + self._cws]
        if self._level is not None and self._level_type == "random":
            decls.append(sl.Decl(self._level_scale))
        return self.genstmt_data() + decls

    def genstmt_gq_simulator(
        self, transform_func: Callable = lambda x: x
    ) -> list[sl.Stmt]:
        """
        Generate an AST for the GQ block of the simulator corresponding to
        this parameter. The AST represents sampling of the random parameter
        from a Stan `x_rng` function (where x is the distribution). The loc
        parameter includes covariate weights. The transform_func is used
        to modify the parameter and the covariate. For instance to index both
        to get unit-specific values.

        Parameters
        ----------
        transform_func : Callable, optional
            function to transform the parameter and covariate.
            The default is lambda x: x.

        Returns
        -------
        list
            a list of Stan statements (sl.Stmt).

        """
        R = sl.intVar("R")
        loc_terms = []
        match len(self._catcovs):
            case 0:
                loc_terms.append(self._loc)
            case 1:
                cov = transform_func(self._catcovs[0].var)
                loc_terms.append(self._loc.idx(cov))
            case _:
                raise NotImplementedError(
                    "parameters can currently only have a single categorical covariate"
                )
        for cov, cw in zip(self._contcovs, self._cws):
            # we have to transpose vector-valued covariate weights
            cwt = cw if cov.dim is None else sl.TransposeOp(cw)
            loc_terms.append(cwt * transform_func(cov.var))
        loc = reduce(lambda a, b: a + b, loc_terms)
        # choose a function to cast the sample to the right shape / type
        shape_transform = lambda x: x
        match self.space:
            case "real":
                pass  # don't change anything
            case "vector":
                shape_transform = lambda x: sl.Call("to_vector", x)
            case "matrix":
                # TODO: test that this works!! or do we have to provide vector arguments to _rng functions?
                n, m = (self.shape.idx(sl.LiteralInt(i)) for i in (1, 2))
                shape_transform = lambda x: sl.Call("to_matrix", [x, n, m])

        # special case for hierarchical parameters: sample all once
        # NB: this is only implemented for non-correlated and scalar parameters
        # FIXME: this will not work with covariates!
        if self.level is not None and self.level_type == "random":
            # choose the correct sampling distribution and transformation
            if len(self._catcovs) > 0:
                # loc already has the right dimension, but needs to be a vector
                mu = sl.Call("to_vector", loc)
            else:  # just repeat the shared loc parameter
                mu = sl.Call("rep_vector", [self.loc, R])
            level_mat = sl.Var(sl.Matrix(R, R), f"level_{self.level.name}")
            eye = sl.Call("identity_matrix", R)
            sigma = (
                sl.square(self.scale) * eye + sl.square(self.level_scale) * level_mat
            )
            # translate the distribution to a multivariate version
            match self._distribution:
                case "normal":
                    sam_transform = lambda x: x
                case "lognormal":
                    sam_transform = lambda x: sl.Call("exp", x)
                case "logitnormal":  ## TODO: TESTING
                    l, u = sl.LiteralReal(self._lbound), sl.LiteralReal(self._ubound)
                    sam_transform = (
                        lambda x: sl.Call("inv_logit", x) * sl.Par(u - l) + l
                    )
                case _:
                    raise NotImplementedError(
                        "random effects distribution not implemented for random level"
                    )

            rng_stmt = sl.Assign(
                self._var,  # we don't need an indexing transform, as we're sampling all at once
                sl.Call(
                    "to_array_1d",
                    sam_transform(sl.Call("multi_normal_rng", [mu, sigma])),
                ),  # to_array_1d
            )
            return [rng_stmt]

        # else, the parameter is not hierarchical.
        rng_params = [loc, self._scale]
        if self._distribution == "logitnormal":
            # the logitnormal distribution requires more parameters
            rng_params += [sl.LiteralReal(self._lbound), sl.LiteralReal(self._ubound)]

        rng_stmt = sl.Assign(
            transform_func(self._var),
            shape_transform(sl.Call(self._distribution + "_rng", rng_params)),
        )
        return [rng_stmt]
    
    def genstmt_prior_samples(self) -> List[sl.Stmt]:
        loc_rng_expr = self._loc_prior.gen_rng_expr()
        scale_rng_expr = self._scale_prior.gen_rng_expr()
        rng_params = [loc_rng_expr, scale_rng_expr]
        match self._distribution:
            case "logitnormal":
                rng_params += [sl.LiteralReal(self._lbound), sl.LiteralReal(self._ubound)]
            case _:
                pass
        rng_expr = sl.Call(self._distribution + "_rng", rng_params)
        # create declaration and assignment statement
        # the variable name is `prior_sample_{parameter_name}`
        return [sl.DeclAssign(sl.Var(self.var.var_type, f"prior_sample_{self._name}"), rng_expr)]



class ConstParameter(Parameter):
    def get_type(this) -> str:
        return "const"

    def expand_and_index_var(self, apply_idx=False) -> sl.Expr:
        return self.var

    def genstmt_data(self) -> list[sl.Decl]:
        decls: list[sl.Decl] = []
        match self.space:
            case "real":
                pass  # real values have no shape: keep it simple
            case "vector" | "matrix":
                decls.append(sl.Decl(self._shape))
            case _:
                raise Exception(f"invalid parameter space: {self.space}")
        decls.append(sl.Decl(self._var))
        return decls

    def genstmt_data_simulator(self) -> list[sl.Decl]:
        return self.genstmt_data()


class ConstIndivParameter(Parameter):
    def get_type(this) -> str:
        return "const_indiv"

    def expand_and_index_var(self, apply_idx=False) -> sl.Expr:
        R, r = sl.intVar("R"), sl.intVar("r")
        if apply_idx:
            return sl.expandVar(self.var, (R,)).idx(r)
        return sl.expandVar(self.var, (R,))

    def genstmt_data(self) -> list[sl.Decl]:
        decls: list[sl.Decl] = []
        match self.space:
            case "real":
                pass  # real values have no shape: keep it simple
            case "vector" | "matrix":
                decls.append(sl.Decl(self._shape))
            case _:
                raise Exception(f"invalid parameter space: {self.space}")

        R = sl.Var(sl.Int(), "R")
        array = sl.expandVar(self.var, (R,))
        decls.append(sl.Decl(array))
        return decls

    def genstmt_data_simulator(self) -> list[sl.Decl]:
        return self.genstmt_data()


class IndivParameter(Parameter):
    def __init__(
        self,
        name: str,
        value: float | list[float],
        covariates: Optional[list[Covariate]] = None,
        lbound: Optional[float] = 0.0,
        ubound: Optional[float] = None,
        space: ParamSpace = "real",
        prior: Optional[Prior] = None,
    ) -> None:
        super().__init__(
            name,
            value,
            covariates=covariates,
            lbound=lbound,
            ubound=ubound,
            space=space,
        )
        if prior is None:  # default prior
            match self._space:
                case "real" | "vector":
                    # default student-t will work fine for scalars and vectors
                    self._prior = Prior("student_t", [3.0, 0.0, 2.5])
                case "matrix":
                    # but matrix variables are not allowed. Add a transform!
                    self._prior = Prior(
                        "student_t",
                        [3.0, 0.0, 2.5],
                        transform=lambda x: sl.Call("to_vector", x),
                    )
                case _:
                    raise NotImplementedError(
                        "default priors for space '{self._space}' are not yet implemented"
                    )

        else:  # user-defined prior
            self._prior = prior

    @property
    def level(self) -> Optional[CatCovariate]:
        return None

    @property
    def level_type(self) -> LevelType:
        return "fixed"
    
    @property
    def prior_name(self) -> str:
        return self._prior.name

    def get_type(self) -> str:
        return "indiv"

    def expand_and_index_var(self, apply_idx=False) -> sl.Expr:
        R, r = sl.intVar("R"), sl.intVar("r")
        if apply_idx:
            return sl.expandVar(self.var, (R,)).idx(r)
        return sl.expandVar(self.var, (R,))

    def genstmt_data(self) -> list[sl.Stmt]:
        decls: list[sl.Decl] = []
        match self.space:
            case "real":
                pass  # real values have no shape: keep it simple
            case "vector" | "matrix":
                decls.append(sl.Decl(self._shape))
            case _:
                raise Exception(f"invalid parameter space: {self.space}")
        return decls

    def genstmt_model(self) -> list[sl.Stmt]:
        """
        Single fixed prior for all indiv parameters.
        If the parameter space is not real, we have to sample in a for loop
        per unit.
        """
        match self.space:
            case "real":
                stmt_list = self._prior.gen_sampling_stmt(self.var)
            case "vector" | "matrix":
                R, r = sl.intVar("R"), sl.intVar("r")
                prior_r = self._prior.gen_sampling_stmt(self.var.idx(r))
                if len(prior_r) == 1:
                    for_loop = sl.ForLoop(r, sl.Range(sl.one(), R), prior_r[0])
                elif len(prior_r) > 1:
                    for_loop = sl.ForLoop(r, sl.Range(sl.one(), R), sl.Scope(prior_r))
                else:
                    raise Exception(
                        "expected Prior.gen_sampling_stmt to return one or more statements"
                    )
                stmt_list = [for_loop]
        return stmt_list

    def genstmt_params(self) -> list[sl.Decl]:
        R = sl.Var(sl.Int(), "R")
        array = sl.expandVar(self.var, (R,))
        decl_list = [sl.Decl(array)]
        return decl_list

    def genstmt_data_simulator(self) -> list[sl.Decl]:
        return self.genstmt_data() + self.genstmt_params()
    

    def genstmt_prior_samples(self) -> List[sl.Stmt]:
        rng_expr = self._prior.gen_rng_expr()
        # create declaration and assignment statement
        # this is done in the same way as fixed parameters.
        return [sl.DeclAssign(sl.Var(self.var.var_type, f"prior_sample_{self._name}"), rng_expr)]


## test functions for checking that correlations are well defined


def params_in_multiple_sets(corr_sets: list[list[str]]) -> list[str]:
    res: list[str] = []
    ps = util.unique(util.flatten(corr_sets))
    for p in ps:
        num_sets = len([corr_set for corr_set in corr_sets if p in corr_set])
        if num_sets != 1:
            res.append(str(p))
    return res


def corr_params_undef(
    params: Sequence[Parameter], corr_sets: List[List[str]]
) -> list[str]:
    res: list[str] = []
    par_type_dict = {p.name: type(p) for p in params}
    ps = util.unique(util.flatten(corr_sets))
    for p in ps:
        if p not in par_type_dict or par_type_dict[p] is not RandomParameter:
            res.append(p)
    return res


def corr_sets_too_small(corr_sets: list[list[str]]) -> list[list[str]]:
    res = []
    for corr_set in corr_sets:
        if len(util.unique(corr_set)) < 2:
            res.append(corr_set)
    return res


class ParameterBlock(Parameter):
    """
    Note that the value attribute of ParameterBlock consists of
    unconstraint values!
    """

    def __init__(
        self, params: list[RandomParameter], corr_value: np.ndarray, intensity: float
    ) -> None:
        # check that all parameters are of the same type
        if not all(isinstance(p, RandomParameter) for p in params):
            raise TypeError(
                "ParameterBlock can only be created from a list of RandomParameters"
            )

        # check the levels of the parameters
        levels = [p.level for p in params]
        self._level = levels[0]
        if not all(self._level == level for level in levels):
            raise ValueError(
                "All parameters in a ParameterBlock must have the same level"
            )

        self._params = params
        self._name = "_".join([p.name for p in params])
        self.init_value() # set the _value attribute. uses self._params

        self._lbound = np.array(
            [p.lbound if p.lbound is not None else -np.inf for p in params]
        )
        self._ubound = np.array(
            [p.ubound if p.ubound is not None else np.inf for p in params]
        )

        self._contcovs = util.unique([cn for p in params for cn in p._contcovs])

        self._catcovs = util.unique([cn for p in params for cn in p._catcovs])
        if len(self._catcovs) > 0:
            raise NotImplementedError(
                "correlated parameters can currently not have categorical covariates."
            )

        self._scale_value = np.array([p.scale_value for p in params])

        self._noncentered = [p.noncentered for p in params]

        n = sl.LiteralInt(len(params))
        self._chol = sl.Var(sl.CholeskyFactorCorr(n), f"chol_{self._name}")
        self._corr_value = corr_value
        self._intensity = intensity

    def __contains__(self, item: Parameter) -> bool:
        return any([item.name == p.name for p in self._params])

    def __len__(self) -> int:
        return len(self._params)

    def init_value(self) -> None:
        """
        Set the value attribute of the parameter block.
        This uses the values of the parameters in the block.
        The values are transformed to an unconstrained space.

        FIXME: make sure this works for parameters for which we 
        specified individual initial values.
        """
        init_defined_per_unit = [isinstance(p.value, list) for p in self._params]
        num_pars_per_elt = [len(p.value) if x else 1 for p, x in zip(self._params, init_defined_per_unit)]
        max_num_pars = np.max(num_pars_per_elt)
        
        if not all(n == max_num_pars or n == 1 for n in num_pars_per_elt):
            raise ValueError(
                "All parameters in a ParameterBlock must have the same number of elements "
                "or contain only one element per parameter."
            )

        def fun(p):
            x = p.value
            if isinstance(x, list):
                return [constr_to_unconstr_float(xi, p.lbound, p.ubound) for xi in x]
            else:
                return constr_to_unconstr_float(x, p.lbound, p.ubound)
            
        unc_values = [fun(p) for p in self._params]
        
        self._value = unc_values


    @property
    def scale_value(self) -> np.array:
        return self._scale_value

    @property
    def corr_value(self) -> np.array:
        return self._corr_value

    @property
    def noncentered(self) -> list[bool]:
        return self._noncentered

    @property
    def level(self) -> Optional[CatCovariate]:
        return self._level

    def get_type(self) -> str:
        return "block"

    def expand_and_index_var(self, apply_idx=False) -> sl.Expr:
        raise NotImplementedError(
            "Expanding and indexing parameter block variables curently not implemented"
        )

    def genstmt_model(self) -> list[sl.Stmt]:
        ## TODO: how to incorporate categorical covariates here???
        n = sl.LiteralInt(len(self._params))
        R = sl.Var(sl.Int(), "R")
        r = sl.Var(sl.Int(), "r")
        blockVar = sl.Var(sl.Array(sl.Vector(n), (R,)), f"block_{self._name}")
        locVar: sl.Expr = sl.Var(sl.Vector(n), f"loc_{self._name}")
        scaleVar = sl.Var(sl.Vector(n, lower=sl.rzero()), f"scale_{self.name}")
        L = sl.Call("diag_pre_multiply", [scaleVar, self._chol])
        covVars = {cov.name: sl.expandVar(cov.var, (R,)) for cov in self._contcovs}

        weightVars: dict[str, sl.Expr] = {}
        for cov in self._contcovs:
            if cov.dim is None:
                wvar = sl.expandVecVar(cov.weight_var(self._name), n)
            else:
                wvart = sl.expandVecVar(cov.weight_var(self._name), n)
                wvar = sl.TransposeOp(wvart)
            weightVars[cov.name] = wvar

        priors: list[sl.Stmt] = []

        if len(self._contcovs) == 0:
            priors = [
                sl.Sample(blockVar, sl.Call("multi_normal_cholesky", [locVar, L]))
            ]
        else:
            loc_terms = [locVar] + [
                weightVars[cov.name] * sl.IndexOp(covVars[cov.name], r)
                for cov in self._contcovs
            ]
            loc_expr = reduce(lambda a, b: a + b, loc_terms)

            priors = [
                sl.ForLoop(
                    r,
                    sl.Range(sl.one(), R),
                    sl.Sample(
                        blockVar.idx(r),
                        sl.Call("multi_normal_cholesky", [loc_expr, L]),
                    ),
                )
            ]

        # priors for location, scale and correlation matrix
        lkj_prior = Prior("lkj_corr_cholesky", [self._intensity])
        for p in self._params:
            priors += p.genstmt_hyper_prior()

        priors += lkj_prior.gen_sampling_stmt(self._chol)
        # return all prior expressions
        return priors

    def genstmt_params(self) -> list[sl.Decl]:
        n = sl.LiteralInt(len(self._params))
        R = sl.Var(sl.Int(), "R")
        one_par_per_unit = self.level is None
        num_pars = R if one_par_per_unit else self.level.num_cat_var

        # declare loc and scale parameters
        loc_decls = sl.gen_decl_lists([p.loc for p in self._params])
        scale_decls = sl.gen_decl_lists([p.scale for p in self._params])

        decls = [
            sl.Decl(sl.Var(sl.Array(sl.Vector(n), (num_pars,)), f"block_{self._name}")),
            sl.Decl(sl.Var(sl.CholeskyFactorCorr(n), f"chol_{self._name}")),
        ]

        # declare weights for any covariates
        decl_cws = [
            sl.Decl(cov.weight_var(p.name)) for p in self._params for cov in p._contcovs
        ]
        return loc_decls + scale_decls + decls + decl_cws

    def genstmt_trans_params(self) -> list[sl.Stmt]:
        n = sl.LiteralInt(len(self._params))
        R = sl.Var(sl.Int(), "R")
        one_par_per_unit = self.level is None
        num_pars = R if one_par_per_unit else self.level.num_cat_var

        trans_decls: list[sl.Stmt] = []

        # transform unconstrained blocks to correct domain and define parameter names
        blockVar = sl.Var(sl.Array(sl.Vector(n), (num_pars,)), f"block_{self._name}")
        vals = [
            sl.MultiIndexOp(blockVar, [sl.fullRange(), sl.LiteralInt(i + 1)])
            for i, _ in enumerate(self._params)
        ]

        # transform the values to the correct domain. For non-centered parameters,
        # we apply an additional linear transformation to the values.
        # the transformation is defined by the loc and scale parameters.

        trans_vals = [
            domain_transform(
                val, p.lbound, p.ubound, array=True, 
                loc=p.loc if p.noncentered else None,
                scale=p.scale if p.noncentered else None
            )
            for val, p in zip(vals, self._params)
        ]
        trans_decls += [
            sl.DeclAssign(sl.expandVar(p.var, (num_pars,)), tr_val)
            for p, tr_val in zip(self._params, trans_vals)
        ]

        # define loc and scale parameters

        loc = sl.Var(sl.Vector(n), f"loc_{self._name}")
        scale = sl.Var(sl.Vector(n, lower=sl.rzero()), f"scale_{self._name}")

        # the loc and scale vectors use the loc and scales from the parameters.
        # if the parameter is non-centered, the loc is zero (and we apply a linear transform later)
        # for the scales, we use 1.0 for non-centered parameters.
        loc_decl = sl.DeclAssign(
            loc, sl.LiteralVector([p.loc if not p.noncentered else sl.rzero() for p in self._params])
        )
        real_one = sl.LiteralReal(1.0)
        scale_decl = sl.DeclAssign(
            scale, sl.LiteralVector([p.scale if not p.noncentered else real_one for p in self._params])
        )

        trans_decls += [
            sl.comment(f"loc and scale vectors for correlated parameter block {self._name}"),
            loc_decl, scale_decl
        ]

        # define weight vectors if there are any covariates
        # group parameters according to covariates, insert zero weights for other parameters
        def zero_weight(c: ContCovariate):
            if c.dim is None:
                return sl.rzero()
            d = sl.LiteralInt(c.dim)
            return sl.Call("zeros_vector", [d])

        cov_par_dict = {
            cov.name: [
                cov.weight_var(p.name) if cov in p._contcovs else zero_weight(cov)
                for p in self._params
            ]
            for cov in self._contcovs
        }

        # weight vectors for scalar covariates
        weight_vecs = []
        for cov in self._contcovs:
            if cov.dim is None:
                vec_decl = sl.DeclAssign(
                    sl.expandVecVar(cov.weight_var(self._name), n),
                    sl.LiteralVector(cov_par_dict[cov.name]),
                )
            else:
                d = sl.LiteralInt(cov.dim)
                vec_decl = sl.DeclAssign(
                    sl.expandVecVar(cov.weight_var(self._name), n),
                    sl.LiteralVector(
                        [sl.TransposeOp(x) for x in cov_par_dict[cov.name]]
                    ),
                )
            weight_vecs.append(vec_decl)

        trans_decls.append(
            sl.comment(f"covariate weight vectors for correlated parameter block {self._name}")
        )
        trans_decls += weight_vecs

        return trans_decls

    def genstmt_gq(self) -> list[sl.Stmt]:
        n = sl.LiteralInt(len(self._params))
        corrmat_decl = sl.DeclAssign(
            sl.Var(sl.Matrix(n, n), f"corr_{self._name}"),
            sl.Call("tcrossprod", [self._chol]),
        )
        return [corrmat_decl]

    def genstmt_data_simulator(self) -> list[sl.Decl]:
        n = sl.LiteralInt(len(self._params))
        corr = sl.Var(sl.Matrix(n, n), f"corr_{self._name}")
        return [sl.Decl(corr)]

    def genstmt_gq_simulator(
        self, transform_func: Callable = lambda x: x
    ) -> list[sl.Stmt]:
        stmts: list[sl.Stmt] = []

        # sample from unconstraint multivariate distribution
        n = sl.LiteralInt(len(self._params))
        blockVar = sl.Var(sl.Vector(n), f"block_{self._name}")
        loc = sl.Var(sl.Vector(n), f"loc_{self._name}")
        scale = sl.Var(sl.Matrix(n, n), f"scale_{self._name}")
        loc_terms_elts = [[p.loc] for p in self._params]
        for p, loc_terms in zip(self._params, loc_terms_elts):
            # add weighted covariates
            for cov, cw in zip(p._contcovs, p._cws):
                cwt = sl.TransposeOp(cw) if cov.dim is not None else cw
                loc_terms.append(cwt * transform_func(cov.var))
        loc_elts = [
            reduce(lambda a, b: a + b, loc_terms) for loc_terms in loc_terms_elts
        ]
        scale_elts = [p.scale for p in self._params]
        loc_decl = sl.DeclAssign(loc, sl.LiteralVector(loc_elts))
        stmts.append(loc_decl)
        scale_decl = sl.DeclAssign(
            scale, sl.Call("diag_matrix", [sl.LiteralVector(scale_elts)])
        )
        stmts.append(scale_decl)

        corr = sl.Var(sl.Matrix(n, n), f"corr_{self._name}")
        cov = sl.Var(sl.Matrix(n, n), f"cov_{self._name}")
        cov_decl = sl.DeclAssign(cov, scale * corr * scale)

        stmts.append(cov_decl)

        multivar_sam = sl.Call("multi_normal_rng", [loc, cov])
        stmts.append(sl.DeclAssign(blockVar, multivar_sam))

        # unpack and transform unconstrained sample
        for i, p in enumerate(self._params):
            unpack_stmt = sl.Assign(
                transform_func(p.var),
                domain_transform(
                    blockVar.idx(sl.LiteralInt(i + 1)), p.lbound, p.ubound
                ),
            )
            stmts.append(unpack_stmt)

        # we have to wrap this in a Scope to allow for a declaration
        return [sl.Scope(stmts)]
    

    def genstmt_prior_samples(self) -> List[sl.Stmt]:
        ## FIXME (?) : we currently ignore correlations when generating prior samples

        from hboms.logger import logger

        logger.warning(
            "Prior samples for correlated parameters are generated independently"
        )

        stmts = []

        for p in self._params:
            stmts += p.genstmt_prior_samples()

        return stmts


def gen_corr_block(corr: Correlation, params: List[Parameter]) -> ParameterBlock:
    """
    Generate a ParameterBlock from a Correlation object and a list of parameters.
    Raises an exception if the correlation contains parameters that are not in the list of parameters.
    """
    ps = [p for p in params if p.name in corr.parnames]
    if len(ps) != len(corr.parnames):
        missing_params = sorted(list(set(corr.parnames) - set(p.name for p in ps)))
        missing_params = ", ".join(missing_params)    
        message = (
            f"Correlation contains parameter names that are not in the model: {missing_params}"
        )
        raise ValueError(message)
    
    # otherwise, we can create the ParameterBlock
    return ParameterBlock(ps, corr.value, corr.intensity)


class TransParameter(Parameter):
    def __init__(
        self, 
        name: str, 
        value: str,
        dependencies: list[Parameter] | None = None,
        rank: int = 0,
        lbound: float | None = 0.0, 
        ubound: float | None = None, 
        space: ParamSpace = "real"
    ) -> None:
        super().__init__(name, value, lbound=lbound, ubound=ubound, space=space)
        self._dependencies = [] if dependencies is None else dependencies
        self._rank = rank

    def get_type(self) -> str:
        return "trans"
    
    def expand_and_index_var(self, apply_idx=False) -> sl.Expr:
        return self.var

    def get_rank(self) -> int:
        """
        The rank is assigned during the "compilation" stage of the HBOMS model.
        As transformed parameters can depend on other transformed parameters, we have to 
        take special care that the values are calculated in the right order.
        The rank determines this order, and is calculated using the topological order
        of the dependency graph.
        """
        return self._rank

    def genstmt_trans_params(self) -> List[sl.Stmt]:
        """
        Declare and define a transformed parameter.
        """
        args = [p.var for p in self._dependencies]
        func_name = f"{self.name}_transform"
        func_call = sl.Call(func_name, args)
        decl = sl.DeclAssign(self.var, func_call)
        return [decl]
    
    def genstmt_functions(self) -> List[sl.Stmt]:
        """
        We make use of a function to define the transformation
        """
        par_args = [p.var for p in self._dependencies]
        trans_stmt = sl.Return(sl.MixinExpr(self.value))
        func_body: list[sl.Stmt] = [
            sl.EmptyStmt(comment="user-defined parameter transform"),
            trans_stmt
        ]

        trans_func = sl.FuncDef(
            self.var.var_type,  ## function's return type
            f"{self.name}_transform",  ## function's name
            par_args,  ## function's arguments
            func_body,  ## function's body
        )
        return [trans_func]

    def genstmt_gq_simulator(self, transform_func: Callable = lambda x: x) -> list[sl.Stmt]:
        return self.genstmt_trans_params()
        


class TransIndivParameter(TransParameter):
    def get_type(self) -> str:
        return "trans_indiv"

    def expand_and_index_var(self, apply_idx=False) -> sl.Expr:
        R, r = sl.intVar("R"), sl.intVar("r")
        if apply_idx:
            return sl.expandVar(self.var, (R,)).idx(r)
        return sl.expandVar(self.var, (R,))

    @property
    def level(self) -> Optional[CatCovariate]:
        return None

    @property
    def level_type(self) -> LevelType:
        return "fixed"

    def genstmt_trans_params(self) -> List[sl.Stmt]:
        """
        Declare and define a transformed parameter.
        TODO: parse user code to generate a definition. This may depend on
        other parameters, which determines if it is a individual or population 
        parameter.
        """
        R, r = sl.intVar("R"), sl.intVar("r")
        expanded_var = self.expand_and_index_var(apply_idx=False)
        indexed_var = self.expand_and_index_var(apply_idx=True)
        args = [p.expand_and_index_var(apply_idx=True) for p in self._dependencies]
        func_name = f"{self.name}_transform"
        func_call = sl.Call(func_name, args)
        decl = sl.Decl(expanded_var)
        assign = sl.Assign(indexed_var, func_call)
        loop = sl.ForLoop(r, sl.Range(sl.one(), R), assign)
        return [decl, loop]

    def genstmt_gq_simulator(self, transform_func: Callable = lambda x: x) -> List[sl.Stmt]:
        indexed_var = self.expand_and_index_var(apply_idx=True)
        args = [p.expand_and_index_var(apply_idx=True) for p in self._dependencies]
        func_name = f"{self.name}_transform"
        func_call = sl.Call(func_name, args)
        assign = sl.Assign(indexed_var, func_call)
        return [assign]
    

def param_dispatch(name: str, value: float, par_type: str, **kwargs) -> Parameter:
    match par_type:
        case "const":
            return ConstParameter(name, value, **kwargs)
        case "const_indiv":
            return ConstIndivParameter(name, value, **kwargs)
        case "fixed":
            return FixedParameter(name, value, **kwargs)
        case "random":
            return RandomParameter(name, value, **kwargs)
        case "indiv":
            return IndivParameter(name, value, **kwargs)
        case "trans":
            # If all dependencies are population-level parameters, then the 
            # transformed parameter is also population-level: use TransParameter.
            # If at least one of the dependencies in individual-level, then we have to
            # use TransIndivParameter, which is also individual-level.
            deps = kwargs["dependencies"]
            if all([p.is_pop_level() for p in deps]):
                return TransParameter(name, value, **kwargs)
            return TransIndivParameter(name, value, **kwargs)
        case _:
            raise Exception(f"invalid parameter type '{par_type}'")
