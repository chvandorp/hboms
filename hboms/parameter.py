import functools
from typing import List, Optional, Sequence, Callable, Literal
from abc import ABC, abstractmethod
import numpy as np
from functools import reduce
import copy

from . import utilities as util
from . import stanlang as sl
from .correlation import Correlation
from .covariate import Covariate, ContCovariate, CatCovariate, group_covars
from .prior import Prior, AbsContPrior, DiracDeltaPrior
from .transform import constr_to_unconstr_float
from .transform_expr import domain_transform, inverse_domain_transform


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
    
    def genstmt_dirac_delta_priors(self) -> list[sl.Stmt]:
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
                    self._prior = AbsContPrior("student_t", [3.0, 0.0, 2.5])
                case "matrix":
                    # but matrix variables are not allowed. Add a transform!
                    self._prior = AbsContPrior(
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
            return sl.expandVar(self.var, (num_params,)).idx(idx)
        return sl.expandVar(self.var, (num_params,))

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



class RandomParameter(Parameter):
    def _check_nc_compatibility(self):
        # non-centered parameters are still very limited
        if self._noncentered:
            if self.space != "real":
                raise NotImplementedError(
                    "non-centered parameterization is only implemented for scalars"
                )
            if self._contcovs:
                raise NotImplementedError(
                    "non-centered parameterization is not implemented for continuous covariates"
                )

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

        def_hyp_prior = AbsContPrior("student_t", [3.0, 0.0, 2.5])
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

        # centered or non-centered parameterization
        self._noncentered = noncentered

        # make sure the non-centered setting is compatible with the parameter settings
        self._check_nc_compatibility()

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
    def level_scale(self) -> Optional[sl.Var]:
        return self._level_scale

    @property
    def level_scale_value(self) -> float:
        return self._level_scale_value

    @property
    def noncentered(self) -> bool:
        return self._noncentered
    
    @noncentered.setter
    def noncentered(self, x: bool) -> None:
        self._noncentered = x

        # verify compatibility
        self._check_nc_compatibility()

    @property
    def loc_prior_name(self) -> str:
        return self._loc_prior.name
    
    @property
    def scale_prior_name(self) -> str:
        return self._scale_prior.name

    @property
    def rand_var(self) -> sl.Var:
        # remove bounds from the stan type
        unres_stan_type = copy.deepcopy(self._stan_type)
        unres_stan_type.lower, unres_stan_type.upper = None, None
        return sl.Var(unres_stan_type, f"rand_{self.name}")

    def get_type(this) -> str:
        return "random"
    
    def num_params(self) -> sl.Expr:
        """
        Get the number of random parameters based on the level type.
        This is equal to the number of units R if the level is None or random,
        and equal to the number of categories of the level otherwise.

        Returns
        -------
        sl.Expr
            expression for the number of random parameters.
        """
        R = sl.intVar("R")
        if self.level is None or self.level_type == "random":
            return R
        return self.level.num_cat_var

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
        # by default, each unit has it's own random effect,
        # but the random effect can also be specified on a group level
        # which is defined by a categorical covariate
        num_pars = self.num_params()
        # expand the parameter to the right shape
        # choose between centered and non-centered parameterization
        if self._noncentered:
            array = sl.expandVar(self.rand_var, (num_pars,)) # "rand_parname"
        else:
            array = sl.expandVar(self.var, (num_pars,)) # just "parname"
        free_pars = [array]
        # check that loc and scale don't have dirac delta priors
        if not isinstance(self._loc_prior, DiracDeltaPrior):
            free_pars.append(self._loc)
        if not isinstance(self._scale_prior, DiracDeltaPrior):
            free_pars.append(self._scale)
        # declare an additional scale if the parameter has additional group-specific random effects
        if self.level is not None and self.level_type == "random":
            if not isinstance(self._level_scale_prior, DiracDeltaPrior):
                free_pars.append(self._level_scale)
        # create declarations for the free parameters
        decls = [sl.Decl(x) for x in free_pars + self._cws]
        return decls

    def genstmt_trans_params(self) -> List[sl.Stmt]:
        R, r = sl.intVar("R"), sl.intVar("r")
        stmts: list[sl.Stmt] = []

        # if the parameter is non-centered, we have to transform it
        # otherwise we can just sample the parameter directly and don't need to transform
        if not self._noncentered:
            return stmts
        # else, we need to do extra work: transform the random effects
        if self.level is not None and self.level_type == "random":
            # build the covariance matrix for the random level effects
            cov_mat1 = sl.square(self.scale) * sl.Call("identity_matrix", sl.intVar("R"))
            cov_mat2 = sl.square(self.level_scale) * self.level.level_matrix_var
            # define the cholesky factor 
            chol_fact = sl.Call("cholesky_decompose", cov_mat1 + cov_mat2)
            loc = self._loc
            if self._catcovs:
                cov = self._catcovs[0]
                loc_array = sl.expandVar(loc, (cov.num_cat_var,))
                cov_var = sl.expandVar(cov.var, (R,))
                loc_array_indexed = loc_array.idx(cov_var)
                loc = sl.Call("to_vector", loc_array_indexed)
            
            # use domain transform
            trans_par = domain_transform(
                self.rand_var, self._lbound, self._ubound,
                array=True, loc=loc, scale=chol_fact
            )
            # declare the transformed parameter
            par_array = sl.expandVar(self.var, (R,))
            par_decl = sl.DeclAssign(par_array, trans_par)
            return [par_decl]
        # else...
        num_pars = self.num_params()
        array = sl.expandVar(self.var, (num_pars,))
        rand_effect_array = sl.expandVar(self.rand_var, (num_pars,))
        # allow for categorical covariates by selecting the correct loc value
        loc = self._loc 
        if self._catcovs:
            catcov = self._catcovs[0]
            if self.level is not None and self.level_type == "fixed":
                # in this case, the loc array is defined on the level
                # so we have to index it with the restricted categories
                restricted_catcov_var = catcov.restricted_var(self.level)
                catcov_var_array = sl.expandVar(restricted_catcov_var, (num_pars,))
                cat = catcov_var_array.idx(r) # the category for level r
            else:
                # otherwise, index with the unit-level categories
                catcov_var_array = sl.expandVar(catcov.var, (num_pars,))
                cat = catcov_var_array.idx(r) # the category for unit r
            loc = loc.idx(cat) # the loc for that category
        scale = self._scale
        # linear transformation of the random effect
        lin_trans_rand_effect = loc + scale * rand_effect_array.idx(r)
        decl = sl.Decl(array)
        stmts.append(decl)
        # use a for loop to fill the array with transformed values
        for_loop = sl.ForLoop(
            r,
            sl.Range(sl.one(), num_pars),
            sl.Assign(
                array.idx(r),
                domain_transform(lin_trans_rand_effect, self._lbound, self._ubound),
            ),
        )
        stmts.append(for_loop)
        return stmts


    def genstmt_dirac_delta_priors(self) -> list[sl.Stmt]:
        """
        This methods makes sure that Diract Delta priors are implemented 
        for the hyper parameters: e.g. loc, scale, level_scale.
        This method should be called during creation of the transformed parameters
        block.

        Returns
        -------
        list[sl.Stmt]
            list of statements implementing the Dirac Delta priors.

        """
        stmts: list[sl.Stmt] = []
        if isinstance(self._loc_prior, DiracDeltaPrior):
            stmts += self._loc_prior.gen_defining_stmt(self._loc)
        if isinstance(self._scale_prior, DiracDeltaPrior):
            stmts += self._scale_prior.gen_defining_stmt(self._scale)
        if self.level is not None and self.level_type == "random":
            if isinstance(self._level_scale_prior, DiracDeltaPrior):
                stmts += self._level_scale_prior.gen_defining_stmt(self._level_scale)
        return stmts


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
        hyper_prior = AbsContPrior("student_t", [3.0, 0.0, 2.5])
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
            # in case of level-specific random effects, add prior for level scale
            if self.level is not None and self.level_type == "random":
                priors += self._level_scale_prior.gen_sampling_stmt(self._level_scale)
            return priors

        # select the correct loc from the loc array in the case of categorical covariates
        if len(self._contcovs) == 0:
            if len(self._catcovs) == 0:
                loc = self._loc
            else:
                catcov = self._catcovs[0]
                if self.level is not None and self.level_type == "fixed":
                    # in this case, the loc array is defined on the level
                    # so we have to index it with the restricted categories
                    restricted_catcov_var = catcov.restricted_var(self.level)
                    num_pars = self.level.num_cat_var
                    catcov_var_array = sl.expandVar(restricted_catcov_var, (num_pars,))
                else:
                    catcov_var_array = sl.expandVar(catcov.var, (R,))
                loc = self._loc.idx(catcov_var_array)

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

    def genstmt_gq(self) -> list[sl.Stmt]:
        """
        Generate an AST for the GQ block corresponding to this parameter.
        If the parameter has a centered parameterization, the GQ block
        will include the "rand_<parname>" variable declaration and definition.
        This is useful for e.g. finding correlationss between "random effects".
        In the non-centered case, there is no additional variable to declare,
        as the rand_<parname> is defined in the parameters block.
        """            

        if self._noncentered:
            return []
        # else...

        # SOME CASES ARE NOT IMPLEMENTED YET!
        if self.space != "real":
            from .logger import logger
            msg = (
                "random effect transformation in GQ block " + 
                f"not implemented parameter '{self.name}' " + 
                "because of non-scalar parameter space."
            )
            logger.warning(msg)
            return []

        if self._contcovs:
            from .logger import logger
            msg = (
                "random effect transformation in GQ block " + 
                f"not implemented for parameter '{self.name}' " +
                "because of continuous covariates."
            )
            logger.warning(msg)
            return []

        stmts: list[sl.Stmt] = []

        num_pars = self.num_params()
        rand_var = self.rand_var
        rand_array = sl.expandVar(rand_var, (num_pars,))
        var_array = sl.expandVar(self.var, (num_pars,))
        # if the parameter has categorical covariates, we have to select the correct loc value
        if len(self._catcovs) == 0:
            loc = self._loc
        else:
            cov = self._catcovs[0]
            num_cats = cov.num_cat_var
            if self.level is not None and self.level_type == "fixed":
                # in this case, the loc array is defined on the level
                # so we have to index it with the restricted categories
                restricted_catcov_var = cov.restricted_var(self.level)
                catcov_var_array = sl.expandVar(restricted_catcov_var, (num_pars,))
            else:
                catcov_var_array = sl.expandVar(cov.var, (num_pars,))
            loc_array = sl.expandVar(self._loc, (num_cats,))
            loc = sl.Call("to_vector", [loc_array.idx(catcov_var_array)])

        # if the parameter has a random level, we have to use both the scale and the level_scale
        if self.level is not None and self.level_type == "random":
            scale = sl.Call("sqrt", sl.square(self.scale) + sl.square(self.level_scale))
        else:
            scale = self._scale

        cent_val = inverse_domain_transform(
            var_array, self._lbound, self._ubound, array=True,
            loc=loc, scale=scale
        )

        stmts.append(sl.DeclAssign(rand_array, cent_val))

        return stmts



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
                if self.level is not None and self.level_type == "fixed":
                    # in this case, the loc array is defined on the level
                    # so we have to index it with the restricted categories
                    restricted_catcov_var = self._catcovs[0].restricted_var(self.level)
                    num_pars = self.level.num_cat_var
                    catcov_var_array = sl.expandVar(restricted_catcov_var, (num_pars,))
                else:
                    catcov_var_array = sl.expandVar(self._catcovs[0].var, (R,))
                # apply additional transform function
                cov = transform_func(catcov_var_array)
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
                    self._prior = AbsContPrior("student_t", [3.0, 0.0, 2.5])
                case "matrix":
                    # but matrix variables are not allowed. Add a transform!
                    self._prior = AbsContPrior(
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
                if len(prior_r) == 0:
                    raise ValueError("expected Prior.gen_sampling_stmt to return one or more statements")
                # add for loop around the prior statement(s)
                prior_r_stmt = prior_r[0] if len(prior_r) == 1 else sl.Scope(prior_r)
                for_loop = sl.ForLoop(r, sl.Range(sl.one(), R), prior_r_stmt)
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


def index_loc_for_block(
        p: RandomParameter, 
        num_pars: sl.Expr,
        fixed_level: CatCovariate | None = None, 
        index: sl.Expr | None = None
    ) -> sl.Expr:
    """
    Helper function to index the loc parameter of a random parameter block.
    This is needed because the loc parameter can depend on categorical covariates.

    Without cat covariates, loc is just p.loc.
    Otherwise, we have to index loc with the categorical covariate.
    This can depend on the level of the block.

    Parameters
    ----------
    p : RandomParameter
        the random parameter from which we take the loc parameter.
    num_pars : sl.Expr
        the number of parameters (i.e. R or number of level categories) 
        used for expansion
    fixed_level : CatCovariate | None, optional
        the fixed level of the block, by default None
    index : sl.Expr | None, optional
        the index to apply to the loc parameter, by default None

    Returns
    -------
    sl.Expr
        the indexed loc parameter. This is a scalar expression if index is provided,
        otherwise a vector expression.
    """
    if p._catcovs:
        catcov = p._catcovs[0]
        if fixed_level is not None:
            # in this case, the loc array is defined on the level
            # so we have to index it with the restricted categories
            restricted_catcov_var = catcov.restricted_var(fixed_level)
            catcov_var_array = sl.expandVar(restricted_catcov_var, (num_pars,))
        else:
            catcov_var_array = sl.expandVar(catcov.var, (num_pars,))
        loc_array = sl.expandVar(p.loc, (catcov.num_cat_var,))
        if index is not None:
            loc_val = loc_array.idx(catcov_var_array.idx(index))
            return loc_val
        else:
            loc_array = loc_array.idx(catcov_var_array)
            loc_vec = sl.Call("to_vector", loc_array)
            return loc_vec
    else:
        return p.loc



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
        
        if not all(p.level_type == "fixed" for p in params):
            raise NotImplementedError(
                "ParameterBlock can currently only be created from parameters with fixed levels"
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

        self._scale_value = np.array([p.scale_value for p in params])

        self._noncentered = [p.noncentered for p in params]

        n = sl.LiteralInt(len(params))
        self._chol = sl.Var(sl.CholeskyFactorCorr(n), f"chol_{self._name}")
        self._corr_value = corr_value
        self._intensity = intensity

        # the var is the vector of all parameters in the block
        self._var = sl.Var(sl.Vector(n), f"block_{self._name}")


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

    @property
    def level_type(self) -> LevelType:
        # currently the level_type has to be fixed for parameter blocks
        return "fixed"        

    def get_type(self) -> str:
        return "block"

    def expand_and_index_var(self, apply_idx=False) -> sl.Expr:
        raise NotImplementedError(
            "Expanding and indexing parameter block variables curently not implemented"
        )

    def genstmt_model(self) -> list[sl.Stmt]:
        n = sl.LiteralInt(len(self._params))
        R, r = sl.Var(sl.Int(), "R"), sl.Var(sl.Int(), "r")
        one_par_per_unit = self.level is None # note that random level is not supported here
        num_pars = R if one_par_per_unit else self.level.num_cat_var

        blockVar = sl.Var(sl.Array(sl.Vector(n), (num_pars,)), f"block_{self._name}")
        locVar: sl.Expr = sl.Var(sl.Vector(n), f"loc_{self._name}")
        if self._catcovs:
            # not critical, but the type of locVar is array if we have categorical covariates
            locVar = sl.expandVar(locVar, (num_pars,))
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

        if not self._contcovs:
            # no continuous covariates: we can sample all at once
            priors = [
                sl.Sample(blockVar, sl.Call("multi_normal_cholesky", [locVar, L]))
            ]
        else:
            if self._catcovs:
                # if we have categorical covariates, we have to index locVar
                locVar = locVar.idx(r)
                # otherwise, locVar is just a vector

            loc_terms = [locVar] + [
                weightVars[cov.name] * covVars[cov.name].idx(r)
                for cov in self._contcovs
            ]
            loc_expr = reduce(lambda a, b: a + b, loc_terms)

            priors = [
                sl.ForLoop(
                    r,
                    sl.Range(sl.one(), num_pars),
                    sl.Sample(
                        blockVar.idx(r),
                        sl.Call("multi_normal_cholesky", [loc_expr, L]),
                    ),
                )
            ]

        # priors for location, scale and correlation matrix
        lkj_prior = AbsContPrior("lkj_corr_cholesky", [self._intensity])
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
        # don't declare loc or scale if they have a DiracDelta prior
        def has_ddp(p: RandomParameter, attr: str) -> bool:
            return isinstance(getattr(p, f"_{attr}_prior"), DiracDeltaPrior)
        loc_decls = sl.gen_decl_lists([p.loc for p in self._params if not has_ddp(p, "loc")])
        scale_decls = sl.gen_decl_lists([p.scale for p in self._params if not has_ddp(p, "scale")])

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
        R, r = sl.Var(sl.Int(), "R"), sl.Var(sl.Int(), "r")
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

        index_loc = functools.partial(
            index_loc_for_block, num_pars=num_pars, fixed_level=self.level
        )

        trans_vals = [
            domain_transform(
                val, p.lbound, p.ubound, array=True, 
                loc=index_loc(p) if p.noncentered else None,
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
        if self._catcovs:
            # if we have categorical covariates, loc is an array of vectors
            loc = sl.expandVar(loc, (num_pars,))

        scale = sl.Var(sl.Vector(n, lower=sl.rzero()), f"scale_{self._name}")

        # the loc and scale vectors use the loc and scales from the parameters.
        # if the parameter is non-centered, the loc is zero (and we apply a linear transform later)
        # for the scales, we use 1.0 for non-centered parameters.
        if not self._catcovs:
            loc_decl = sl.DeclAssign(
                loc, sl.LiteralVector([p.loc if not p.noncentered else sl.rzero() for p in self._params])
            )
            # in this case we can directly assign the loc vector, below we would have to use a for-loop
            loc_decl_list = [loc_decl]
        else:
            # loc is an array, create this with a for-loop 
            loc_elts = []
            for p in self._params:
                if p.noncentered:
                    loc_elts.append(sl.rzero())
                elif p._catcovs:
                    catcov = p._catcovs[0]
                    if self.level is not None and self.level_type == "fixed":
                        # in this case, the loc array is defined on the level
                        # so we have to index it with the restricted categories
                        restricted_catcov_var = catcov.restricted_var(self.level)
                        catcov_var_array = sl.expandVar(restricted_catcov_var, (num_pars,))
                    else:
                        catcov_var_array = sl.expandVar(catcov.var, (num_pars,))

                    # select the correct loc value based on the categorical covariate
                    loc_elt = p.loc.idx(catcov_var_array.idx(r))
                    loc_elts.append(loc_elt)
                else:
                    # otherwise, just use the single loc value
                    loc_elts.append(p.loc)
            loc_decl = sl.Decl(loc)
            loc_assign = sl.ForLoop(
                r, sl.Range(sl.one(), num_pars),
                sl.Assign(
                    loc.idx(r),
                    sl.LiteralVector(loc_elts)
                )
            )
            # first declare loc, then assign values in for-loop
            loc_decl_list = [loc_decl, loc_assign]
        real_one = sl.LiteralReal(1.0)
        scale_decl = sl.DeclAssign(
            scale, sl.LiteralVector([p.scale if not p.noncentered else real_one for p in self._params])
        )

        trans_decls += [
            sl.comment(f"loc and scale vectors for correlated parameter block {self._name}"),
        ] + loc_decl_list + [scale_decl]

        # define weight vectors if there are any covariates
        # group parameters according to covariates, insert zero weights for other parameters
        def zero_weight(cov: ContCovariate):
            if cov.dim is None:
                return sl.rzero()
            # else: return zero vector of correct dimension
            return sl.Call("zeros_vector", [cov.dim_var])
        
        cov_par_dict = {
            cov.name: [
                cov.weight_var(p.name) if cov in p._contcovs else zero_weight(cov)
                for p in self._params
            ]
            for cov in self._contcovs
        }

        weight_vecs = []
        for cov in self._contcovs:
            if cov.dim is None:
                # weight vectors for scalar covariates
                # warning: don't use expandVecVar here: this creates a row_vector
                vec_decl = sl.DeclAssign(
                    sl.Var(sl.Vector(n), cov.weight_name(self._name)),
                    sl.LiteralVector(cov_par_dict[cov.name]),
                )
            else:
                # weight matrices for vector-valued covariates
                d = cov.dim_var
                vec_decl = sl.DeclAssign(
                    sl.Var(sl.Matrix(d, n), cov.weight_name(self._name)),
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
        # compute correlation matrix
        n = sl.LiteralInt(len(self._params))
        corrmat_decl = sl.DeclAssign(
            sl.Var(sl.Matrix(n, n), f"corr_{self._name}"),
            sl.Call("tcrossprod", [self._chol]),
        )
        stmts: list[sl.Stmt] = [corrmat_decl]
        # compute "random effects" for non-centered parameters (if any)

        nc_params = [p for p in self._params if p.noncentered]
        if not nc_params:
            return stmts

        stmts.append(sl.comment("random effects for non-centered parameters in the correlation block"))

        R = sl.intVar("R")
        one_par_per_unit = self.level is None # note that random level is not supported here
        num_pars = R if one_par_per_unit else self.level.num_cat_var
        array_var = sl.expandVar(self.var, (num_pars,)) # array[R] vector[n] block_x_y_z
        
        for i, p in enumerate(self._params):
            if not p.noncentered:
                continue
            rand_var = sl.expandVar(p.rand_var, (num_pars,))
            value = array_var.idx(sl.fullRange(), sl.LiteralInt(i + 1))
            decl = sl.DeclAssign(rand_var, value)
            stmts.append(decl)
        return stmts

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

        R, r = sl.Var(sl.Int(), "R"), sl.Var(sl.Int(), "r")
        one_par_per_unit = self.level is None
        num_pars = R if one_par_per_unit else self.level.num_cat_var

        # use the right category from loc if there are categorical covariates
        index_loc = functools.partial(
            index_loc_for_block, num_pars=num_pars, 
            fixed_level=self.level, index=r
        )

        loc_terms_elts = [[index_loc(p)] for p in self._params]
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
