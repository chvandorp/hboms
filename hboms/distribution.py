from typing import List, Union, Literal

from .observation import Observation
from . import stanlang as sl
from . import definitions as defn


def genstmt_censored_loglik(
    family: str, obs_str: str, cc_str: str, *pars: sl.Expr, discrete: bool = False
) -> sl.Stmt:

    if discrete:
        raise NotImplementedError("discrete censored distributions not yet implemented")

    ll_obs = sl.Var(sl.Real(), f"ll_{obs_str}")  # TODO: shape of obs? (if independent)
    obs = sl.Var(sl.Real(), obs_str)  # TODO: type of obs
    cc = sl.Var(sl.Int(), cc_str)
    stmt = sl.IfElseStatement(
        sl.EqOp(cc, defn.left_censored_code),
        sl.Assign(
            ll_obs,
            sl.PCall(family, obs, list(pars), "lcdf"),
            comment="left-censored observation",
        ),
        sl.IfElseStatement(
            sl.EqOp(cc, defn.uncensored_code),
            sl.Assign(
                ll_obs,
                sl.PCall(family, obs, list(pars), "lpdf"),
                comment="uncensored observation",
            ),
            sl.IfStatement(
                sl.EqOp(cc, defn.right_censored_code),
                sl.Assign(
                    ll_obs,
                    sl.PCall(family, obs, list(pars), "lccdf"),
                    comment="right-censored observation",
                ),
            ),
        ),
    )
    return stmt


def genstmt_loglik(
    family: str, obs_str: str, *pars: sl.Expr, discrete: bool = False
) -> sl.Stmt:
    obs_type = (
        sl.Int() if discrete else sl.Real()
    )  ## FIXME: use actual observation object
    obs: sl.Expr = sl.Var(obs_type, obs_str)
    suffix: sl.DistSuffix = "lpmf" if discrete else "lpdf"
    stmt = sl.Assign(
        sl.Var(obs_type, f"ll_{obs_str}"),
        sl.PCall(family, obs, list(pars), suffix),
        comment=f"{family} log-likelihood",
    )
    return stmt


def genstmt_rng(family: str, obs: str, *pars: sl.Expr) -> sl.Stmt:
    stmt = sl.Assign(
        sl.Var(sl.Real(), f"{obs}"),
        sl.Call(family + "_rng", list(pars)),
        comment=f"random {family} sample",
    )
    return stmt


# Distribution classes and derived classes


class Distribution:
    def __init__(self, obs: Observation) -> None:
        self._obs = obs

    @property
    def obs(self) -> Observation:
        return self._obs

    def genstmt_loglik(self) -> sl.Stmt:
        return sl.EmptyStmt(comment="trivial likelihood function")

    def genstmt_rng(self) -> sl.Stmt:
        return sl.EmptyStmt(comment="no rng specified")

    def params(self) -> list[sl.Expr]:
        return []


### continuous distributions


class NormalDist(Distribution):
    def __init__(self, obs: Observation, loc: sl.Expr, scale: sl.Expr) -> None:
        self._obs = obs
        self._loc = loc
        self._scale = scale

    def genstmt_loglik(self) -> sl.Stmt:
        if self._obs.censored:
            stmt = genstmt_censored_loglik(
                "normal", self._obs.name, self._obs.cc_name, self._loc, self._scale
            )
        else:
            stmt = genstmt_loglik("normal", self._obs.name, self._loc, self._scale)

        return stmt

    def genstmt_rng(self) -> sl.Stmt:
        code = genstmt_rng("normal", self._obs.name, self._loc, self._scale)
        return code

    def params(self) -> list[sl.Expr]:
        return [self._loc, self._scale]


class LognormalDist(Distribution):
    def __init__(self, obs: Observation, loc: sl.Expr, scale: sl.Expr) -> None:
        self._obs = obs
        self._loc = loc
        self._scale = scale
        self._log_loc = sl.Call("log", [self._loc])

    def genstmt_loglik(self) -> sl.Stmt:
        if self._obs.censored:
            stmt = genstmt_censored_loglik(
                "lognormal",
                self._obs.name,
                self._obs.cc_name,
                self._log_loc,
                self._scale,
            )
        else:
            stmt = genstmt_loglik(
                "lognormal", self._obs.name, self._log_loc, self._scale
            )

        return stmt

    def genstmt_rng(self) -> sl.Stmt:
        stmt = genstmt_rng("lognormal", self._obs.name, self._log_loc, self._scale)
        return stmt

    def params(self) -> list[sl.Expr]:
        return [self._loc, self._scale]


### discrete distributions


class PoissonDist(Distribution):
    def __init__(self, obs: Observation, loc: sl.Expr) -> None:
        self._obs = obs
        self._loc = loc

    def genstmt_loglik(self) -> sl.Stmt:
        stmt = genstmt_loglik("poisson", self._obs.name, self._loc, discrete=True)
        return stmt

    def genstmt_rng(self) -> sl.Stmt:
        stmt = genstmt_rng("poisson", self._obs.name, self._loc)
        return stmt

    def params(self) -> list[sl.Expr]:
        return [self._loc]


class NegBinomDist(Distribution):
    def __init__(self, obs: Observation, loc: sl.Expr, shape: sl.Expr) -> None:
        self._obs = obs
        self._loc = loc
        self._shape = shape

    def genstmt_loglik(self) -> sl.Stmt:
        stmt = genstmt_loglik(
            "neg_binomial_2", self._obs.name, self._loc, self._shape, discrete=True
        )
        return stmt

    def genstmt_rng(self) -> sl.Stmt:
        stmt = genstmt_rng("neg_binomial_2", self._obs.name, self._loc, self._shape)
        return stmt

    def params(self) -> list[sl.Expr]:
        return [self._loc, self._shape]


# do we really need these individual distributions? Just use a common class


class StanDist(Distribution):
    def __init__(
        self,
        dist_name: str,
        obs: Observation,
        *params: sl.Expr,
        discrete: bool | Literal["infer"] = "infer",
        censored: bool | Literal["infer"] = "infer",
    ) -> None:
        """
        Generic Stan-style distribution. The user can specify if the
        distribution is discrete, or this can be inferred from the observation
        type.

        TODO: more documentation!
        """
        self._obs = obs
        self._dist_name = dist_name
        self._params = list(params)

        if isinstance(discrete, bool):
            self._discrete = discrete
        elif discrete == "infer":
            self._discrete = obs.obs_type.is_discrete()
        else:
            raise Exception(f"invalid value '{discrete}' for kwarg 'discrete'")

        if isinstance(censored, bool):
            self._censored = censored
        elif censored == "infer":
            self._censored = obs.censored
        else:
            raise Exception(f"invalid value '{censored}' for kwarg 'censored'")

    def genstmt_loglik(self) -> sl.Stmt:
        if self._censored:
            stmt = genstmt_censored_loglik(
                self._dist_name,
                self._obs.name,
                self._obs.cc_name,
                *self._params,
                discrete=self._discrete,
            )
        else:
            stmt = genstmt_loglik(
                self._dist_name, self._obs.name, *self._params, discrete=self._discrete
            )
        return stmt

    def genstmt_rng(self) -> sl.Stmt:
        stmt = genstmt_rng(self._dist_name, self._obs.name, *self._params)
        return stmt

    def params(self) -> list[sl.Expr]:
        return self._params


### custom distribution


## TODO: rng should be optional!!


class CustomDist(Distribution):
    def __init__(self, code_loglik: str, code_rng: str) -> None:
        self._code_loglik = code_loglik.strip()
        self._code_rng = code_rng.strip()

    def genstmt_loglik(self) -> sl.Stmt:
        return sl.MixinStmt(self._code_loglik)

    def genstmt_rng(self) -> sl.Stmt:
        return sl.MixinStmt(self._code_rng)

    ## FIXME: params is not defined
