from enum import Enum

from . import stanlang as sl


class Censoring(Enum):
    LEFT_CENSORED = -1
    UNCENSORED = 0
    RIGHT_CENSORED = 1
    MISSING = 2


"""
TODO: we may want to restrict censoring possibilities. Often
only e.g. left-censored and uncensored observations exist
in which case we don't have to test for all the possibilies
TODO: the user might want to define a name different from cc_X
for the censor codes of observation X.
TODO: left and right censoring only makes sense for real observations
"""


class Observation:
    def __init__(
        self,
        name: str,
        obs_type: sl.Type = sl.Real(),
        censored: bool = False,
    ) -> None:
        self._name = name

        self._obs_type = obs_type

        self._censored = censored
        self._cc_name = f"cc_{name}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def obs_type(self) -> sl.Type:
        return self._obs_type

    @property
    def var(self) -> sl.Var:
        return sl.Var(self._obs_type, self._name)

    @property
    def censored(self) -> bool:
        return self._censored

    @property
    def cc_name(self) -> str:
        return self._cc_name

    def genstmt_data(self) -> list[sl.Stmt]:
        R = sl.Var(sl.Int(), "R")
        N = sl.Var(sl.Int(), "N")
        maxN = sl.Call("max", [N])
        decl_obs = sl.Decl(sl.expandVar(self.var, (R, maxN)))
        if not self._censored:
            return [decl_obs]
        # else, add data object for censoring codes
        # -1 = left-censored, 0 = uncensored, 1 = right-censored, 2 = missing
        decl_cc = sl.Decl(
            sl.Var(
                sl.Array(
                    sl.Int(lower=sl.LiteralInt(-1), upper=sl.LiteralInt(2)), (R, maxN)
                ),
                self._cc_name,
            ),
            comment=f"censor codes for {self._name}",
        )
        return [decl_obs, decl_cc]
