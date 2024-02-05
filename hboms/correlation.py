import numpy as np
from typing import Optional


class Correlation:
    def __init__(
        self,
        parnames: list[str],
        value: Optional[np.ndarray] = None,
        intensity: float = 2.0,
    ) -> None:
        self._parnames = parnames
        self._dim = len(parnames)
        self._intensity = intensity
        if value is None:
            # default correlation matrix is identity
            self._value = np.eye(self._dim)
        elif isinstance(value, np.ndarray):
            # check shape
            req_shp = (self._dim, self._dim)  # required shape
            given_shp = value.shape
            if given_shp != req_shp:
                raise Exception(
                    f"value has incompatible shape {given_shp}. Required shape is {req_shp}"
                )
            # if shape OK, use argument
            self._value = value
        else:
            raise Exception("optional argument value must be a numpy array")

    @property
    def parnames(self) -> list[str]:
        return self._parnames

    @property
    def value(self) -> np.ndarray:
        return self._value

    @property
    def intensity(self) -> float:
        return self._intensity

    @property
    def dim(self) -> int:
        return self._dim
