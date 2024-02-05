import numpy as np
import scipy.special
from typing import Callable

class Transform:
    def __init__(self, forward: Callable, backward: Callable) -> None:
        self._forward = forward
        self._backward = backward
    
    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._forward(x)
        
    def inverse(self):
        return Transform(self._backward, self._forward)
    
    def __matmul__(self, other):
        def forward(x):
            return self._forward(other._forward(x))
        def backward(y):
            return other._backward(self._backward(y))
        return Transform(forward, backward)
    
    
    
class IdentityTransform(Transform):
    def __init__(self):
        def iden(x):
            return x
        super().__init__(iden, iden)
    
    
class LogTransform(Transform):
    def __init__(self) -> None:
        super().__init__(np.log, np.exp)
        

class LogitTransform(Transform):
    def __init__(self) -> None:
        super().__init__(scipy.special.logit, scipy.special.expit)
        
        
class AffineTransform(Transform):
    def __init__(self, offset: float, slope: float):
        self._offset = offset
        self._slope = slope
        def forward(x):
            return self._offset + x * self._slope
        def backward(y):
            return (y - self._offset) / self._slope
        super().__init__(forward, backward)
        
        
class ShiftedLogTransform(Transform):
    def __new__(cls, lower_bound: float):
        afft = AffineTransform(-lower_bound, 1.0)
        logt = LogTransform()
        obj = logt @ afft
        obj.__class__ = cls
        return obj
    def __init__(self, lower_bound: float):
        self._lower_bound = lower_bound
        

class NegLogTransform(Transform):
    def __new__(cls):
        negt = AffineTransform(0.0, -1.0)
        logt = LogTransform()
        obj = negt @ logt @ negt
        obj.__class__ = cls
        return obj
    def __init__(self):
        pass


class ShiftedNegLogTransform(Transform):
    def __new__(cls, upper_bound: float):
        afft = AffineTransform(upper_bound, -1.0)
        negt = AffineTransform(0.0, -1.0)
        logt = LogTransform()
        obj = negt @ logt @ afft
        obj.__class__ = cls
        return obj
    def __init__(self, upper_bound):
        self._upper_bound = upper_bound
        
        
class GeneralizedLogitTransform(Transform):
    def __new__(cls, lower_bound: float, upper_bound: float):
        width = upper_bound - lower_bound
        afft = AffineTransform(-lower_bound / width, 1.0 / width)
        logitt = LogitTransform()
        obj = logitt @ afft
        obj.__class__ = cls
        return obj
    def __init__(self, lower_bound: float, upper_bound: float):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    
def logit_transform_dispatch(lower_bound: float | None, upper_bound: float | None) -> Transform:
    match (lower_bound, upper_bound):
        case (None, None):
            return IdentityTransform()
        case (0.0, None):
            return LogTransform()
        case (float(), None):
            return ShiftedLogTransform(lower_bound)
        case (None, 0.0):
            return NegLogTransform()
        case (None, float()):
            return ShiftedNegLogTransform(upper_bound)
        case (0.0, 1.0):
            return LogitTransform()
        case (float(), float()):
            return GeneralizedLogitTransform(lower_bound, upper_bound)
        case _:
            raise ValueError(f"invalid combination of parameters {lower_bound} and {upper_bound}")


def constr_to_unconstr_float(
    val: float, lbound: float | None, ubound: float | None
) -> float:
    """
    Transform an constraint float to an unconstraint float.
    Unbounded values are returned as-is. Values bounded from below
    are translated and log-transformed. Idem for values bounded from above,
    but the Jacobian is kept positive. Values on a compact domain are scaled
    to the unit interval and logit-transformed.

    Parameters
    ----------
    val : float
        Value in the constraint domain.
    lbound : Optional[float]
        Lower bound. None means -inf
    ubound : Optional[float]
        Upper bound, None means inf.

    Raises
    ------
    Exception
        If an invalid combination of lower and upper bounds is given,
        an Exception is raised.

    Returns
    -------
    float
        unconstrained value.

    """
    return logit_transform_dispatch(lbound, ubound)(val)
