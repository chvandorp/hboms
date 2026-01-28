from hboms import stanlang as sl
from typing import Optional

def domain_transform(
    val: sl.Expr, lbound: Optional[float], ubound: Optional[float], array: bool = False,
    loc: Optional[sl.Expr] = None, scale: Optional[sl.Expr] = None,
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

    to_vec_applied = False

    def to_vec(x):
        nonlocal to_vec_applied
        to_vec_applied = True
        return sl.Call("to_vector", [x]) if array else x

    def to_array(x):
        return sl.Call("to_array_1d", [x]) if array else x

    match (loc, scale):
        case (None, None):
            # no linear transformation
            pass
        case (sl.Expr(), None):
            # apply a linear transformation to the value
            val = to_vec(val) + loc
        case (None, sl.Expr()):
            # apply a linear transformation to the value
            val = scale * to_vec(val)
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
            tval = val
        case (float(), None):
            exp_val = sl.Call("exp", [val])
            if lbound == 0.0:
                tval = exp_val
            else: # we have to convert to vector to translate
                tval = sl.LiteralReal(lbound) + to_vec(exp_val)
        case (None, float()):
            exp_vec_val = sl.Call("exp", [sl.Negate(to_vec(val))])
            if ubound == 0.0:
                tval = sl.Negate(exp_vec_val)
            else:
                tval = sl.LiteralReal(ubound) - exp_vec_val
        case (float(), float()):
            expit_val = sl.Call("inv_logit", [val])
            lb = sl.LiteralReal(lbound)
            ub = sl.LiteralReal(ubound)
            if lbound == 0.0 and ubound == 1.0:
                tval = expit_val
            elif lbound == 0.0:
                tval = ub * to_vec(expit_val)
            else:
                tval = lb + sl.Par(ub - lb) * to_vec(expit_val)
        case _:
            raise Exception("invalid parameter bounds")

    return to_array(tval) if to_vec_applied else tval
    


def inverse_domain_transform(
    val: sl.Expr, lbound: Optional[float], ubound: Optional[float], array: bool = False,
    loc: Optional[sl.Expr] = None, scale: Optional[sl.Expr] = None,
) -> sl.Expr:
    """
    Generate expressions for transforming constraint values to a unbounded domain.

    Parameters
    ----------
    val : sl.Expr
        the constrained value to-be transformed.
    lbound : Optional[float]
        lower bound of the untransformed value.
    ubound : Optional[float]
        upper bound of the untransformed value.
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
        Stan expression for the inverse-transformed value.
    """

    to_vec_applied = False

    def to_vec(x):
        nonlocal to_vec_applied
        to_vec_applied = True
        return sl.Call("to_vector", [x]) if array else x

    def to_array(x):
        return sl.Call("to_array_1d", [x]) if array else x
    
    match (lbound, ubound):
        case (None, None):
            tval = val
        case (float(), None):
            if lbound == 0.0:
                tval = sl.Call("log", [val])
            else:
                tval = sl.Call("log", [to_vec(val) - sl.LiteralReal(lbound)])
        case (None, float()):
            vec_val = to_vec(val)
            if ubound == 0.0:
                tval = sl.Negate(sl.Call("log", [sl.Negate(vec_val)]))
            else:
                tval = sl.Negate(sl.Call("log", [sl.LiteralReal(ubound) - vec_val]))
        case (float(), float()):
            lb = sl.LiteralReal(lbound)
            ub = sl.LiteralReal(ubound)
            if lbound == 0.0 and ubound == 1.0:
                tval = sl.Call("logit", [val])
            elif lbound == 0.0:
                tval = sl.Call("logit", [to_vec(val) / ub])
            else:
                scaled_val = sl.Par(to_vec(val) - lb) / sl.Par(ub - lb)
                tval = sl.Call("logit", [scaled_val])
        case _:
            raise Exception("invalid parameter bounds")

    # apply inverse linear transformation if needed

    if to_vec_applied:
        # do not apply to_vector twice..."""
        to_vec = lambda x: x

    match (loc, scale):
        case (None, None):
            # no linear transformation
            ltval = tval
        case (sl.Expr(), None):
            # apply a linear transformation to the value
            ltval = to_vec(tval) - loc
        case (None, sl.Expr()):
            # apply a linear transformation to the value
            ltval = to_vec(tval) / scale
        case (sl.Expr(), sl.Expr()):
            # apply a linear transformation to the value
            ltval = sl.Par(to_vec(tval) - loc) / scale
        case _:
            raise Exception("invalid loc and scale parameters for domain transformation")
    
    return to_array(ltval) if to_vec_applied else ltval

