"""
check user code for HBOMS errors
"""
from . import stanparser
from . import stanlang as sl

from .logger import logger

def verify_vars_assigned(code: str, var_names: list[str]) -> dict[str, bool]:
    """
    Check that in the user's code, the variables with names in var_names
    are assigned values. For instance, in the ODE model, the user should
    define values to the derivatives ddt_X of the state variables X

    Parameters
    ----------
    code : str
        User's code.
    state_var_names : list[str]
        list of variables that require assignments.

    Returns
    -------
    dict[str, bool]
        Each variable in var_names is mapped to a bool indicating that the
        code contains an assignment or not for that variable.

    """

    vars_assigned = {x: False for x in var_names}
    if len(code) == 0:
        # if the code is empty, then no variables are assigned
        return vars_assigned

    stmt_list = stanparser.parser.parse(
        code, lexer=stanparser.lexer
    )  ## FIXME: choose entry point!
    if stmt_list is None:  ## parse error
        return vars_assigned
    for stmt in stmt_list:
        match stmt:
            case sl.Assign(sl.Var(var_type, name), rhs):
                if name in var_names:
                    vars_assigned[name] = True
            case _:
                pass

    # FIXME: assignments don't have to ba at top level, so we should
    # also check for assignments in blocks.
    
    return vars_assigned


def verify_odes(code: str, state_var_names: list[str]) -> bool:
    var_names = [f"ddt_{X}" for X in state_var_names]
    vars_assigned = verify_vars_assigned(code, var_names)
    if not all(vars_assigned.values()):
        message = "The following derivatives were not assigned: "
        message += ", ".join([x for x in var_names if not vars_assigned[x]]) + ". "
        message += "Please check your ODE model code. "
        message += "This warning could also be the result of a parse error."
        logger.warning(message)
        return False
    return True


def verify_init(code: str, state_var_names: list[str]) -> bool:
    var_names = [f"{X}_0" for X in state_var_names]
    vars_assigned = verify_vars_assigned(code, var_names)
    if not all(vars_assigned.values()):
        message = "The following initial conditions were not assigned: "
        message += ", ".join([x for x in var_names if not vars_assigned[x]]) + ". "
        message += "Please check your initial condition code. "
        message += "This warning could also be the result of a parse error."
        logger.warning(message)
        return False
    return True


def verify_transform(code: str, var_names: list[str]) -> bool:
    vars_assigned = verify_vars_assigned(code, var_names)
    if not all(vars_assigned.values()):
        message = "The following transformed state variables were not assigned: "
        message += ", ".join([x for x in var_names if not vars_assigned[x]]) + ". "
        message += "Please check your transform code. "
        message += "This warning could also be the result of a parse error."
        logger.warning(message)
        return False
    return True
