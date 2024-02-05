from . import stanlang as sl

left_censored_code = sl.LiteralInt(-1)
uncensored_code = sl.LiteralInt(0)
right_censored_code = sl.LiteralInt(1)
missing_code = sl.LiteralInt(2)


supported_integrators = [
    "ode_rk45_tol",
    "ode_rk45",
    "ode_bdf_tol",
    "ode_bdf",
    "ode_adams_tol",
    "ode_adams",
    "ode_ckrk_tol",
    "ode_ckrk",
]
