# check names: parameters, state variables, covariates, observations

def check_names(params: list[str], state_vars: list[str], covariates: list[str], observations: list[str]) -> None:
    """
    Check that the names of parameters, state variables, covariates, and observations are unique.
    """
    all_names = set()
    
    for p in params:
        if p in all_names:
            raise ValueError(f"Duplicate parameter name: {p}")
        all_names.add(p)
    
    for sv in state_vars:
        if sv in all_names:
            raise ValueError(f"Duplicate state variable name: {sv}")
        all_names.add(sv)
    
    for c in covariates:
        if c in all_names:
            raise ValueError(f"Duplicate covariate name: {c}")
        all_names.add(c)
    
    for ob in observations:
        if ob in all_names:
            raise ValueError(f"Duplicate observation name: {ob}")
        all_names.add(ob)

    # Check for reserved names
    reserved_names = { ## TODO: add more reserved names as needed
        "R", "r", "N", "n", "Nsim", "Time", "TimeSim", 
        "t", "u_sim", "u_sim_obs", "ppar", "upars", 
        "rdats", "idats", "state"
    }
    for name in reserved_names:
        if name in all_names:
            raise ValueError(f"Reserved variable name used: {name}")
        
    # check that reserved prefixes are not used
    reserved_prefixes = { ## TODO: add more reserved prefixes as needed
        "loc_", "scale_", "chol_", "rand_", "block_", "weight_"
    }

    for name in all_names:
        if any(name.startswith(prefix) for prefix in reserved_prefixes):
            raise ValueError(f"Reserved prefix used in name: {name}")
        
    # Check for empty names
    if any(not name for name in all_names):
        raise ValueError("Empty name found. All names must be non-empty.")
    
