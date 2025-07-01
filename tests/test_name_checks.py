from hboms import name_checks
import pytest


class TestNameChecks:
    def test_check_names(self):
        # Test with unique names: this should pass without raising an error

        name_checks.check_names(
            params=["param1", "param2"],
            state_vars=["state1", "state2"],
            covariates=["cov1", "cov2"],
            observations=["obs1", "obs2"]
        )

        # Test with duplicate names
        with pytest.raises(ValueError, match="Duplicate parameter name: param1"):
            name_checks.check_names(
                params=["param1", "param1"],
                state_vars=["state1", "state2"],
                covariates=["cov1", "cov2"],
                observations=["obs1", "obs2"]
            )

        # Test with reserved names
        with pytest.raises(ValueError, match="Reserved variable name used: R"):
            name_checks.check_names(
                params=["param1", "param2"],
                state_vars=["state1", "state2"],
                covariates=["R", "cov2"],
                observations=["obs1", "obs2"]
            )

        # Test with reserved prefixes
        with pytest.raises(ValueError, match="Reserved prefix used in name: loc_param"):
            name_checks.check_names(
                params=["loc_param", "param2"],
                state_vars=["state1", "state2"],
                covariates=["cov1", "cov2"],
                observations=["obs1", "obs2"]
            )

        # Test with empty names
        with pytest.raises(ValueError, match="Empty name found. All names must be non-empty."):
            name_checks.check_names(
                params=["param1", "param2"],
                state_vars=["state1", ""],
                covariates=["cov1", "cov2"],
                observations=["obs1", "obs2"]
            )