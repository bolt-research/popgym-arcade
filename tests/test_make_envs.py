import jax
import pytest

import popgym_arcade
from popgym_arcade.registration import REGISTERED_ENVIRONMENTS


@pytest.mark.parametrize("env_name", REGISTERED_ENVIRONMENTS)
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("obs_size", [128, 256])
def test_make(env_name, partial, obs_size):
    env, env_params = popgym_arcade.make(
        env_name, partial_obs=partial, obs_size=obs_size
    )


if __name__ == "__main__":
    pytest.main()
