from typing import Any, Dict, Optional, Tuple, Union

import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
import functools
import time
from gymnax.environments import environment
from gymnax.environments import spaces

from popgym_arcade.environments.draw_utils import (draw_crooked_arrow,
                                            draw_horizontal_arrow,
                                            draw_rectangle,
                                            draw_pole,
                                            draw_number,
                                            draw_sub_canvas,
                                            draw_str)


@struct.dataclass
class EnvState(environment.EnvState):
    x: chex.Array
    x_dot: chex.Array
    theta: chex.Array
    theta_dot: chex.Array
    score: int
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    total_mass: float = 1.0 + 0.1  # (masscart + masspole)
    length: float = 0.5
    polemass_length: float = 0.05  # (masspole * length)
    force_mag: float = 10.0
    tau: float = 0.02
    theta_threshold_radians: float = 12 * 2 * jnp.pi / 360
    x_threshold: float = 2.4


class CartPole(environment.Environment[EnvState, EnvParams]):
    """
    ### Description

    JAX Compatible version of CartPole OpenAI gym environment.
    Source: github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

    We have modified some settings of the original code to align with our main design ideas.

    ### 1. Action Space
    The action is a `int / float / chex.Array` with shape `(1, )`,
    which can take value `{0, 1, 2, 3, 4}`

    | Num | Action                     |
    |-----|----------------------------|
    | 0   | No force added to the cart |
    | 1   | No force added to the cart |
    | 2   | Push cart to the left      |
    | 3   | Push cart to the right     |
    | 4   | No force added to the cart |

    ### 2.Fully Observation Space
    The observation space is a `chex.Array` with shape `(256, 256, 3)`
    Current state is rendered into a matrix using multiple graphical
    elements to form a visual observation. The entire observation
    consists of a large canvas with a smaller canvas embedded into it.
    The smaller canvas primarily displays the game interface,
    while the larger canvas shows additional information, such as the score.

    In CartPole, the main visuals are a cart with a vertical pole,
    and they update based on the state variables. Horizontal arrows
    show the cart's speed,and crooked arrows show the pole's speed.
    The size of the arrows changes depending on how fast the cart and pole are moving.

    At the top of the observation, the current score is displayed dynamically.
    Whenever the reward function returns a positive value, the score increases by one point.

    ### 3.Partially Observation Space
    In Partially Observation Space, the cart and pole are hidden when timesteps > 0.
    The agent can only observe two arrows.

    ### 4.Rewards
    Reward is normalized to (0, 1)
    Each step: reward = 1.0 / max_steps_in_episode

    ### 5.args:
    n_sigma: std for noise in NoisyCartPole, from easy to hard is in {0.1, 0.2, 0.3}
    partial_obs: bool switch with POMDP and FOMDP.
    max_steps_in_episode: max steps agent can play in each episode.
    """

    render_canvas = {
        "size": 256,
        "clr": jnp.array([0.2, 0.2, 0.2]),
        "sub_size": 192,
        "sub_clr": jnp.array([0.2, 0.2, 0.2]),
    }

    render_cart = {
        "cart_w": 64,
        "cart_h": 32,
        "cart_pos": 96,
        "cart_clr": jnp.array([0.96, 0.29, 0.55]),
    }

    render_pole = {
        "pole_clr": jnp.array([0.55, 0.27, 0.07]),
        "pole_px": 5,
    }

    render_harrow = {
        "harrow_t_l": (10, 138),
        "harrow_b_r": (54, 182),
        "harrow_clr":  jnp.array([0.8, 0.8, 0.8]),
    }

    render_carrow = {
        "carrow_t_l": (138, 138),
        "carrow_b_r": (182, 182),
        "carrow_clr": jnp.array([0.8, 0.8, 0.8]),
    }

    render_score = {
        "sc_t_l": (86, 2),
        "sc_b_r": (171, 30),
        "sc_clr": jnp.array([0.0, 1.0, 0.5]),

    }

    render_envName = {
        "env_t_l": (0, 231),
        "env_b_r": (256, 256),
        "env_clr": jnp.array([0.29, 0.84, 0.97]),
    }

    def __init__(
            self,
            n_sigma: float = 0.0,
            max_steps_in_episode: int = 200,
            partial_obs: bool = False,
    ):
        """
        Initialize the environment.

        Args:
            n_sigma: Standard deviation of noise added to observations.
            max_steps_in_episode: Maximum number of steps per episode.
            partial_obs: Whether to use partial observability.
        """
        self.n_sigma = n_sigma
        self.max_steps_in_episode = max_steps_in_episode
        self.partial_obs = partial_obs

    @property
    def default_params(self) -> EnvParams:
        """Return the default environment parameters."""
        return EnvParams()

    @functools.partial(jax.jit, static_argnums=(0,))
    def step_env(
            self,
            key: chex.PRNGKey,
            state: EnvState,
            action: Union[int, float, chex.Array],
            params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        """
        Perform a step in the environment.

        Args:
            key: Random key for JAX operations.
            state: Current environment state.
            action: Action to take.
            params: Environment parameters.

        Returns:
            A tuple containing:
            - Observation after the step.
            - Updated state.
            - Reward.
            - Whether the episode is done.
            - Additional info (e.g., discount factor).
        """
        prev_terminal = self.is_terminal(state, params)

        force = lax.cond(
            action == 2,
            lambda _: -params.force_mag,
            lambda _: lax.cond(
                action == 3,
                lambda _: params.force_mag,
                lambda _: 0.0,
                operand=None,
            ),
            operand=None,
        )

        # Calculate trigonometric values

        costheta = jnp.cos(state.theta)
        sintheta = jnp.sin(state.theta)

        # Calculate acceleration and update state
        temp = (
            force + params.polemass_length * state.theta_dot ** 2 * sintheta
        ) / params.total_mass

        thetaacc = (
                (params.gravity * sintheta - costheta * temp)
                / (
                        params.length
                        * (
                                4.0 / 3.0
                                - params.masspole * costheta ** 2 / params.total_mass
                        )
                )
        )
        xacc = temp - params.polemass_length * thetaacc * costheta / params.total_mass
        x = state.x + params.tau * state.x_dot
        x_dot = state.x_dot + params.tau * xacc
        theta = state.theta + params.tau * state.theta_dot
        theta_dot = state.theta_dot + params.tau * thetaacc

        # Calculate reward and check if episode is done
        reward = (1.0 - prev_terminal) / self.max_steps_in_episode
        done = self.is_terminal(state, params)

        # Update score
        new_score = state.score + (reward > 0).astype(jnp.int32)

        # Update state
        state = EnvState(
            x=x,
            x_dot=x_dot,
            theta=theta,
            theta_dot=theta_dot,
            time=state.time + 1,
            score=new_score,
        )

        return (
            lax.stop_gradient(self.get_obs(state, params, key=key)),
            lax.stop_gradient(state),
            jnp.array(reward),
            done,
            {"discount": self.discount(state, params)},
        )
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset_env(
            self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """
        Reset the environment to an initial state.

        Args:
            key: Random key for JAX operations.
            params: Environment parameters.

        Returns:
            A tuple containing the initial observation and state.
        """
        init_state = jax.random.uniform(
            key, minval=-0.05, maxval=0.05, shape=(4,)
        )
        key_obs, _key = jax.random.split(key)
        state = EnvState(
            x=init_state[0],
            x_dot=init_state[1],
            theta=init_state[2],
            theta_dot=init_state[3],
            score=0,
            time=0,
        )
        init_obs = self.get_obs(state, params, key=key_obs)
        return init_obs, state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """
        Get the observation from the current state.

        Args:
            state: Current environment state.
            params: Environment parameters.
            key: Random key for JAX operations.

        Returns:
            A `chex.Array` representing the observation.
        """
        return self.render(state, params, key=key)

    @functools.partial(jax.jit, static_argnums=(0,))
    def render(
            self, state: EnvState, params: EnvParams, key=None
    ) -> chex.Array:
        """
        Render the current state into an image observation.

        Args:
            state: Current environment state.
            params: Environment parameters.
            key: Random key for JAX operations.

        Returns:
            A `chex.Array` of shape `(256, 256, 3)` representing the observation.
        """

        def map_value_to_canvas(value, car_width):
            """
            Map a value from the range (-2.4, 2.4) to (0, sub_size - car_width).

            Args:
                value: The value to map.
                car_width: The width of the cart.

            Returns:
                The mapped value.
            """
            map_factor = (
                    (self.render_canvas["sub_size"] - car_width)
                    / (params.x_threshold * 2)
            )
            map_bias = (
                    (self.render_canvas["sub_size"] - car_width)
                    - map_factor * 2.4
            )
            return value * map_factor + map_bias

        # Initialize canvas and sub-canvas
        canvas = jnp.zeros(
            (self.render_canvas["size"], self.render_canvas["size"], 3)
        ) + self.render_canvas["clr"]

        sub_canvas = jnp.zeros(
            (self.render_canvas["sub_size"], self.render_canvas["sub_size"], 3)
        ) + self.render_canvas["sub_clr"]

        # Add noise to the state
        noise = jax.random.normal(key, shape=(4,)) * self.n_sigma
        noisy_state = (
            jnp.clip(
                state.x + noise[0], -params.x_threshold, params.x_threshold
            ),
            state.x_dot + noise[1],
            jnp.clip(
                state.theta + noise[2],
                -params.theta_threshold_radians,
                params.theta_threshold_radians,
            ),
            state.theta_dot + noise[3],
        )

        # Map noisy state to canvas coordinates
        x = map_value_to_canvas(noisy_state[0], self.render_cart["cart_w"]).astype(jax.numpy.int32)
        x_dot = noisy_state[1]
        theta = noisy_state[2]
        theta_dot = noisy_state[3]

        # Define cart and pole positions
        cart_t_l = (x, self.render_cart["cart_pos"])
        cart_b_r = (
            x + self.render_cart["cart_w"],
            self.render_cart["cart_pos"] + self.render_cart["cart_h"],
        )
        pole_start = (
            (cart_t_l[0] + cart_b_r[0]) // 2,
            (cart_t_l[1] + cart_b_r[1]) // 2,
        )
        pole_end = (
            pole_start[0],
            pole_start[1] - self.render_cart["cart_w"],
        )

        def render_partial(sub_canvas):
            """Render the partially observable state."""
            sub_canvas = draw_horizontal_arrow(
                self.render_harrow["harrow_t_l"],
                self.render_harrow["harrow_b_r"],
                self.render_harrow["harrow_clr"],
                x_dot,
                sub_canvas,
            )
            sub_canvas = draw_crooked_arrow(
                self.render_carrow["carrow_t_l"],
                self.render_carrow["carrow_b_r"],
                self.render_carrow["carrow_clr"],
                theta_dot,
                sub_canvas,
            )
            return sub_canvas

        def render_full(sub_canvas):
            """Render the fully observable state."""
            sub_canvas = draw_rectangle(
                cart_t_l, cart_b_r, self.render_cart["cart_clr"], sub_canvas
            )
            sub_canvas = draw_pole(
                pole_start,
                pole_end,
                self.render_pole["pole_clr"],
                theta,
                self.render_pole["pole_px"],
                sub_canvas,
            )
            sub_canvas = draw_horizontal_arrow(
                self.render_harrow["harrow_t_l"],
                self.render_harrow["harrow_b_r"],
                self.render_harrow["harrow_clr"],
                x_dot,
                sub_canvas,
            )
            sub_canvas = draw_crooked_arrow(
                self.render_carrow["carrow_t_l"],
                self.render_carrow["carrow_b_r"],
                self.render_carrow["carrow_clr"],
                theta_dot,
                sub_canvas,
            )
            return sub_canvas

        # Draw score and environment name
        canvas = draw_number(
            self.render_score["sc_t_l"],
            self.render_score["sc_b_r"],
            jnp.array([1, 0, 0]),
            canvas,
            state.score,
        )
        sub_canvas = lax.select(
            state.time == 0,
            render_full(sub_canvas),
            lax.select(
                self.partial_obs,
                render_partial(sub_canvas),
                render_full(sub_canvas),
            ),
        )
        canvas = draw_str(
            self.render_envName["env_t_l"],
            self.render_envName["env_b_r"],
            self.render_envName["env_clr"],
            canvas,
            self.name,
            horizontal=True,
        )
        canvas = draw_sub_canvas(sub_canvas, canvas)
        return canvas

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""

        done1 = jnp.logical_or(
            state.x < -params.x_threshold,
            state.x > params.x_threshold,
        )
        done2 = jnp.logical_or(
            state.theta < -params.theta_threshold_radians,
            state.theta > params.theta_threshold_radians,
        )

        done_steps = state.time >= self.max_steps_in_episode

        done = jnp.logical_or(
            jnp.logical_or(done1, done2),
            done_steps
        )
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "CartPole"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.max_steps_in_episode

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(5)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(jnp.array(0, ), jnp.array(1, ), (256, 256, 3), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        high = jnp.array(
            [
                params.x_threshold * 2,
                jnp.finfo(jnp.float32).max,
                params.theta_threshold_radians * 2,
                jnp.finfo(jnp.float32).max,
            ]
        )
        return spaces.Dict(
            {
                "x": spaces.Box(-high[0], high[0], (), jnp.float32),
                "x_dot": spaces.Box(-high[1], high[1], (), jnp.float32),
                "theta": spaces.Box(-high[2], high[2], (), jnp.float32),
                "theta_dot": spaces.Box(-high[3], high[3], (), jnp.float32),
                "score": spaces.Discrete(self.max_steps_in_episode),
                "time": spaces.Discrete(self.max_steps_in_episode),
            }
        )


class CartPoleEasy(CartPole):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode=200, **kwargs)


class CartPoleMedium(CartPole):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode=400, **kwargs)


class CartPoleHard(CartPole):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode=600, **kwargs)


class NoisyCartPoleEasy(CartPole):
    def __init__(self, **kwargs):
        super().__init__(n_sigma=0.1, max_steps_in_episode=200, **kwargs)


class NoisyCartPoleMedium(CartPole):
    def __init__(self, **kwargs):
        super().__init__(n_sigma=0.2, max_steps_in_episode=200, **kwargs)


class NoisyCartPoleHard(CartPole):
    def __init__(self, **kwargs):
        super().__init__(n_sigma=0.3, max_steps_in_episode=200, **kwargs)
