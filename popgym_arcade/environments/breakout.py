"""JAX implementation of Breakout MinAtar environment."""
"""
Breakout
    Nonstationary: Random initial ball location
    POMDP: Hide last position/velocity and hide ball when falling back down
"""
import functools
from typing import Any

import jax
import jax.numpy as jnp
from chex import dataclass

from gymnax.environments import environment, spaces
from popgym_arcade.environments.draw_utils import (
    draw_number,
    draw_str,
    draw_sub_canvas,
)


@dataclass(frozen=True)
class EnvState(environment.EnvState):
    ball_y: jax.Array
    ball_x: jax.Array
    ball_dir: jax.Array
    pos: int
    brick_map: jax.Array
    strike: bool
    last_y: jax.Array
    last_x: jax.Array
    time: int
    terminal: bool
    score: int


@dataclass(frozen=True)
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 1000


def step_agent(
    state: EnvState,
    action: jax.Array,
    paddle_width: int = 1,
) -> tuple[EnvState, jax.Array, jax.Array]:
    """Helper that steps the agent and checks boundary conditions."""
    max_pos = 10 - paddle_width
    pos = (
        jnp.maximum(0, state.pos - 1) * (action == 1)
        + jnp.minimum(max_pos, state.pos + 1) * (action == 3)
        + state.pos * jnp.logical_and(action != 1, action != 3)
    )

    last_x = state.ball_x
    last_y = state.ball_y
    new_x = (
        (state.ball_x - 1) * (state.ball_dir == 0)
        + (state.ball_x + 1) * (state.ball_dir == 1)
        + (state.ball_x + 1) * (state.ball_dir == 2)
        + (state.ball_x - 1) * (state.ball_dir == 3)
    )
    new_y = (
        (state.ball_y - 1) * (state.ball_dir == 0)
        + (state.ball_y - 1) * (state.ball_dir == 1)
        + (state.ball_y + 1) * (state.ball_dir == 2)
        + (state.ball_y + 1) * (state.ball_dir == 3)
    )

    border_cond_x = jnp.logical_or(new_x < 0, new_x > 9)
    new_x = jax.lax.select(border_cond_x, (0 * (new_x < 0) + 9 * (new_x > 9)), new_x)

    ball_dir = jax.lax.select(
        border_cond_x, jnp.array([1, 0, 3, 2])[state.ball_dir], state.ball_dir
    )
    return (
        state.replace(
            pos=pos,
            last_x=last_x,
            last_y=last_y,
            ball_dir=ball_dir,
        ),
        new_x,
        new_y,
    )


def step_ball_brick(
    state: EnvState, new_x: jax.Array, new_y: jax.Array, params: EnvParams, paddle_width: int = 1
) -> tuple[EnvState, jax.Array]:
    """Helper that computes reward and termination cond. from brickmap."""

    reward = 0

    border_cond1_y = new_y < 0
    new_y = jax.lax.select(border_cond1_y, 0, new_y)
    ball_dir = jax.lax.select(
        border_cond1_y, jnp.array([3, 2, 1, 0])[state.ball_dir], state.ball_dir
    )

    strike_toggle = jnp.logical_and(
        1 - border_cond1_y, state.brick_map[new_y, new_x] == 1
    )
    strike_bool = jnp.logical_and((1 - state.strike), strike_toggle)

    row_rewards = jnp.linspace(0.015, 0.005, 6)
    row_rewards = row_rewards * (0.1 / jnp.sum(row_rewards))
    row_index = jnp.clip(new_y - 1, 0, 5) 
    reward += strike_bool * row_rewards[row_index]

    brick_map = jax.lax.select(
        strike_toggle, state.brick_map.at[new_y, new_x].set(0), state.brick_map
    )
    new_y = jax.lax.select(strike_bool, state.last_y, new_y)
    ball_dir = jax.lax.select(strike_bool, jnp.array([3, 2, 1, 0])[ball_dir], ball_dir)

    brick_cond = jnp.logical_and(1 - strike_toggle, new_y == 19)

    all_bricks_cleared = jnp.count_nonzero(brick_map) == 0

    ball_in_paddle_range = jnp.logical_and(state.ball_x >= state.pos, 
                                          state.ball_x < state.pos + paddle_width)
    redirect_ball1 = jnp.logical_and(brick_cond, ball_in_paddle_range)
    ball_dir = jax.lax.select(
        redirect_ball1, jnp.array([3, 2, 1, 0])[ball_dir], ball_dir
    )
    new_y = jax.lax.select(redirect_ball1, state.last_y, new_y)

    redirect_ball2a = jnp.logical_and(brick_cond, 1 - redirect_ball1)
    new_ball_in_paddle_range = jnp.logical_and(new_x >= state.pos, 
                                              new_x < state.pos + paddle_width)
    redirect_ball2 = jnp.logical_and(redirect_ball2a, new_ball_in_paddle_range)
    ball_dir = jax.lax.select(
        redirect_ball2, jnp.array([2, 3, 0, 1])[ball_dir], ball_dir
    )
    new_y = jax.lax.select(redirect_ball2, state.last_y, new_y)
    redirect_cond = jnp.logical_and(1 - redirect_ball1, 1 - redirect_ball2)
    terminal = jnp.logical_or(
        jnp.logical_and(brick_cond, redirect_cond),
        all_bricks_cleared
    )

    strike = strike_toggle
    return (
        state.replace(
            ball_dir=ball_dir,
            brick_map=brick_map,
            strike=strike,
            ball_x=new_x,
            ball_y=new_y,
            terminal=terminal,
            score=state.score + strike_bool.astype(jnp.int32),
        ),
        reward,
    )



class Breakout(environment.Environment[EnvState, EnvParams]):
    """JAX implementation of Breakout MinAtar environment.


    Source:
    github.com/kenjyoung/MinAtar/blob/master/minatar/environments/breakout.py


    ENVIRONMENT DESCRIPTION - 'Breakout-MinAtar'
    - Player controls paddle on bottom of screen.
    - Must bounce ball to break 6 rows of bricks along top of screen.
    - Variable reward system: top bricks (harder to reach) give more reward than bottom bricks.
    - Total reward = +1.0 when all bricks cleared, death penalty = -(fraction of bricks left).
    - Game terminates when all bricks are cleared or ball hits bottom.
    - Ball travels only along diagonals, when paddle/wall hit it bounces off
    - Termination if ball hits bottom of screen.
    - Ball direction is indicated by a trail channel.
    - There is no difficulty increase.
    - Channels are encoded as follows: 'paddle':0, 'ball':1, 'trail':2, 'brick':3
    - Observation has dimensionality (20, 10, 4)
    - Actions are encoded as follows: ['n','l','r']
    """
    color = {
        "red": jnp.array([255, 0, 0], dtype=jnp.uint8),
        "dark_red": jnp.array([191, 26, 26], dtype=jnp.uint8),
        "bright_red": jnp.array([255, 48, 71], dtype=jnp.uint8),
        "black": jnp.array([0, 0, 0], dtype=jnp.uint8),
        "white": jnp.array([255, 255, 255], dtype=jnp.uint8),
        "metallic_gold": jnp.array([217, 166, 33], dtype=jnp.uint8),
        "light_gray": jnp.array([245, 245, 245], dtype=jnp.uint8),
        "light_blue": jnp.array([173, 217, 230], dtype=jnp.uint8),
        "electric_blue": jnp.array([0, 115, 189], dtype=jnp.uint8),
        "neon_pink": jnp.array([255, 105, 186], dtype=jnp.uint8),
        "yellow": jnp.array([255, 255, 0], dtype=jnp.uint8),
        "gray": jnp.array([119, 122, 127], dtype=jnp.uint8),
        "ball_and_paddle": jnp.array([200, 72, 72], dtype=jnp.uint8),
        "ball_trail": jnp.array([255, 50, 50], dtype=jnp.uint8),  # Dimmer version for trail
        "brick1": jnp.array([200, 72, 72], dtype=jnp.uint8),
        "brick2": jnp.array([198, 108, 58], dtype=jnp.uint8),
        "brick3": jnp.array([255, 122, 48], dtype=jnp.uint8),
        "brick4": jnp.array([162, 162, 42], dtype=jnp.uint8),
        "brick5": jnp.array([72, 255, 72], dtype=jnp.uint8),
        "brick6": jnp.array([66, 72, 200], dtype=jnp.uint8),
    }
    size = {
        256: {
            "canvas_size": 256,
            "small_canvas_size": 200,
            "name_pos": {
                "top_left": (0, 231),
                "bottom_right": (256, 256),
            },
            "score": {
                "top_left": (86, 2),
                "bottom_right": (171, 30),
            },
        },
        128: {
            "canvas_size": 128,
            "small_canvas_size": 100,
            "name_pos": {
                "top_left": (0, 115),
                "bottom_right": (128, 128),
            },
            "score": {
                "top_left": (43, 1),
                "bottom_right": (85, 15),
            },
        },
    }

    def __init__(self, obs_size: int = 128, partial_obs=False, paddle_width=3, max_steps_in_episode=1000):
        super().__init__()
        self.obs_shape = (20, 10, 4)
        self.full_action_set = jnp.array([0, 1, 2, 3, 4, 5])
        self.minimal_action_set = jnp.array([0, 1, 3])
        self.action_set = jnp.array([2, 4, 1, 3, 0])

        self.max_steps_in_episode = max_steps_in_episode
        self.reward_scale = 1.0 / max_steps_in_episode
        self.obs_size = obs_size
        self.partial_obs = partial_obs
        self.paddle_width = paddle_width
        

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=self.max_steps_in_episode)

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jnp.ndarray, jnp.ndarray, dict[Any, Any]]:
        """Perform single timestep state transition."""
        a = self.action_set[action]
        state, new_x, new_y = step_agent(state, a, self.paddle_width)
        state, reward = step_ball_brick(state, new_x, new_y, params, self.paddle_width)

        ball_hit_bottom = jnp.logical_and(state.terminal, jnp.count_nonzero(state.brick_map) > 0)
        negative_reward = jax.lax.select(ball_hit_bottom, 
                                        -jnp.count_nonzero(state.brick_map) / 60.0, 
                                        0.0)
        reward = reward + negative_reward
        state = state.replace(time=state.time + 1)
        done = self.is_terminal(state, params)
        state = state.replace(terminal=done)
        truncated = state.time >= params.max_steps_in_episode

        info = {"discount": self.discount(state, params),
                "terminated": state.terminal,
                "truncated": truncated}
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward.astype(jnp.float32),
            done,
            info,
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        ball_start = jax.random.choice(key, jnp.array([0, 1]), shape=())
        state = EnvState(
            ball_y=jnp.array(13),
            ball_x=jnp.array([0, 9])[ball_start],
            ball_dir=jnp.array([0, 1])[ball_start],
            pos=5,
            brick_map=jnp.zeros((20, 10)).at[1:7, :].set(1),
            strike=False,
            last_y=jnp.array(13),
            last_x=jnp.array([0, 9])[ball_start],
            time=0,
            terminal=False,
            score=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Return observation from raw state trafo."""
        return self.render(state)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        truncated = state.time >= params.max_steps_in_episode
        return jnp.logical_or(truncated, state.terminal)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Breakout"
     
    @functools.partial(jax.jit, static_argnums=(0,))
    def render(self, state: EnvState) -> jax.Array:
        canvas = jnp.zeros(
            (self.size[self.obs_size]["canvas_size"], self.size[self.obs_size]["canvas_size"], 3), dtype=jnp.uint8
        ) + self.color["gray"]
        
        small_canvas = jnp.full(
            (self.size[self.obs_size]["small_canvas_size"], self.size[self.obs_size]["small_canvas_size"], 3),
            self.color["black"]
        )

        grid_height = 20
        grid_width = 10
        cell_height = self.size[self.obs_size]["small_canvas_size"] // grid_height
        cell_width = self.size[self.obs_size]["small_canvas_size"] // grid_width

        y_coords, x_coords = jnp.meshgrid(
            jnp.arange(self.size[self.obs_size]["small_canvas_size"]), 
            jnp.arange(self.size[self.obs_size]["small_canvas_size"]), 
            indexing='ij'
        )

        brick_y = jnp.minimum(jnp.floor(y_coords / cell_height).astype(jnp.int32), grid_height - 1)
        brick_x = jnp.minimum(jnp.floor(x_coords / cell_width).astype(jnp.int32), grid_width - 1)

        brick_values = state.brick_map[brick_y, brick_x]
        brick_mask = brick_values == 1

        brick_colors = jnp.array([
            self.color["brick1"], self.color["brick2"], self.color["brick3"],
            self.color["brick4"], self.color["brick5"], self.color["brick6"],
        ])

        color_indices = jnp.clip(brick_y - 1, 0, 5)
        pixel_colors = brick_colors[color_indices]

        small_canvas = jnp.where(brick_mask[:, :, None], pixel_colors, small_canvas)

        paddle_y_start = 19 * cell_height
        paddle_y_end = 20 * cell_height
        paddle_x_start = state.pos * cell_width
        paddle_x_end = jnp.minimum((state.pos + self.paddle_width) * cell_width, 
                                  self.size[self.obs_size]["small_canvas_size"])
        
        paddle_mask = jnp.logical_and(
            jnp.logical_and(y_coords >= paddle_y_start, y_coords < paddle_y_end),
            jnp.logical_and(x_coords >= paddle_x_start, x_coords < paddle_x_end)
        )
        
        small_canvas = jnp.where(paddle_mask[:, :, None], 
                                self.color["ball_and_paddle"], small_canvas)

        ball_center_x = state.ball_x * cell_width + cell_width // 2
        ball_center_y = state.ball_y * cell_height + cell_height // 2
        ball_radius = jnp.floor(jnp.minimum(cell_width, cell_height) // 3)

        ball_dist = jnp.sqrt((x_coords - ball_center_x) ** 2 + 
                           (y_coords - ball_center_y) ** 2)
        ball_mask = ball_dist <= ball_radius

        ball_falling_down = jnp.logical_or(state.ball_dir == 2, state.ball_dir == 3)
        should_hide_ball = jnp.logical_and(self.partial_obs, ball_falling_down)

        ball_mask = jnp.logical_and(ball_mask, jnp.logical_not(should_hide_ball))
        small_canvas = jnp.where(ball_mask[:, :, None], 
                                self.color["ball_and_paddle"], small_canvas)

        # Draw ball trail/track to show velocity direction
        should_show_trail = jnp.logical_not(self.partial_obs)

        trail_center_x = state.last_x * cell_width + cell_width // 2
        trail_center_y = state.last_y * cell_height + cell_height // 2
        trail_radius = jnp.minimum(cell_width, cell_height) // 4  # Smaller than ball
        
        trail_dist = jnp.sqrt((x_coords - trail_center_x) ** 2 + 
                             (y_coords - trail_center_y) ** 2)
        trail_mask = jnp.logical_and(trail_dist <= trail_radius, should_show_trail)
        small_canvas = jnp.where(trail_mask[:, :, None], self.color["ball_trail"], small_canvas)

        canvas = draw_number(
            self.size[self.obs_size]["score"]["top_left"],
            self.size[self.obs_size]["score"]["bottom_right"],
            self.color["bright_red"],
            canvas,
            state.score,
        )

        canvas = draw_str(
            self.size[self.obs_size]["name_pos"]["top_left"],
            self.size[self.obs_size]["name_pos"]["bottom_right"],
            self.color["yellow"],
            canvas,
            self.name,
        )
        
        return draw_sub_canvas(small_canvas, canvas)



    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(self.action_set))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 255, (self.obs_size, self.obs_size, 3), dtype=jnp.uint8)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "ball_y": spaces.Discrete(20),
                "ball_x": spaces.Discrete(10),
                "ball_dir": spaces.Discrete(10),
                "pos": spaces.Discrete(10),
                "brick_map": spaces.Box(0, 1, (20, 10)),
                "strike": spaces.Discrete(2),
                "last_y": spaces.Discrete(20),
                "last_x": spaces.Discrete(10),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "terminal": spaces.Discrete(2),
            }
        )

class BreakoutEasy(Breakout):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode=2000, paddle_width=6, **kwargs)


class BreakoutMedium(Breakout):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode=2000, paddle_width=5, **kwargs)


class BreakoutHard(Breakout):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode=2000, paddle_width=4, **kwargs)
