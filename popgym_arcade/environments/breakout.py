"""JAX implementation of Breakout MinAtar environment."""
"""
Breakout
    Nonstationary: Random initial ball location or random block values (e.g.,1 vs 2 hit to break)
    POMDP: Hide blocks, paddle, or ball
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
    draw_rectangle,
    draw_circle,
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
) -> tuple[EnvState, jax.Array, jax.Array]:
    """Helper that steps the agent and checks boundary conditions."""
    # Update player position
    pos = (
        # Action left & border condition
        jnp.maximum(0, state.pos - 1) * (action == 1)
        # Action right & border condition
        + jnp.minimum(19, state.pos + 1) * (action == 3)
        # Don't move player if not l/r chosen
        + state.pos * jnp.logical_and(action != 1, action != 3)
    )

    # Update ball position - based on direction of movement
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

    # Boundary conditions for x position
    border_cond_x = jnp.logical_or(new_x < 0, new_x > 19)
    new_x = jax.lax.select(border_cond_x, (0 * (new_x < 0) + 20 * (new_x > 19)), new_x)
    # Reflect ball direction if bounced off at x border
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

    # Reflect ball direction if bounced off at y border
    border_cond1_y = new_y < 0
    new_y = jax.lax.select(border_cond1_y, 0, new_y)
    ball_dir = jax.lax.select(
        border_cond1_y, jnp.array([3, 2, 1, 0])[state.ball_dir], state.ball_dir
    )

    # 1st NASTY ELIF BEGINS HERE... = Brick collision
    strike_toggle = jnp.logical_and(
        1 - border_cond1_y, state.brick_map[new_y, new_x] == 1
    )
    strike_bool = jnp.logical_and((1 - state.strike), strike_toggle)
    # Variable reward system: harder-to-reach bricks give more reward
    # Rows 1-6 have bricks, with row 1 being hardest (top) and row 6 being easiest (bottom)
    # Total reward = 1.0 when all bricks cleared (6 rows * 20 columns * average reward)
    row_rewards = jnp.linspace(0.015, 0.005, 6)  # from hard (top) to easy (bottom)
    row_index = jnp.clip(new_y - 1, 0, 5)  # rows 1-6 map to indices 0-5
    reward += strike_bool * row_rewards[row_index]
    # next line wasn't used anywhere
    # strike = jax.lax.select(strike_toggle, strike_bool, False)

    brick_map = jax.lax.select(
        strike_bool, state.brick_map.at[new_y, new_x].set(0), state.brick_map
    )
    new_y = jax.lax.select(strike_bool, state.last_y, new_y)
    ball_dir = jax.lax.select(strike_bool, jnp.array([3, 2, 1, 0])[ball_dir], ball_dir)

    # 2nd NASTY ELIF BEGINS HERE... = Wall collision
    brick_cond = jnp.logical_and(1 - strike_toggle, new_y == 19)

    # # Spawn new bricks if there are no more around - everything is collected
    # spawn_bricks = jnp.logical_and(brick_cond, jnp.count_nonzero(brick_map) == 0)
    # brick_map = jax.lax.select(spawn_bricks, brick_map.at[1:7, :].set(1), brick_map)

    # Check if all bricks are cleared for termination
    all_bricks_cleared = jnp.count_nonzero(brick_map) == 0

    # Redirect ball because it collided with old player position
    ball_in_paddle_range = jnp.logical_and(state.ball_x >= state.pos, 
                                          state.ball_x < state.pos + paddle_width)
    redirect_ball1 = jnp.logical_and(brick_cond, ball_in_paddle_range)
    ball_dir = jax.lax.select(
        redirect_ball1, jnp.array([3, 2, 1, 0])[ball_dir], ball_dir
    )
    new_y = jax.lax.select(redirect_ball1, state.last_y, new_y)

    # Redirect ball because it collided with new player position
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
        jnp.logical_and(brick_cond, redirect_cond),  # Ball hits bottom
        all_bricks_cleared  # All bricks cleared
    )

    strike = jax.lax.select(1 - strike_toggle == 1, False, True)
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
    - Observation has dimensionality (10, 10, 4)
    - Actions are encoded as follows: ['n','l','r']
    """
    # color = {
    #     "red": jnp.array([255, 0, 0], dtype=jnp.float32),
    #     "dark_red": jnp.array([191, 26, 26], dtype=jnp.float32),
    #     "bright_red": jnp.array([255, 48, 71], dtype=jnp.float32),
    #     "black": jnp.array([0, 0, 0], dtype=jnp.float32),
    #     "white": jnp.array([255, 255, 255], dtype=jnp.float32),
    #     "metallic_gold": jnp.array([217, 166, 33], dtype=jnp.float32),
    #     "light_gray": jnp.array([245, 245, 245], dtype=jnp.float32),
    #     "light_blue": jnp.array([173, 217, 230], dtype=jnp.float32),
    #     "electric_blue": jnp.array([0, 115, 189], dtype=jnp.float32),
    #     "neon_pink": jnp.array([255, 105, 186], dtype=jnp.float32),
    #     "gray": jnp.array([128, 128, 128], dtype=jnp.float32),
    # }
    color = {
        "red": jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
        "dark_red": jnp.array([191/255.0, 26/255.0, 26/255.0], dtype=jnp.float32),
        "bright_red": jnp.array([1.0, 48/255.0, 71/255.0], dtype=jnp.float32),
        "black": jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        "white": jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
        "metallic_gold": jnp.array([217/255.0, 166/255.0, 33/255.0], dtype=jnp.float32),
        "light_gray": jnp.array([245/255.0, 245/255.0, 245/255.0], dtype=jnp.float32),
        "light_blue": jnp.array([173/255.0, 217/255.0, 230/255.0], dtype=jnp.float32),
        "electric_blue": jnp.array([0.0, 115/255.0, 189/255.0], dtype=jnp.float32),
        "neon_pink": jnp.array([1.0, 105/255.0, 186/255.0], dtype=jnp.float32),

        "yellow": jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32),
        # "gray": jnp.array([128/255.0, 128/255.0, 128/255.0], dtype=jnp.float32),
        "gray": jnp.array([119, 122, 127], dtype=jnp.float32) / 255.0,
        "ball_and_paddle": jnp.array([200/255.0, 72/255.0, 72/255.0], dtype=jnp.float32),
        "brick1": jnp.array([200, 72, 72], dtype=jnp.float32) / 255.0,
        "brick2": jnp.array([198, 108, 58], dtype=jnp.float32) / 255.0,
        "brick3": jnp.array([180, 122, 48], dtype=jnp.float32) / 255.0,
        "brick4": jnp.array([162, 162, 42], dtype=jnp.float32) / 255.0,
        "brick5": jnp.array([72, 160, 72], dtype=jnp.float32) / 255.0,
        "brick6": jnp.array([66, 72, 200], dtype=jnp.float32) / 255.0,
    }
    size = {
        256: {
            "canvas_size": 256,
            "small_canvas_size": 192,
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
            "small_canvas_size": 96,
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
        self.obs_shape = (20, 20, 4)
        # Full action set: ['n','l','u','r','d','f']
        self.full_action_set = jnp.array([0, 1, 2, 3, 4, 5])
        # Minimal action set: ['n', 'l', 'r']
        self.minimal_action_set = jnp.array([0, 1, 3])
        self.action_set = jnp.array([2, 4, 1, 3, 0])

        self.max_steps_in_episode = max_steps_in_episode
        self.reward_scale = 1.0 / max_steps_in_episode
        self.obs_size = obs_size
        self.partial_obs = partial_obs
        self.paddle_width = paddle_width
        

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
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
        state, new_x, new_y = step_agent(state, a)
        state, reward = step_ball_brick(state, new_x, new_y, params, self.paddle_width)

        # Add negative reward if game terminates due to ball hitting bottom
        # Penalty is proportional to fraction of bricks remaining
        # This ensures total reward stays in [-1, 1] range
        ball_hit_bottom = jnp.logical_and(state.terminal, jnp.count_nonzero(state.brick_map) > 0)
        negative_reward = jax.lax.select(ball_hit_bottom, 
                                        -jnp.count_nonzero(state.brick_map) / 120.0, 
                                        0.0)
        reward = reward + negative_reward
        jax.debug.print("Reward: {}", reward)
        # Check game condition & no. steps for termination condition
        state = state.replace(time=state.time + 1)
        done = self.is_terminal(state, params)
        state = state.replace(terminal=done)
        info = {"discount": self.discount(state, params)}
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
            ball_x=jnp.array([0, 19])[ball_start],
            ball_dir=jnp.array([2, 3])[ball_start],
            pos=9,
            brick_map=jnp.zeros((20, 20)).at[1:7, :].set(1),
            strike=False,
            last_y=jnp.array(13),
            last_x=jnp.array([0, 19])[ball_start],
            time=0,
            terminal=False,
            score=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Return observation from raw state trafo."""
        # obs = jnp.zeros(self.obs_shape, dtype=jnp.bool)
        # # Set the position of the player paddle, paddle, trail & brick map
        # obs = obs.at[19, state.pos, 0].set(True)
        # obs = obs.at[state.ball_y, state.ball_x, 1].set(True)
        # obs = obs.at[state.last_y, state.last_x, 2].set(True)
        # obs = obs.at[:, :, 3].set(state.brick_map.astype(jnp.bool))
        return self.render(state)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        done_steps = state.time >= params.max_steps_in_episode
        return jnp.logical_or(done_steps, state.terminal)
    

    @property
    def name(self) -> str:
        """Environment name."""
        return "Breakout"
     
    @functools.partial(jax.jit, static_argnums=(0,))
    def render(self, state: EnvState) -> jax.Array:
        # Initialize canvases
        canvas = jnp.zeros(
            (self.size[self.obs_size]["canvas_size"], self.size[self.obs_size]["canvas_size"], 3)
        ) + self.color["gray"]
        
        small_canvas = jnp.full(
            (self.size[self.obs_size]["small_canvas_size"], self.size[self.obs_size]["small_canvas_size"], 3),
            self.color["black"]
        )
        
        # Calculate scaling factors for rendering the 20x20 game grid on the small canvas
        cell_size = self.size[self.obs_size]["small_canvas_size"] // 20
        
        # Vectorized brick rendering - much more efficient than loops
        y_coords, x_coords = jnp.meshgrid(
            jnp.arange(self.size[self.obs_size]["small_canvas_size"]), 
            jnp.arange(self.size[self.obs_size]["small_canvas_size"]), 
            indexing='ij'
        )
        
        # Calculate which brick each pixel belongs to
        brick_y = y_coords // cell_size
        brick_x = x_coords // cell_size
        
        # Ensure indices are within bounds
        brick_y = jnp.clip(brick_y, 0, 19)
        brick_x = jnp.clip(brick_x, 0, 19)
        
        # Get brick values for each pixel and create mask
        brick_values = state.brick_map[brick_y, brick_x]
        brick_mask = brick_values == 1
        
        # Pre-compute brick colors array for all rows
        brick_colors = jnp.array([
            self.color["brick1"], self.color["brick2"], self.color["brick3"],
            self.color["brick4"], self.color["brick5"], self.color["brick6"],
        ])
        
        # Get color index for each pixel based on its brick row (rows 1-6 map to indices 0-5)
        color_indices = jnp.clip(brick_y - 1, 0, 5)
        pixel_colors = brick_colors[color_indices]
        
        # Apply brick colors where mask is true
        small_canvas = jnp.where(brick_mask[:, :, None], pixel_colors, small_canvas)

        # Efficient paddle rendering using vectorized rectangle drawing
        paddle_y_start = 19 * cell_size
        paddle_y_end = 20 * cell_size
        paddle_x_start = state.pos * cell_size
        paddle_x_end = jnp.minimum((state.pos + self.paddle_width) * cell_size, 
                                  self.size[self.obs_size]["small_canvas_size"])
        
        # Create paddle mask
        paddle_mask = jnp.logical_and(
            jnp.logical_and(y_coords >= paddle_y_start, y_coords < paddle_y_end),
            jnp.logical_and(x_coords >= paddle_x_start, x_coords < paddle_x_end)
        )
        
        # Apply paddle color
        small_canvas = jnp.where(paddle_mask[:, :, None], 
                                self.color["ball_and_paddle"], small_canvas)

        # Efficient ball rendering using vectorized circle
        ball_center_x = state.ball_x * cell_size + cell_size // 2
        ball_center_y = state.ball_y * cell_size + cell_size // 2
        ball_radius = cell_size // 3
        
        # Create ball mask using distance calculation
        ball_dist = jnp.sqrt((x_coords - ball_center_x) ** 2 + 
                           (y_coords - ball_center_y) ** 2)
        ball_mask = ball_dist <= ball_radius
        
        # POMDP: Hide ball when it's falling down (direction 2 or 3)
        ball_falling_down = jnp.logical_or(state.ball_dir == 2, state.ball_dir == 3)
        should_hide_ball = jnp.logical_and(self.partial_obs, ball_falling_down)
        
        # Apply ball color conditionally
        ball_mask = jnp.logical_and(ball_mask, jnp.logical_not(should_hide_ball))
        small_canvas = jnp.where(ball_mask[:, :, None], 
                                self.color["ball_and_paddle"], small_canvas)

        # Draw score and name (these are less performance critical)
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
        
        # Combine canvases
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
        return spaces.Box(0, 1, self.obs_shape)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "ball_y": spaces.Discrete(20),
                "ball_x": spaces.Discrete(20),
                "ball_dir": spaces.Discrete(10),
                "pos": spaces.Discrete(20),
                "brick_map": spaces.Box(0, 1, (20, 20)),
                "strike": spaces.Discrete(2),
                "last_y": spaces.Discrete(20),
                "last_x": spaces.Discrete(20),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "terminal": spaces.Discrete(2),
            }
        )

class BreakoutEasy(Breakout):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode=6000, paddle_width=4, **kwargs)


class BreakoutMedium(Breakout):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode=4000, paddle_width=3, **kwargs)


class BreakoutHard(Breakout):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode=3000, paddle_width=2, **kwargs)