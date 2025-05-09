from typing import Any, Dict, Optional, Tuple, Union

import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
import functools
from gymnax.environments import environment, spaces
from popgym_arcade.environments.draw_utils import (
                                            draw_rectangle,
                                            draw_number,
                                            draw_str,
                                            draw_grid,
                                            draw_sub_canvas)

@struct.dataclass
class EnvState(environment.EnvState):
    matrix_state: chex.Array
    color_indexes: chex.Array
    x: int
    xp: int
    over: int
    time: int
    score: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    pass


class Skittles(environment.Environment[EnvState, EnvParams]):
    """
    Jax compilable environment for the Swimming Dragon.
    
    ### Description
    In Swimming Dragon, the agent is tasked with avoiding enemies that are moving down the screen. 
    The agent can move left or right to dodge the enemies. The goal is to survive as long as possible without being hit by an enemy.
    There are three difficulties: easy, medium, and hard. Each difficulty has a different grid size and maximum steps in an episode.
    Easy: 8x8 grid, 200 steps
    Medium: 10x10 grid, 400 steps
    Hard: 12x12 grid, 600 steps
    The episode ends when the agent is hit by an enemy or the maximum number of steps is reached.

    ### Board Elements
    - 0: Empty
    - 1: Enemy
    The player can only move within the last row of the matrix, and their position is indicated by the column index.

    ### Action Space
    | Action | Description                         |
    |--------|-------------------------------------|
    | 0      | Up (No-op)                          |
    | 1      | Down (No-op)                        |
    | 2      | Left                                |
    | 3      | Right                               |
    | 4      | Fire (No-op)                        |
    
    ### Observation Space

    ### Reward
    - Reward Scale: 1.0 / max_steps_in_episode

    ### Termination & Truncation
    The episode ends when the agent is hit by an enemy or 
     the maximum number of steps is reached.

    ### Args
    - max_steps_in_episode: Maximum number of steps in an episode.
    - grid_size: Size of the grid (number of rows and columns).
    
    TODO:
        - obs_size: Size of the observation space.
        - partial_obs: Whether to use partial observation or not.

    """

    render_256x = {
        # parameters for rendering (256, 256, 3) canvas
        "size": 256,
        "clr": jnp.array([0.0, 0.0, 0.0]),
        "sub_size": {
            8: 186,
            10: 192,
            12: 182,
        },
        # parameters for rendering sub canvas
        "sub_clr": jnp.array([0.0, 0.0, 0.0]),
        
        # parameters for current action position
        "action_clr": jnp.array([1.0, 1.0, 1.0]),
        
        # parameters for rendering enemy
        # "enemy_clr": jnp.array([0.29, 0.84, 0.97]),
        # rainbow color
        "red": jnp.array([1.0, 0.0, 0.0]),
        "orange": jnp.array([1.0, 0.5, 0.0]),
        "yellow": jnp.array([1.0, 1.0, 0.0]),
        "green": jnp.array([0.0, 1.0, 0.0]),
        "blue": jnp.array([0.0, 0.68, 0.94]),
        "indigo": jnp.array([0.031, 0.44, 0.73]),
        "violet": jnp.array([0.4, 0.19, 0.56]),

        # parameters for rendering grids
        "grid_px": 2,
        "grid_clr": jnp.array([1.0, 1.0, 1.0]),

        # parameters for rendering score
        "sc_t_l": (86, 2),
        "sc_b_r": (171, 30),
        "sc_clr": jnp.array([0.0, 1.0, 0.5]),
        
        # parameters for rendering env name
        "env_t_l": (0, 231),
        "env_b_r": (256, 256),
        "env_clr": jnp.array([1.0, 0.96, 0.0]),
    }

    render_128x = {
        # parameters for rendering (128, 128, 3) canvas
        "size": 128,
        "clr": jnp.array([0.0, 0.0, 0.0]),
        "sub_size": {
            8: 90,
            10: 92,
            12: 98,
        },
        # parameters for rendering sub canvas
        "sub_clr": jnp.array([0.0, 0.0, 0.0]),
        
        # parameters for current action position
        "action_clr": jnp.array([0.93, 0.53, 0.18]),
        
        # parameters for rendering enemy
        # "enemy_clr": jnp.array([0.29, 0.84, 0.97]),
        # rainbow color
        "red": jnp.array([1.0, 0.0, 0.0]),
        "orange": jnp.array([1.0, 0.5, 0.0]),
        "yellow": jnp.array([1.0, 1.0, 0.0]),
        "green": jnp.array([0.0, 1.0, 0.0]),
        "blue": jnp.array([0.0, 0.0, 1.0]),
        "indigo": jnp.array([0.29, 0.0, 0.51]),
        "violet": jnp.array([0.56, 0.0, 1.0]),

        # parameters for rendering grids
        "grid_px": 2,
        "grid_clr": jnp.array([1.0, 1.0, 1.0]),

        # parameters for rendering score
        "sc_t_l": (43, 1),
        "sc_b_r": (85, 15),
        "sc_clr": jnp.array([0.0, 1.0, 0.5]),
        
        # parameters for rendering env name
        "env_t_l": (0, 115),
        "env_b_r": (128, 128),
        "env_clr": jnp.array([1.0, 0.96, 0.0]),
    }
    render_mode = {
        256: render_256x,
        128: render_128x,
    }

    def __init__(
            self, 
            max_steps_in_episode: int, 
            grid_size: int,
            obs_size: int = 128,
            partial_obs = False,
            enemy_num: int = 2,
            ):
        super().__init__()
        self.obs_size = obs_size
        self.max_steps_in_episode = max_steps_in_episode
        self.reward_scale = (1.0 / max_steps_in_episode)
        self.grid_size = grid_size
        self.partial_obs = partial_obs
        self.enemy_num = enemy_num


    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Perform a step in the environment."""
        key, newkey = jax.random.split(key)
        xp = state.xp
        over = state.over
        x = state.x
        x = jnp.clip(jnp.where(action == 2, x-1, x), 0, self.grid_size-1)
        x = jnp.clip(jnp.where(action == 3, x+1, x), 0, self.grid_size-1)
        
        matrix_state = state.matrix_state
        xp = matrix_state[self.grid_size-1, x]
        over = xp 
        # each row moves down by one, and the first row is replaced by a new enemy row
        matrix_state = matrix_state.at[1:self.grid_size, :].set(matrix_state[0:self.grid_size-1, :])

        newkey, enemy_key = jax.random.split(newkey)
        enemy_new = self.random_enemy(enemy_key)
        enemy_new = jnp.squeeze(enemy_new)

        matrix_state = matrix_state.at[0, :].set(enemy_new)
        xp = matrix_state[self.grid_size-1, x]

        new_color_idx = (state.color_indexes[0] + 1) % 7

        new_color_indexes = jnp.roll(state.color_indexes, shift=1)
        new_color_indexes = new_color_indexes.at[0].set(new_color_idx)
        state = EnvState(
            matrix_state = matrix_state,
            x = x,
            xp = matrix_state[self.grid_size-1, x],
            over = over,
            time = state.time + 1,
            score = state.score + 1,
            color_indexes = new_color_indexes,
        )
        
        done = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            jnp.array(self.reward_scale),
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset the environment to an initial state."""
        key, subkey1 = jax.random.split(key)
        matrix_state = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        x = jax.random.randint(subkey1, shape=(), minval=0, maxval=self.grid_size).astype(jnp.int32) 

        state = EnvState(
            matrix_state = matrix_state,
            color_indexes=jnp.zeros(self.grid_size).at[0].set(0),
            x = x,
            xp = matrix_state[self.grid_size-1, x],
            time = 0,
            score = 0,
            over = 0,
        )
        return self.get_obs(state), state

    def random_enemy(self, key) -> jnp.ndarray:
        """Generate a random enemy row."""
        key, subkey2 = jax.random.split(key)
        enemy_row = jnp.zeros(self.grid_size, dtype=jnp.int32)
        indices = jax.random.choice(subkey2, jnp.arange(self.grid_size), shape=(self.enemy_num,), replace=False)
        enemy_row = enemy_row.at[indices].set(1)
        enemy_row = enemy_row.reshape(1, -1)
        return enemy_row

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        return self.render(state)
        # return state.matrix_state

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check if the episode is done."""
        done_crash = state.xp + state.over
        done_steps = state.time >= self.max_steps_in_episode
        done = jnp.logical_or(done_crash, done_steps)
        return done

    @functools.partial(jax.jit, static_argnums=(0,))
    def render(self, state: EnvState) -> chex.Array:
        """Render the current state of the environment."""
        render_config = self.render_mode[self.obs_size]
        board_size = self.grid_size

        grid_px = render_config["grid_px"]
        sub_size = render_config["sub_size"][board_size]
        square_size = (sub_size - (board_size + 1) * grid_px) // board_size

        # Generate grid coordinates using meshgrid
        x_coords, y_coords = jnp.arange(board_size), jnp.arange(board_size)
        xx, yy = jnp.meshgrid(x_coords, y_coords,  indexing='ij')
        # top_left_x = grid_px + xx * (square_size + grid_px)
        # top_left_y = grid_px + yy * (square_size + grid_px)
        top_left_x = grid_px + yy * (square_size + grid_px)
        top_left_y = grid_px + xx * (square_size + grid_px)
        all_top_left = jnp.stack([top_left_x, top_left_y], axis=-1)
        all_bottom_right = all_top_left + square_size

        # Initialize canvases
        canvas = jnp.full(
            (render_config["size"],) * 2 + (3,),
            render_config["clr"]
        )
        sub_canvas = jnp.full(
            (sub_size, sub_size, 3), 
            render_config["sub_clr"]
        )

        action_x, action_y = board_size - 1, state.x

        # Precompute board values
        board_flat = state.matrix_state.flatten()

        # Define cell rendering function
        def render_cell(pos, canvas):
            x = pos // board_size
            y = pos % board_size
            tl = all_top_left[x, y]
            br = all_bottom_right[x, y]
            cell_val = board_flat[pos]

            # Rainbow color list
            rainbow_colors = jnp.array([
                render_config["red"],
                render_config["orange"],
                render_config["yellow"],
                render_config["green"],
                render_config["blue"],
                render_config["indigo"],
                render_config["violet"],
            ])
            color_idx = jnp.int32(state.color_indexes[x])
            enemy_color = rainbow_colors[color_idx % len(rainbow_colors)] 
            # Draw enemy block if cell_val is 1
            canvas = lax.cond(
                cell_val == 1,
                lambda: draw_rectangle(tl, br, enemy_color, canvas),
                lambda: canvas,
            )

            return canvas

        # Partial rendering: only the action cell + the first row
        def _render_partial(sub_canvas):
            pos = action_x * board_size + action_y
            sub_canvas = render_cell(pos, sub_canvas)

            first_row_indices = jnp.arange(board_size)
            def render_first_row_cell(idx, canvas):
                return render_cell(idx, canvas)
            sub_canvas = jax.lax.fori_loop(
                0, board_size, 
                lambda i, c: render_first_row_cell(first_row_indices[i], c), 
                sub_canvas
            )
            return sub_canvas

        # Full rendering: all cells
        def _render_full(sub_canvas):
            cell_indices = jnp.arange(board_size**2)
            updated = jax.vmap(render_cell, in_axes=(0, None))(cell_indices, sub_canvas)
            return jnp.max(updated, axis=0)
    
        # Conditional rendering logic
        sub_canvas = lax.cond(
            state.time == 0,
            lambda: _render_full(sub_canvas),
            lambda: lax.cond(
                self.partial_obs,
                lambda: _render_partial(sub_canvas),
                lambda: _render_full(sub_canvas),
            )
        )

        action_tl = all_top_left[action_x, action_y]
        action_br = all_bottom_right[action_x, action_y]
        sub_canvas = draw_rectangle(
            action_tl,
            action_br,
            render_config["action_clr"],
            sub_canvas
        )

        # Draw grid lines
        sub_canvas = draw_grid(
            square_size, 
            grid_px, 
            render_config["grid_clr"],
            sub_canvas
        )

        # Draw score on canvas
        canvas = draw_number(
            render_config["sc_t_l"],
            render_config["sc_b_r"],
            render_config["sc_clr"],
            canvas, 
            state.score
        )

        # Draw environment name
        canvas = draw_str(
            render_config["env_t_l"],
            render_config["env_b_r"],
            render_config["env_clr"],
            canvas, 
            self.name
        )

        # Merge sub-canvas onto main canvas
        return draw_sub_canvas(sub_canvas, canvas)

    @property
    def name(self) -> str:
        return "Skittles"

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(5)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(jnp.zeros((0,)), jnp.ones((1,)), (self.obs_size, self.obs_size, 3), dtype=jnp.float32)


class SkittlesEasy(Skittles):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode = 200, grid_size = 8, enemy_num=1, **kwargs)


class SkittlesMedium(Skittles):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode = 400, grid_size = 10, enemy_num=2, **kwargs)


class SkittlesHard(Skittles):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode = 600, grid_size = 12, enemy_num=2, **kwargs)