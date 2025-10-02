from typing import Any, Dict, Optional, Tuple, Union

import chex
from chex import dataclass
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

@dataclass(frozen=True)
class EnvState(environment.EnvState):
    matrix_state: chex.Array
    color_indexes: chex.Array
    x: int
    xp: int
    over: int
    time: int
    score: int

@dataclass(frozen=True)
class EnvParams(environment.EnvParams):
    pass


class Skittles(environment.Environment[EnvState, EnvParams]):
    """
    Jax compilable environment for the Swimming Dragon.
    
    ### Description
    In Swimming Dragon, the agent is tasked with avoiding enemies that fall from the top of the screen.
    The agent can move left or right to dodge the enemies. The goal is to survive as long as possible without being hit by an enemy.
    There are three difficulties: easy, medium, and hard. Each difficulty has a different grid size and maximum steps in an episode.
    Easy: 8x8 grid, agent's goal is to survive 200 steps, have 1 enemy in each row
    Medium: 10x10 grid, agent's goal is to survive 400 steps, have 2 enemies in each row
    Hard: 12x12 grid, agent's goal is to survive 600 steps, have 2 enemies in each row
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
    OBS_SIZE can be either 128 or 256. The observation is a rendered image of the state with shape (OBS_SIZE, OBS_SIZE, 3).
    The image contains:
        - The current action position (only move on the last row of the matrix), with white color.
        - The enemies falling down from the top of the screen, with rainbow colors.
        - The grid lines, with white color.
        - The score, with green color.
        - The environment name, with yellow color.

    ### Reward
    - Reward Scale: 1.0 / max_steps_in_episode

    ### Termination & Truncation
    The episode ends when the agent is hit by an enemy or 
     the maximum number of steps is reached.

    ### Args
    - max_steps_in_episode: Maximum number of steps in an episode.
    - grid_size: Size of the grid (number of rows and columns).
    - obs_size: Size of the observation space, choose between 128 and 256.
    - partial_obs: Whether to use partial observation or not.
    - enemy_num: Number of enemies in the difficulty level.
    - enemy_spawn_width: Number of columns at the top row where enemies can spawn (<= grid_size). Smaller makes a narrow lane.
    - enemy_spawn_offset: Left offset (column index) of the spawn region.

    """

    render_common = {
        "clr": jnp.array([0, 0, 0], dtype=jnp.uint8),
        # parameters for rendering sub canvas
        "sub_clr": jnp.array([0, 0, 0], dtype=jnp.uint8),
        # parameters for current action position
        "action_clr": jnp.array([255, 255, 255], dtype=jnp.uint8),
        # parameters for rendering enemy
        # rainbow color
        "red": jnp.array([255, 0, 0], dtype=jnp.uint8),
        "orange": jnp.array([255, 127, 0], dtype=jnp.uint8),
        "yellow": jnp.array([255, 255, 0], dtype=jnp.uint8),
        "green": jnp.array([0, 255, 0], dtype=jnp.uint8),
        "blue": jnp.array([0, 0, 255], dtype=jnp.uint8),
        "indigo": jnp.array([74, 214, 247], dtype=jnp.uint8),
        "violet": jnp.array([125, 0, 237], dtype=jnp.uint8),
        # parameters for rendering grids
        "grid_clr": jnp.array([255, 255, 255], dtype=jnp.uint8),
        # parameters for rendering score
        "sc_clr": jnp.array([0, 255, 128], dtype=jnp.uint8),
        # parameters for rendering env name
        "env_clr": jnp.array([255, 245, 0], dtype=jnp.uint8),
    }

    render_256x = {
        **render_common,
        # parameters for rendering (256, 256, 3) canvas
        "size": 256,
        "sub_size": {
            8: 186,
            10: 192,
            12: 182,
        },
        # parameters for rendering grids
        "grid_px": 2,
        # parameters for rendering score
        "sc_t_l": (86, 2),
        "sc_b_r": (171, 30),
        # parameters for rendering env name
        "env_t_l": (0, 231),
        "env_b_r": (256, 256),
    }

    render_128x = {
        **render_common,
        # parameters for rendering (128, 128, 3) canvas
        "size": 128,
        "sub_size": {
            8: 90,
            10: 92,
            12: 98,
        },
        # parameters for rendering grids
        "grid_px": 2,
        # parameters for rendering score
        "sc_t_l": (43, 1),
        "sc_b_r": (85, 15),
        # parameters for rendering env name
        "env_t_l": (0, 115),
        "env_b_r": (128, 128),
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
            p: float = 0.5,
            enemy_spawn_width: Optional[int] = None,
            enemy_spawn_offset: int = 0,
            ):
        super().__init__()
        self.obs_size = obs_size
        self.max_steps_in_episode = max_steps_in_episode
        self.reward_scale = (1.0 / max_steps_in_episode)
        self.grid_size = grid_size
        self.partial_obs = partial_obs

        enemy_spawn_offset = int(max(0, min(enemy_spawn_offset, grid_size - 1)))
        enemy_spawn_width = int(max(1, min(enemy_spawn_width, grid_size - enemy_spawn_offset)))
        self.enemy_spawn_width = enemy_spawn_width
        self.enemy_spawn_offset = enemy_spawn_offset

        self.enemy_num = int(min(enemy_num, self.enemy_spawn_width))
        self.p = p


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
        min_x = jnp.int32(self.enemy_spawn_offset)
        max_x = jnp.int32(self.enemy_spawn_offset + self.enemy_spawn_width - 1)
        x = jnp.clip(jnp.where(action == 2, x - 1, x), min_x, max_x)
        x = jnp.clip(jnp.where(action == 3, x + 1, x), min_x, max_x)

        matrix_state = state.matrix_state
        xp = matrix_state[self.grid_size - 1, x]
        over = xp

        matrix_state = matrix_state.at[1:self.grid_size, :].set(
            matrix_state[0:self.grid_size - 1, :]
        )

        newkey, enemy_key = jax.random.split(newkey)
        enemy_new = self.random_enemy(enemy_key)
        enemy_new = jnp.squeeze(enemy_new)

        matrix_state = matrix_state.at[0, :].set(enemy_new)
        xp = matrix_state[self.grid_size - 1, x]

        new_color_idx = (state.color_indexes[0] + 1) % 7

        new_color_indexes = jnp.roll(state.color_indexes, shift=1)
        new_color_indexes = new_color_indexes.at[0].set(new_color_idx)
        state = EnvState(
            matrix_state=matrix_state,
            x=x,
            xp=matrix_state[self.grid_size - 1, x],
            over=over,
            time=state.time + 1,
            score=state.score + 1,
            color_indexes=new_color_indexes,
        )

        done = self.is_terminal(state, params)
        terminated = state.xp + state.over
        truncated = state.time >= self.max_steps_in_episode
        
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            jnp.array(self.reward_scale),
            done,
            {"discount": self.discount(state, params),
             "terminated": terminated,
             "truncated": truncated},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset the environment to an initial state."""
        key, subkey1 = jax.random.split(key)
        matrix_state = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        x = jax.random.randint(
            subkey1,
            shape=(),
            minval=self.enemy_spawn_offset,
            maxval=self.enemy_spawn_offset + self.enemy_spawn_width,
        ).astype(jnp.int32)

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
        spawn_cols = jnp.arange(
            self.enemy_spawn_offset, self.enemy_spawn_offset + self.enemy_spawn_width
        )
        indices = jax.random.choice(
            subkey2, spawn_cols, shape=(self.enemy_num,), replace=False
        )
        enemy_row = enemy_row.at[indices].set(1)
        enemy_row = enemy_row.reshape(1, -1)
        return enemy_row

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        return self.render(state)
        # return state.matrix_state

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check if the episode is done."""
        terminated = state.xp + state.over
        truncated = state.time >= self.max_steps_in_episode
        done = jnp.logical_or(terminated, truncated)
        return done

    @functools.partial(jax.jit, static_argnums=(0,))
    def render(self, state: EnvState) -> chex.Array:
        """Render the current state of the environment."""
        rng = jax.random.PRNGKey(state.time)
        render_config = self.render_mode[self.obs_size]
        board_size = self.grid_size

        grid_px = render_config["grid_px"]
        sub_size = render_config["sub_size"][board_size]
        square_size = (sub_size - (board_size + 1) * grid_px) // board_size

        # Generate grid coordinates using meshgrid
        x_coords, y_coords = jnp.arange(board_size), jnp.arange(board_size)
        xx, yy = jnp.meshgrid(x_coords, y_coords, indexing="ij")
        top_left_x = grid_px + yy * (square_size + grid_px)
        top_left_y = grid_px + xx * (square_size + grid_px)
        all_top_left = jnp.stack([top_left_x, top_left_y], axis=-1)
        all_bottom_right = all_top_left + square_size

        # Initialize canvases
        canvas = jnp.full(
            (render_config["size"],) * 2 + (3,), render_config["clr"]
        )
        sub_canvas = jnp.full(
            (sub_size, sub_size, 3), render_config["sub_clr"]
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
            rainbow_colors = jnp.array(
                [
                    render_config["red"],
                    render_config["orange"],
                    render_config["yellow"],
                    render_config["green"],
                    render_config["blue"],
                    render_config["indigo"],
                    render_config["violet"],
                ], dtype=jnp.uint8
            )
            color_idx = jnp.int32(state.color_indexes[x])
            enemy_color = rainbow_colors[color_idx % len(rainbow_colors)]
            # Draw enemy block if cell_val is 1
            canvas = lax.cond(
                cell_val == 1,
                lambda: draw_rectangle(tl, br, enemy_color, canvas),
                lambda: canvas,
            )

            return canvas

        def _render_partial(sub_canvas, rng):
            pos = action_x * board_size + action_y
            sub_canvas = render_cell(pos, sub_canvas)

            cell_indices = jnp.arange(board_size * board_size)

            rng, rng_mask = jax.random.split(rng, 2)
            mask = jax.random.bernoulli(rng_mask, p=self.p, shape=(board_size * board_size,))

            # agent always True
            mask = mask.at[pos].set(True)

            def render_masked(idx, canvas):
                return lax.cond(
                    mask[idx],
                    lambda: render_cell(cell_indices[idx], canvas),
                    lambda: canvas,
                )

            sub_canvas = jax.lax.fori_loop(
                0,
                board_size * board_size,
                lambda i, c: render_masked(i, c),
                sub_canvas,
            )
            return sub_canvas

        def _render_full(sub_canvas):
            cell_indices = jnp.arange(board_size**2)
            updated = jax.vmap(render_cell, in_axes=(0, None))(cell_indices, sub_canvas)
            return jnp.max(updated, axis=0)

        sub_canvas = lax.cond(
            state.time == 0,
            lambda: _render_full(sub_canvas),
            lambda: lax.cond(
                self.partial_obs,
                lambda: _render_partial(sub_canvas, rng),
                lambda: _render_full(sub_canvas),
            ),
        )

        action_tl = all_top_left[action_x, action_y]
        action_br = all_bottom_right[action_x, action_y]
        sub_canvas = draw_rectangle(
            action_tl, action_br, render_config["action_clr"], sub_canvas
        )

        canvas = draw_number(
            render_config["sc_t_l"],
            render_config["sc_b_r"],
            render_config["sc_clr"],
            canvas,
            state.score,
        )

        canvas = draw_str(
            render_config["env_t_l"],
            render_config["env_b_r"],
            render_config["env_clr"],
            canvas,
            self.name,
        )

        return draw_sub_canvas(sub_canvas, canvas)

    @property
    def name(self) -> str:
        return "Skittles"

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(5)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 255, (self.obs_size, self.obs_size, 3), dtype=jnp.uint8)


class SkittlesEasy(Skittles):
    def __init__(self, **kwargs):
        super().__init__(
            max_steps_in_episode=100,
            grid_size=10,
            p=0.5,
            enemy_num=1,
            enemy_spawn_width=8,
            enemy_spawn_offset=1,
            **kwargs,
        )


class SkittlesMedium(Skittles):
    def __init__(self, **kwargs):
        super().__init__(
            max_steps_in_episode=100,
            grid_size=10,
            p=0.5,
            enemy_num=1,
            enemy_spawn_width=7,
            enemy_spawn_offset=2,
            **kwargs,
        )


class SkittlesHard(Skittles):
    def __init__(self, **kwargs):
        super().__init__(
            max_steps_in_episode=100,
            grid_size=10,
            p=0.5,
            enemy_num=1,
            enemy_spawn_width=6,
            enemy_spawn_offset=2,
            **kwargs,
        )