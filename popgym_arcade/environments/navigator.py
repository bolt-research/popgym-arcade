import functools
from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from chex import dataclass
from gymnax.environments import environment, spaces
from jax import lax

from popgym_arcade.environments.draw_utils import (
    draw_grid,
    draw_hexagon,
    draw_number,
    draw_rectangle,
    draw_str,
    draw_sub_canvas,
    draw_tnt_block,
)


def is_valid_tnt_placement(board, row, col, direction, tnt_size):
    """Check if a TNT block placement is valid without modifying the board."""
    board_shape = board.shape

    horizontal_tnt = jax.lax.dynamic_slice(board, (row, col), (1, tnt_size))
    vertical_tnt = jax.lax.dynamic_slice(board, (row, col), (tnt_size, 1))

    horizontal_validity = jnp.logical_and(
        col + tnt_size <= board_shape[1], jnp.all(horizontal_tnt == 0)
    )

    vertical_validity = jnp.logical_and(
        row + tnt_size <= board_shape[0], jnp.all(vertical_tnt == 0)
    )

    return jnp.where(direction == 0, horizontal_validity, vertical_validity)


vectorized_validity_check = jax.vmap(
    jax.vmap(
        jax.vmap(is_valid_tnt_placement, in_axes=(None, 0, None, None, None)),
        in_axes=(None, None, 0, None, None),
    ),
    in_axes=(None, None, None, 0, None),
)


def place_tnt_on_board(board, row, col, direction, tnt_size):
    horizontal_tnt = jnp.ones((1, tnt_size))
    vertical_tnt = jnp.ones((tnt_size, 1))

    horizontal_board = jax.lax.dynamic_update_slice(board, horizontal_tnt, (row, col))
    vertical_board = jax.lax.dynamic_update_slice(board, vertical_tnt, (row, col))

    updated_board = jax.lax.select(direction == 0, horizontal_board, vertical_board)

    return updated_board


def place_random_tnt_on_board(rng, board, tnt_size):
    size = board.shape[0]
    dirs = jnp.array([0, 1])
    rows = jnp.arange(size)
    cols = jnp.arange(size)
    valid_spots = vectorized_validity_check(board, rows, cols, dirs, tnt_size)
    total_num_spots = np.prod(valid_spots.shape)
    rand_valid = jax.random.choice(
        rng, jnp.arange(total_num_spots), shape=(1,), p=valid_spots.flatten()
    )[0]
    direction, col, row = (
        rand_valid // (size * size),
        (rand_valid % (size * size)) // size,
        (rand_valid % (size * size)) % size,
    )

    board = place_tnt_on_board(board, row, col, direction, tnt_size)
    return board


def generate_random_tnt_board(rng, board_size, tnt_sizes):
    board = jnp.zeros((board_size, board_size))
    for tnt_size in tnt_sizes:
        rng, _rng = jax.random.split(rng)
        board = place_random_tnt_on_board(_rng, board, tnt_size)
    return board


@dataclass(frozen=True)
class EnvState:
    action_x: chex.Array
    action_y: chex.Array
    timestep: int
    board: chex.Array
    score: int


@dataclass(frozen=True)
class EnvParams:
    pass


class Navigator(environment.Environment):
    """
    JAX compilable environment for 2d Grid-Based Navigation Game.

    ### Description
    In Navigator, the agent is tasked with navigating a grid-based environment to collect
    treasures while avoiding barriers.
    The agent can move in four directions: Up, Down, Left, and Right. And can also choose to Hit.
    The primary goal is to collect treasures while avoiding barriers, all while maximizing
    efficiency to receive the highest cumulative reward. There are three different levels
    of difficulty: Easy, Medium, and Hard. Easy has a board size of 8x8, Medium has a board
    size of 10x10, and Hard has a board size of 12x12.

    ### Board Elements:
    - Empty Spaces (0): The agent can move freely through these unobstructed areas.
    - Barriers (1): These block the agent's movement and incur a penalty if encountered.
    - Treasures (2): Collecting these rewards the agent with a positive reward.

    ### Action Space
    |--------|-------------|
    | Action | Description |
    |--------|-------------|
    |---0----|-----Move-Up-|
    |---1----|---Move-Down-|
    |---2----|---Move-Left-|
    |---3----|--Move-Right-|
    |---4----|--------Hit--|

    ### Observation Space
    The observation space is a chex.Array with shape `(256, 256, 3)`
    Current state is rendered into a matrix using multiple graphical elements to form a visual observation.
    The entire observation consists of a large canvas with a smaller canvas embedded within it.
    The smaller canvas primarily displays the game interface, while the larger canvas shows additional
    information, such as the score.

    In Fully Observable space, the agent can see the entire board at all times. In Navigator,
    the agent's current action is represented by a yellow match man, while barriers are shown as red TNT blocks,
    and treasures are displayed as light blue hexagons.

    In Partially Observable space, the agent can only see the initial state of the board at the start of the episode.
    When timestep > 0, the agent can only see the match man's current position and the whole grid.

    Agent can always see the score at the top right corner of the canvas.

    ### Reward
    The agent receives a reward of -0.5/(board_size*board_size) for each step taken,
    a reward of 1.0 for hitting the treasure, and a reward of -0.5 for reaching a TNT block.

    ### Termination & Truncation
    The episode terminates when the agent hits a TNT block or collects the treasure.
    The episode is truncated if the agent exceeds the maximum episode length.

    ### Args
    board_size: Size of the board. Easy: 8, Medium: 10, Hard: 12.
    partial_obs: Whether the environment is partially observable or not.
    max_steps_in_episode: The maximum number of steps in an episode.

    """

    render_common = {
        # parameters for rendering (256, 256, 3) canvas
        "clr": jnp.array([119, 122, 127], dtype=jnp.uint8),
        "sub_clr": jnp.array([119, 112, 127], dtype=jnp.uint8),
        # parameters for rendering grids
        "grid_clr": jnp.array([255, 255, 255], dtype=jnp.uint8),
        # parameters for current action position
        "action_clr": jnp.array([255, 255, 0], dtype=jnp.uint8),
        # parameters for rendering treasure
        "trea_clr": jnp.array([73, 214, 247], dtype=jnp.uint8),
        # parameters for rendering score
        "sc_clr": jnp.array([0, 0, 127], dtype=jnp.uint8),
        # parameters for rendering env name
        "env_clr": jnp.array([138, 0, 138], dtype=jnp.uint8),
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
        # parameters for rendering envName
        "env_t_l": (0, 115),
        "env_b_r": (128, 128),
    }
    render_mode = {
        256: render_256x,
        128: render_128x,
    }

    def __init__(
        self,
        board_size: int,
        partial_obs=False,
        obs_size: int = 128,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.board_size = board_size
        self.barrier_sizes = [1, 1, 1, 1, 1, 1, 1]
        self.max_steps_in_episode = self.board_size * self.board_size
        self.reward_step = 0.5 / (self.board_size * self.board_size)
        self.reward_win = 1.0
        self.reward_die = 0.5
        self.partial_obs = partial_obs

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @property
    def name(self) -> str:
        return "Navigator"

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

        def move_up(state):
            action_y = lax.max(state.action_y - 1, 0)
            new_timestep = state.timestep + 1
            new_state = state.replace(action_y=action_y, timestep=new_timestep)
            terminated = state.board[new_state.action_x, new_state.action_y] == 1
            truncated = new_timestep >= self.max_steps_in_episode
            done = jnp.logical_or(terminated, truncated)
            reward = lax.cond(
                terminated,
                lambda _: -self.reward_die,
                lambda _: -self.reward_step,
                operand=None,
            )
            reward = lax.cond(truncated, lambda _: 0.0, lambda _: reward, operand=None)
            new_score = state.score + lax.cond(
                reward > 0.0, lambda _: 100, lambda _: 0, operand=None
            )
            new_state = new_state.replace(score=new_score)
            infos = {
                "terminated": terminated,
                "truncated": truncated,
            }
            return self.get_obs(new_state), new_state, reward, done, infos

        def move_down(state):
            action_y = lax.min(state.action_y + 1, self.board_size - 1)
            new_timestep = state.timestep + 1
            new_state = state.replace(action_y=action_y, timestep=new_timestep)
            terminated = state.board[new_state.action_x, new_state.action_y] == 1
            truncated = new_timestep >= self.max_steps_in_episode
            done = jnp.logical_or(terminated, truncated)
            reward = lax.cond(
                terminated,
                lambda _: -self.reward_die,
                lambda _: -self.reward_step,
                operand=None,
            )
            reward = lax.cond(truncated, lambda _: 0.0, lambda _: reward, operand=None)
            new_score = state.score + lax.cond(
                reward > 0.0, lambda _: 100, lambda _: 0, operand=None
            )
            new_state = new_state.replace(score=new_score)
            infos = {
                "terminated": terminated,
                "truncated": truncated,
            }
            return self.get_obs(new_state), new_state, reward, done, infos

        def move_left(state):
            action_x = lax.max(state.action_x - 1, 0)
            new_timestep = state.timestep + 1
            new_state = state.replace(action_x=action_x, timestep=new_timestep)
            terminated = state.board[new_state.action_x, new_state.action_y] == 1
            truncated = new_timestep >= self.max_steps_in_episode
            done = jnp.logical_or(terminated, truncated)
            reward = lax.cond(
                terminated,
                lambda _: -self.reward_die,
                lambda _: -self.reward_step,
                operand=None,
            )
            reward = lax.cond(truncated, lambda _: 0.0, lambda _: reward, operand=None)
            new_score = state.score + lax.cond(
                reward > 0.0, lambda _: 100, lambda _: 0, operand=None
            )
            new_state = new_state.replace(score=new_score)
            infos = {
                "terminated": terminated,
                "truncated": truncated,
            }
            return self.get_obs(new_state), new_state, reward, done, infos

        def move_right(state):
            action_x = lax.min(state.action_x + 1, self.board_size - 1)
            new_timestep = state.timestep + 1
            new_state = state.replace(action_x=action_x, timestep=new_timestep)
            terminated = state.board[new_state.action_x, new_state.action_y] == 1
            truncated = new_timestep >= self.max_steps_in_episode
            done = jnp.logical_or(terminated, truncated)
            reward = lax.cond(
                terminated,
                lambda _: -self.reward_die,
                lambda _: -self.reward_step,
                operand=None,
            )
            reward = lax.cond(truncated, lambda _: 0.0, lambda _: reward, operand=None)
            new_score = state.score + lax.cond(
                reward > 0.0, lambda _: 100, lambda _: 0, operand=None
            )
            new_state = new_state.replace(score=new_score)
            infos = {
                "terminated": terminated,
                "truncated": truncated,
            }
            return self.get_obs(new_state), new_state, reward, done, infos


        def hit(state):
            action_x, action_y = state.action_x, state.action_y
            is_treasure = state.board[action_x, action_y] == 2
            new_timestep = state.timestep + 1
            terminated = is_treasure
            truncated = new_timestep >= self.max_steps_in_episode
            done = jnp.logical_or(is_treasure, truncated)
            reward = lax.cond(
                terminated,
                lambda _: self.reward_win,
                lambda _: -self.reward_step,
                operand=None,
            )
            reward = jnp.where((new_timestep >= self.max_steps_in_episode), 0.0, reward)
            new_score = state.score + lax.cond(
                reward > 0.0, lambda _: 100, lambda _: 0, operand=None
            )
            new_state = state.replace(
                timestep=new_timestep,
                board=state.board,
                score=new_score,
            )
            obs = self.get_obs(new_state)
            infos = {
                "terminated": terminated,
                "truncated": truncated,
            }
            return obs, new_state, reward, done, infos

        action_functions = [move_up, move_down, move_left, move_right, hit]

        info = lax.switch(
            action,
            action_functions,
            state,
        )

        return info

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        board = generate_random_tnt_board(key, self.board_size, self.barrier_sizes)

        x_key, y_key = jax.random.split(key)
        key_2 = jax.random.PRNGKey(1)
        treasure_x_key, treasure_y_key = jax.random.split(key_2)

        non_zero_positions = jax.jit(jnp.where, static_argnames="size")(
            board == 0,
            size=((self.board_size * self.board_size) - sum(self.barrier_sizes)),
        )

        treasure_x = jax.random.choice(treasure_x_key, non_zero_positions[0])
        treasure_y = jax.random.choice(treasure_y_key, non_zero_positions[1])
        board = board.at[treasure_x, treasure_y].set(2)

        action_x = jax.random.choice(x_key, non_zero_positions[0])
        action_y = jax.random.choice(y_key, non_zero_positions[1])

        state = EnvState(
            action_x=action_x,
            action_y=action_y,
            timestep=0,
            board=board,
            score=0,
        )
        obs = self.get_obs(state)
        return obs, state

    def get_obs(self, state, params=None, key=None) -> chex.Array:
        return self.render(state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def render(self, state) -> chex.Array:
        """Render the current state of the environment."""

        # Render mode setup
        render_config = self.render_mode[self.obs_size]
        # Board and grid setup
        board_size = self.board_size
        grid_px = render_config["grid_px"]
        sub_size = render_config["sub_size"][board_size]
        square_size = (sub_size - (board_size + 1) * grid_px) // board_size

        # Generate grid coordinates using meshgrid
        x_coords, y_coords = jnp.arange(board_size), jnp.arange(board_size)
        xx, yy = jnp.meshgrid(x_coords, y_coords, indexing="ij")
        top_left_x = grid_px + xx * (square_size + grid_px)
        top_left_y = grid_px + yy * (square_size + grid_px)
        all_top_left = jnp.stack([top_left_x, top_left_y], axis=-1)
        all_bottom_right = all_top_left + square_size

        # Initialize canvases
        canvas = jnp.full((render_config["size"],) * 2 + (3,), render_config["clr"])
        sub_canvas = jnp.full((sub_size, sub_size, 3), render_config["sub_clr"])

        # Extract action coordinates
        action_x, action_y = state.action_x, state.action_y

        # Precompute board values
        board_flat = state.board.flatten()

        # Define cell rendering function
        def render_cell(pos, canvas):
            x = pos // board_size
            y = pos % board_size
            tl = all_top_left[x, y]
            br = all_bottom_right[x, y]
            cell_val = board_flat[pos]

            # Draw TNT block if cell_val is 1
            canvas = lax.cond(
                cell_val == 1, lambda: draw_tnt_block(tl, br, canvas), lambda: canvas
            )
            # Draw treasure hexagon if cell_val is 2
            return lax.cond(
                cell_val == 2,
                lambda: draw_hexagon(tl, br, render_config["trea_clr"], canvas),
                lambda: canvas,
            )

        # Partial rendering: only the current action cell
        def _render_partial(sub_canvas):
            pos = action_x * board_size + action_y
            return render_cell(pos, sub_canvas)

        # Full rendering: all cells
        def _render_full(sub_canvas):
            cell_indices = jnp.arange(board_size**2)
            updated = jax.vmap(render_cell, in_axes=(0, None))(cell_indices, sub_canvas)
            return jnp.max(updated, axis=0)

        # Conditional rendering logic
        sub_canvas = lax.cond(
            state.timestep == 0,
            lambda: _render_full(sub_canvas),
            lambda: lax.cond(
                self.partial_obs,
                lambda: _render_partial(sub_canvas),
                lambda: _render_full(sub_canvas),
            ),
        )

        # Draw matchstick man on the current action cell
        action_tl = all_top_left[action_x, action_y]
        action_br = all_bottom_right[action_x, action_y]
        sub_canvas = draw_rectangle(
            action_tl, action_br, render_config["action_clr"], sub_canvas
        )
        # sub_canvas = draw_matchstick_man(
        #     action_tl,
        #     action_br,
        #     render_config["action_clr"],
        #     sub_canvas
        # )

        # Draw grid lines
        sub_canvas = draw_grid(
            square_size, grid_px, render_config["grid_clr"], sub_canvas
        )

        # Draw score on canvas
        canvas = draw_number(
            render_config["sc_t_l"],
            render_config["sc_b_r"],
            render_config["sc_clr"],
            canvas,
            state.score,
        )

        # Draw environment name
        canvas = draw_str(
            render_config["env_t_l"],
            render_config["env_b_r"],
            render_config["env_clr"],
            canvas,
            self.name,
        )

        # Merge sub-canvas onto main canvas
        return draw_sub_canvas(sub_canvas, canvas)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(5)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 255, (self.obs_size, self.obs_size, 3), dtype=jnp.uint8)


class NavigatorEasy(Navigator):
    def __init__(self, **kwargs):
        super().__init__(board_size=8, **kwargs)


class NavigatorMedium(Navigator):
    def __init__(self, **kwargs):
        super().__init__(board_size=10, **kwargs)


class NavigatorHard(Navigator):
    def __init__(self, **kwargs):
        super().__init__(board_size=12, **kwargs)
