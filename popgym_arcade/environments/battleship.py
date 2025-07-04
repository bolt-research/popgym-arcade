import functools
from typing import Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from chex import dataclass
from gymnax.environments import environment, spaces
from jax import lax, random

from popgym_arcade.environments.draw_utils import (
    draw_grid,
    draw_number,
    draw_o,
    draw_rectangle,
    draw_str,
    draw_sub_canvas,
    draw_x,
)


def is_valid_placement(board, row, col, direction, ship_size):
    """Check if a placement is valid without modifying the board."""
    board_shape = board.shape

    # Slice the board
    horizontal_board = jax.lax.dynamic_slice(board, (row, col), (1, ship_size))
    vertical_board = jax.lax.dynamic_slice(board, (row, col), (ship_size, 1))

    # Check validities
    horizontal_validity = jnp.logical_and(
        col + ship_size <= board_shape[1], jnp.all(horizontal_board == 0)
    )

    vertical_validity = jnp.logical_and(
        row + ship_size <= board_shape[0], jnp.all(vertical_board == 0)
    )

    return jnp.where(direction == 0, horizontal_validity, vertical_validity)


vectorized_validity_check = jax.vmap(
    jax.vmap(
        jax.vmap(is_valid_placement, in_axes=(None, 0, None, None, None)),
        in_axes=(None, None, 0, None, None),
    ),
    in_axes=(None, None, None, 0, None),
)


def place_ship_on_board(board, row, col, direction, ship_size):
    """Place a ship on the board at the given position and direction."""
    # Generate the horizontal and vertical ship placements

    horizontal_ship = jnp.ones((1, ship_size))
    vertical_ship = jnp.ones((ship_size, 1))

    # Create boards with the ship placed in each direction
    horizontal_board = jax.lax.dynamic_update_slice(board, horizontal_ship, (row, col))
    vertical_board = jax.lax.dynamic_update_slice(board, vertical_ship, (row, col))

    # Use `lax.select` to choose the appropriate board based on the direction
    updated_board = jax.lax.select(direction == 0, horizontal_board, vertical_board)

    return updated_board


def place_random_ship_on_board(rng, board, ship_size):
    size = board.shape[0]
    dirs = jnp.array([0, 1])
    rows = jnp.arange(size)
    cols = jnp.arange(size)
    valid_spots = vectorized_validity_check(board, rows, cols, dirs, ship_size)
    total_num_spots = np.prod(valid_spots.shape)
    rand_valid = jax.random.choice(
        rng, jnp.arange(total_num_spots), shape=(1,), p=valid_spots.flatten()
    )[0]
    direction, col, row = (
        rand_valid // (size * size),
        (rand_valid % (size * size)) // size,
        (rand_valid % (size * size)) % size,
    )

    board = place_ship_on_board(board, row, col, direction, ship_size)
    return board


def generate_random_board(rng, board_size, ship_sizes):
    board = jnp.zeros((board_size, board_size))
    for ship_size in ship_sizes:
        rng, _rng = jax.random.split(rng)
        board = place_random_ship_on_board(_rng, board, ship_size)
    return board


@dataclass(frozen=True)
class EnvState:
    action_x: chex.Array
    action_y: chex.Array
    board: chex.Array
    guesses: chex.Array
    hits: int
    score: int
    repeat_count: int
    timestep: int


@dataclass(frozen=True)
class EnvParams:
    pass


class BattleShip(environment.Environment):
    """
    ### Description

    JAX Compatible version of BattleShip POPGym Environment.
    Source: https://github.com/proroklab/popgym/blob/master/popgym/envs/battleship.py

    Modifications:

    ### 1. Action Space
    The action is a `int / float / chex.Array` with shape `(1, )`,
    which can take value `{0, 1, 2, 3, 4}`

    | Num | Action      |
    |-----|-------------|
    | 0   | Go up       |
    | 1   | Go down     |
    | 2   | Go left     |
    | 3   | Go right    |
    | 4   | Fire        |

    ### 2.Fully Observation Space
    The observation space is a `chex.Array` with shape `(256, 256, 3)`
    Current state is rendered into a matrix using multiple graphical
    elements to form a visual observation. The entire observation
    consists of a large canvas with a smaller canvas embedded into it.
    The smaller canvas primarily displays the game interface,
    while the larger canvas shows additional information, such as the score.

    In the Battleship game, the visual information provided
    is a grid-based board. The current action position is highlighted,
    and when the fire key is used to target a grid,
    an "X" is displayed if the grid contains a ship,
    while an "O" is displayed if the grid is empty.


    At the top of the observation, the current score is displayed dynamically.
    Whenever the reward function returns a positive value, the score increases by one point.

    ### 3.Partially Observation Space
    In Partially Observation Space, the cart and pole are hidden when timesteps > 0.
    The agent can only observe two arrows.

    ### 4.Rewards
    Reward is normalized to (-1, 1)
    bad_action: reward = -1.0 / (board_size ** 2)
    hit_ship: reward = 1.0 / (board_size ** 2)
    # Note: bad_action --> repeated hit

    ### 5.args:
    board_size: The length and width of the square board.
                It is also directly related to the difficulty
                settings of the game.
    partial_obs: bool switch with POMDP and FOMDP.
    """

    render_common = {
        # parameters for rendering (256, 256, 3) canvas
        "clr": jnp.array([255, 255, 255], dtype=jnp.uint8),
        "sub_clr": jnp.array([191, 191, 191], dtype=jnp.uint8),
        # parameters for rendering grids
        "grid_clr": jnp.array([102, 102, 102], dtype=jnp.uint8),
        # parameters for rendering current action position
        "action_clr": jnp.array([217, 166, 33], dtype=jnp.uint8),
        # parameters for render hit ship grids
        "x_clr": jnp.array([255, 0, 0], dtype=jnp.uint8),
        # parameters for render hit enpty grids
        "o_clr": jnp.array([0, 0, 0], dtype=jnp.uint8),
        # parameters for render score
        "sc_clr": jnp.array([255, 128, 0], dtype=jnp.uint8),
        # parameters for render env name
        "env_clr": jnp.array([74, 214, 247], dtype=jnp.uint8),
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
        # parameters for render hit ship grids
        "x_px": 2,
        # parameters for render hit enpty grids
        "o_px": 2,
        # parameters for render score
        "sc_t_l": (86, 2),
        "sc_b_r": (171, 30),
        # parameters for render env name
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
        # parameters for render hit ship grids
        "x_px": 1,
        # parameters for render hit enpty grids
        "o_px": 1,
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
        partial_obs: bool = False,
        obs_size: int = 128,
    ):
        """Initialize the Battleship environment."""
        self.obs_size = obs_size
        self.partial_obs = partial_obs
        self.board_size = board_size
        self.ship_sizes = [2, 3, 3, 4]
        self.max_episode_length = self.board_size * self.board_size * 2
        self.needed_hits = sum(self.ship_sizes)
        self.reward_hit = 1.0 / self.needed_hits
        # self.reward_repeated_hit = -1.0 / (
        #         self.board_size * self.board_size - self.needed_hits
        # )
        self.reward_repeated_hit = -1.0 / (self.board_size * self.board_size / 2)
        self.reward_miss = 0.0

    @property
    def default_params(self) -> EnvParams:
        """Return the default environment parameters."""
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Perform a step in the environment."""

        def move_up(state):
            """Move the action position up."""
            action_y = lax.max(state.action_y - 1, 0)
            new_timestep = state.timestep + 1
            new_state = state.replace(
                action_y=action_y,
                timestep=new_timestep,
            )
            done = new_timestep >= self.max_episode_length
            infos = {
                "terminated": False,
                "truncated": new_timestep >= self.max_episode_length,
            }
            return self.get_obs(new_state), new_state, 0.0, done, infos

        def move_down(state):
            """Move the action position down."""
            action_y = lax.min(state.action_y + 1, self.board_size - 1)
            new_timestep = state.timestep + 1
            new_state = state.replace(
                action_y=action_y,
                timestep=new_timestep,
            )
            done = new_timestep >= self.max_episode_length
            infos = {
                "terminated": False,
                "truncated": new_timestep >= self.max_episode_length,
            }
            return self.get_obs(new_state), new_state, 0.0, done, infos

        def move_left(state):
            """Move the action position left."""
            action_x = lax.max(state.action_x - 1, 0)
            new_timestep = state.timestep + 1
            new_state = state.replace(
                action_x=action_x,
                timestep=new_timestep,
            )
            done = new_timestep >= self.max_episode_length
            infos = {
                "terminated": False,
                "truncated": new_timestep >= self.max_episode_length,
            }
            return self.get_obs(new_state), new_state, 0.0, done, infos

        def move_right(state):
            """Move the action position right."""
            action_x = lax.min(state.action_x + 1, self.board_size - 1)
            new_timestep = state.timestep + 1
            new_state = state.replace(
                action_x=action_x,
                timestep=new_timestep,
            )
            done = new_timestep >= self.max_episode_length
            infos = {
                "terminated": False,
                "truncated": new_timestep >= self.max_episode_length,
            }
            return self.get_obs(new_state), new_state, 0.0, done, infos

        def hit(state):
            """Perform a hit action."""
            action_x, action_y = state.action_x, state.action_y
            is_ship = state.board[action_x, action_y] == 1
            guessed_before = state.guesses[action_x, action_y] == 1
            hit = jnp.logical_and(is_ship, jnp.logical_not(guessed_before))

            new_guesses = state.guesses.at[action_x, action_y].set(1)
            new_timestep = state.timestep + 1
            new_hits = state.hits + hit
            terminated = jnp.logical_or(
                new_hits >= self.needed_hits,
                new_timestep >= self.max_episode_length,
            )
            terminated = jnp.logical_or(
                terminated, state.repeat_count >= self.board_size * self.board_size / 2
            )
            repeat_count = state.repeat_count + jnp.where(guessed_before, 1, 0)
            reward = lax.cond(
                guessed_before,
                lambda _: self.reward_repeated_hit,
                lambda _: jnp.where(hit, self.reward_hit, self.reward_miss),
                operand=None,
            )
            new_score = state.score + lax.cond(
                reward > 0,
                lambda _: 1,
                lambda _: 0,
                operand=None,
            )

            new_state = state.replace(
                timestep=new_timestep,
                board=state.board,
                guesses=new_guesses,
                hits=new_hits,
                score=new_score,
                repeat_count=repeat_count,
            )
            infos = {
                "terminated": jnp.logical_or(
                    new_state.hits >= self.needed_hits,
                    state.repeat_count >= self.board_size * self.board_size / 2,
                ),
                "truncated": new_timestep >= self.max_episode_length,
            }
            return self.get_obs(new_state), new_state, reward, terminated, infos

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
        """Reset the environment."""
        board = generate_random_board(key, self.board_size, self.ship_sizes)
        guesses = jnp.zeros((self.board_size, self.board_size))
        x_key, y_key = jax.random.split(key)
        init_action_x = random.randint(x_key, (), 0, self.board_size - 1)
        init_action_y = random.randint(y_key, (), 0, self.board_size - 1)
        state = EnvState(
            action_x=init_action_x,
            action_y=init_action_y,
            repeat_count=0,
            timestep=0,
            board=board,
            guesses=guesses,
            hits=0,
            score=0,
        )
        obs = self.get_obs(state)
        return obs, state

    def get_obs(self, state, params=None, key=None) -> chex.Array:
        """Get the observation from the current state."""
        return self.render(state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def render(self, state) -> chex.Array:
        """Render the current state into an image observation."""
        # Define board and square sizes
        render_config = self.render_mode[self.obs_size]
        board_size = self.board_size
        square_size = (
            render_config["sub_size"][board_size]
            - (board_size + 1) * render_config["grid_px"]
        ) // board_size

        # Generate grid coordinates using meshgrid
        x_coords, y_coords = jnp.arange(board_size), jnp.arange(board_size)
        xx, yy = jnp.meshgrid(x_coords, y_coords, indexing="ij")
        top_left_x = render_config["grid_px"] + xx * (
            square_size + render_config["grid_px"]
        )
        top_left_y = render_config["grid_px"] + yy * (
            square_size + render_config["grid_px"]
        )
        bottom_right_x = top_left_x + square_size
        bottom_right_y = top_left_y + square_size

        # Initialize canvases
        canvas = jnp.full(
            (render_config["size"], render_config["size"], 3),
            render_config["clr"],
            dtype=jnp.uint8,
        )
        sub_canvas = jnp.full(
            (
                render_config["sub_size"][board_size],
                render_config["sub_size"][board_size],
                3,
            ),
            render_config["sub_clr"],
            dtype=jnp.uint8,
        )

        # Extract action coordinates
        action_x, action_y = state.action_x, state.action_y

        # Draw action rectangle
        tl_x, tl_y = top_left_x[action_x, action_y], top_left_y[action_x, action_y]
        br_x, br_y = (
            bottom_right_x[action_x, action_y],
            bottom_right_y[action_x, action_y],
        )
        sub_canvas = draw_rectangle(
            (tl_x, tl_y), (br_x, br_y), render_config["action_clr"], sub_canvas
        )

        # Precompute hit conditions
        hit_ship = jnp.logical_and(state.board, state.guesses)
        hit_empty = jnp.logical_and(jnp.logical_not(state.board), state.guesses)

        # Define partial rendering function
        def _render_partial(_sub_canvas):
            # Draw X if hit ship, O if hit empty
            _sub_canvas = lax.cond(
                hit_ship[action_x, action_y],
                lambda: draw_x(
                    (tl_x, tl_y),
                    (br_x, br_y),
                    render_config["x_px"],
                    render_config["x_clr"],
                    _sub_canvas,
                ),
                lambda: _sub_canvas,
            )
            return lax.cond(
                hit_empty[action_x, action_y],
                lambda: draw_o(
                    (tl_x, tl_y),
                    (br_x, br_y),
                    render_config["o_px"],
                    render_config["o_clr"],
                    _sub_canvas,
                ),
                lambda: _sub_canvas,
            )

        # Define full rendering function using vmap
        def _render_full(_sub_canvas):
            indices = jnp.arange(board_size * board_size)
            xs, ys = indices // board_size, indices % board_size

            def _render_cell(x, y, canvas):
                # Get cell coordinates
                cell_tl = (top_left_x[x, y], top_left_y[x, y])
                cell_br = (bottom_right_x[x, y], bottom_right_y[x, y])

                # Draw X if hit ship
                canvas = lax.cond(
                    hit_ship[x, y],
                    lambda: draw_x(
                        cell_tl,
                        cell_br,
                        render_config["x_px"],
                        render_config["x_clr"],
                        canvas,
                    ),
                    lambda: canvas,
                )
                # Draw O if hit empty
                return lax.cond(
                    hit_empty[x, y],
                    lambda: draw_o(
                        cell_tl,
                        cell_br,
                        render_config["o_px"],
                        render_config["o_clr"],
                        canvas,
                    ),
                    lambda: canvas,
                )

            # Batch render all cells
            batched_render = jax.vmap(_render_cell, in_axes=(0, 0, None))
            updated_canvas = batched_render(xs, ys, _sub_canvas)
            return jnp.min(updated_canvas, axis=0)

        # Draw score on canvas
        canvas = draw_number(
            render_config["sc_t_l"],
            render_config["sc_b_r"],
            render_config["sc_clr"],
            canvas,
            state.score,
        )

        # Conditional rendering logic
        sub_canvas = lax.cond(
            (state.timestep == 0) | (self.partial_obs & (state.timestep > 0)),
            lambda: _render_partial(sub_canvas),
            lambda: _render_full(sub_canvas),
        )

        # Draw grid lines
        sub_canvas = draw_grid(
            square_size, render_config["grid_px"], render_config["grid_clr"], sub_canvas
        )

        # Draw environment name
        canvas = draw_str(
            render_config["env_t_l"],
            render_config["env_b_r"],
            render_config["env_clr"],
            canvas,
            self.name,
            horizontal=True,
        )

        # Merge sub-canvas onto main canvas
        return draw_sub_canvas(sub_canvas, canvas)

    @property
    def name(self) -> str:
        """Environment name."""
        return "BattleShip"

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(5)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 255, (256, 256, 3), dtype=jnp.uint8)


class BattleShipEasy(BattleShip):
    def __init__(self, **kwargs):
        super().__init__(board_size=8, **kwargs)


class BattleShipMedium(BattleShip):
    def __init__(self, **kwargs):
        super().__init__(board_size=10, **kwargs)


class BattleShipHard(BattleShip):
    def __init__(self, **kwargs):
        super().__init__(board_size=12, **kwargs)
