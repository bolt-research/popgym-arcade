import functools
from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from chex import dataclass
from gymnax.environments import environment, spaces
from jax import lax

from popgym_arcade.environments.draw_utils import (
    draw_grid,
    draw_number,
    draw_rectangle,
    draw_single_digit,
    draw_str,
    draw_sub_canvas,
)


@dataclass(frozen=True)
class EnvState:
    """
    - mine_grid:
    0: no mine and no viewed
    1: mine
    2: viewed
    - neighbor_grid: record the number of the mines in a (3 * 3) grid
    """

    action_x: chex.Array
    action_y: chex.Array
    timestep: int
    score: int
    mine_grid: chex.Array
    neighbor_grid: chex.Array
    viewed_count: int


@dataclass(frozen=True)
class EnvParams:
    pass


def convolve2d(input: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:

    # Ensure inputs are JAX arrays and have the correct dtype
    input = jnp.asarray(input, dtype=jnp.float32)
    kernel = jnp.asarray(kernel, dtype=jnp.float32)

    # Get input and kernel shapes
    input_height, input_width = input.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate padding sizes
    pad_height = (kernel_height - 1) // 2
    pad_width = (kernel_width - 1) // 2

    # Pad the input array
    padded_input = jnp.pad(
        input, ((pad_height, pad_height), (pad_width, pad_width)), mode="constant"
    )

    # Initialize the output array
    output = jnp.zeros_like(input)

    # Perform convolution
    for i in range(input_height):
        for j in range(input_width):
            # Extract the region of interest from the padded input
            region = padded_input[i : i + kernel_height, j : j + kernel_width]
            # Compute the dot product between the region and the kernel
            output = output.at[i, j].set(jnp.sum(region * kernel))

    return output


class MineSweeper(environment.Environment):
    """
    ### Description

    JAX Compatible version of BattleShip POPGym Environment.
    Source: https://github.com/proroklab/popgym/blob/master/popgym/envs/minesweeper.py

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

    In the Minesweeper game, the visual representation of the observation space
    provides a grid board, where the current action position will be highlighted.
    The fire action will reveal a cell; if it is not a mine, the game continues
    and a certain reward is obtained. The revealed cell will display the number
    of nearby mines. If it is a mine, the game ends immediately.

    At the top of the observation, the current score is displayed dynamically.
    Whenever the reward function returns a positive value, the score increases by one point.

    ### 3.Partially Observation Space
    In Partially Observation Space, the agent can only observe the state
    at the current action position, resulting in the loss of access to
    historical trajectories and anticipated states.

    ### 4.Rewards
    Reward is normalized to (-1, 1)
    bad_action: reward = -1.0 / (board_size ** 2)
    hit_ship: reward = 1.0 / (board_size ** 2)
    # Note: bad_action --> repeated hit

    ### 5.args:
    board_size: The length and width of the square board.
                It is also directly related to the difficulty
                settings of the game.
    num_mines: number of mines to generate.
    partial_obs: bool switch with POMDP and FOMDP.
    """

    render_common = {
        "clr": jnp.array([229, 234, 242], dtype=jnp.uint8),
        "sub_clr": jnp.array([0, 0, 0], dtype=jnp.uint8),
        # parameters for rendering numbers
        "num_clr": jnp.array([255, 255, 255], dtype=jnp.uint8),
        # parameters for rendering current action position
        "action_clr": jnp.array([255, 127, 0], dtype=jnp.uint8),
        # parameters for rendering grids
        "grid_clr": jnp.array([102, 102, 102], dtype=jnp.uint8),
        # parameters for rendering score
        "sc_clr": jnp.array([153, 0, 204], dtype=jnp.uint8),
        # parameters for rendering env name
        "env_clr": jnp.array([0, 51, 102], dtype=jnp.uint8),
    }
    render_256x = {
        # parameters for rendering (256, 256, 3) canvas
        **render_common,
        "size": 256,
        "sub_size": {
            4: 186,
            6: 182,
            8: 186,
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
            4: 94,
            6: 92,
            8: 90,
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
        num_mines: int = 2,
        partial_obs: bool = False,
        obs_size: int = 128,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.board_size = board_size
        self.num_mines = num_mines
        self.partial_obs = partial_obs
        self.max_episode_length = self.board_size * self.board_size * 2
        self.success_reward_scale = 1 / (
            self.board_size * self.board_size - self.num_mines
        )
        self.fail_reward_scale = 0.0
        self.bad_action_reward_scale = -1.0 / (self.board_size * self.board_size / 2)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        def move_up(state):
            action_y = lax.max(state.action_y - 1, 0)
            new_timestep = state.timestep + 1
            new_state = state.replace(action_y=action_y, timestep=new_timestep)
            done = new_timestep >= self.max_episode_length
            infos = {
                'terminated': False,
                'truncated': new_timestep >= self.max_episode_length,
            }
            return self.get_obs(new_state), new_state, 0.0, done, infos

        def move_down(state):
            action_y = lax.min(state.action_y + 1, self.board_size - 1)
            new_timestep = state.timestep + 1
            new_state = state.replace(action_y=action_y, timestep=new_timestep)
            done = new_timestep >= self.max_episode_length
            infos = {
                'terminated': False,
                'truncated': new_timestep >= self.max_episode_length,
            }
            return self.get_obs(new_state), new_state, 0.0, done, infos

        def move_left(state):
            action_x = lax.max(state.action_x - 1, 0)
            new_timestep = state.timestep + 1
            new_state = state.replace(action_x=action_x, timestep=new_timestep)
            done = new_timestep >= self.max_episode_length
            infos = {
                'terminated': False,
                'truncated': new_timestep >= self.max_episode_length,
            }
            return self.get_obs(new_state), new_state, 0.0, done, infos

        def move_right(state):
            action_x = lax.min(state.action_x + 1, self.board_size - 1)
            new_timestep = state.timestep + 1
            new_state = state.replace(action_x=action_x, timestep=new_timestep)
            done = new_timestep >= self.max_episode_length
            infos = {
                'terminated': False,
                'truncated': new_timestep >= self.max_episode_length,
            }
            return self.get_obs(new_state), new_state, 0.0, done, infos

        def hit(state):
            action_x, action_y = state.action_x, state.action_y
            mine = state.mine_grid[action_x, action_y] == 1
            viewed = state.mine_grid[action_x, action_y] == 2

            new_grid = state.mine_grid.at[action_x, action_y].set(2)

            reward = self.success_reward_scale
            reward = jnp.where(viewed, self.bad_action_reward_scale, reward)
            reward = jnp.where(mine, self.fail_reward_scale, reward)
            new_score = state.score + lax.cond(
                reward > 0, lambda _: 1, lambda _: 0, operand=None
            )
            viewed_count = state.viewed_count + jnp.where(viewed, 1, 0)

            truncated = state.timestep >= self.max_episode_length
            terminated = jnp.where(mine, True, False)
            terminated = jnp.logical_or(
                terminated,
                jnp.sum(new_grid == 2) == (self.board_size**2 - self.num_mines),
            )
            terminated = jnp.logical_or(
                terminated,
                state.viewed_count >= (self.board_size * self.board_size / 2),
            )

            new_state = state.replace(
                score=new_score,
                timestep=state.timestep + 1,
                mine_grid=new_grid,
                neighbor_grid=state.neighbor_grid,
                viewed_count=viewed_count,
            )
            obs = self.get_obs(new_state)
            done = jnp.logical_or(terminated, truncated)
            infos = {
                'terminated': terminated,
                'truncated': truncated,
            }
            return obs, new_state, reward, done, infos

        action_functions = [move_up, move_down, move_left, move_right, hit]

        info = lax.switch(action, action_functions, state)

        return info

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        # hidden_grid = jnp.zeros((params.dims[0] * params.dims[1],), dtype=jnp.int8)
        hidden_grid = jnp.zeros((self.board_size * self.board_size,), dtype=jnp.int8)
        mines_flat = jax.random.choice(
            key, hidden_grid.shape[0], shape=(self.num_mines,), replace=False
        )
        hidden_grid = hidden_grid.at[mines_flat].set(1)
        hidden_grid = hidden_grid.reshape((self.board_size, self.board_size))
        neighbor_grid = convolve2d(hidden_grid, jnp.ones((3, 3)))
        neighbor_grid = jnp.array(neighbor_grid, dtype=jnp.int8)
        x_key, y_key = jax.random.split(key)
        action_x = jax.random.randint(x_key, (), 0, self.board_size - 1)
        action_y = jax.random.randint(y_key, (), 0, self.board_size - 1)
        state = EnvState(
            action_x=action_x,
            action_y=action_y,
            timestep=0,
            score=0,
            mine_grid=hidden_grid,
            neighbor_grid=neighbor_grid,
            viewed_count=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        return self.render(state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def render(self, state) -> chex.Array:
        # Define board and square sizes
        render_config = self.render_mode[self.obs_size]
        board_size = self.board_size
        square_size = (
            render_config["sub_size"][board_size]
            - (board_size + 1) * render_config["grid_px"]
        ) // board_size

        # Generate grid coordinates
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

        # Initialize canvas and sub-canvas
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

        # Draw rectangle for the action
        tl_x, tl_y = top_left_x[action_x, action_y], top_left_y[action_x, action_y]
        br_x, br_y = (
            bottom_right_x[action_x, action_y],
            bottom_right_y[action_x, action_y],
        )
        sub_canvas = draw_rectangle(
            (tl_x, tl_y), (br_x, br_y), render_config["action_clr"], sub_canvas
        )

        # Check hit map for mines
        hit_map = state.mine_grid == 2

        # Define partial rendering function
        def _render_partial(_sub_canvas):
            return lax.cond(
                hit_map[action_x, action_y],
                lambda: draw_number(
                    (tl_x, tl_y),
                    (br_x, br_y),
                    render_config["num_clr"],
                    _sub_canvas,
                    state.neighbor_grid[action_x, action_y],
                ),
                lambda: _sub_canvas,
            )

        # Define full rendering function
        def _render_full(_sub_canvas):
            indices = jnp.arange(board_size * board_size)
            xs, ys = indices // board_size, indices % board_size

            def _render_cell(x, y, canvas):
                return lax.cond(
                    hit_map[x, y],
                    lambda: draw_single_digit(
                        (top_left_x[x, y], top_left_y[x, y]),
                        (bottom_right_x[x, y], bottom_right_y[x, y]),
                        render_config["num_clr"],
                        canvas,
                        state.neighbor_grid[x, y],
                    ),
                    lambda: canvas,
                )

            batched_render = jax.vmap(_render_cell, in_axes=(0, 0, None))

            updated_canvas = batched_render(xs, ys, _sub_canvas)

            return jnp.max(updated_canvas, axis=0)

        # Draw score on the canvas
        canvas = draw_number(
            render_config["sc_t_l"],
            render_config["sc_b_r"],
            render_config["sc_clr"],
            canvas,
            state.score,
        )

        # Conditionally render partial or full sub-canvas
        sub_canvas = lax.cond(
            (state.timestep > 0) & self.partial_obs,
            _render_partial,
            lambda _: _render_full(sub_canvas),
            operand=sub_canvas,
        )

        # Draw grid on the sub-canvas
        sub_canvas = draw_grid(
            square_size, render_config["grid_px"], render_config["grid_clr"], sub_canvas
        )

        # Draw environment name on the canvas
        canvas = draw_str(
            render_config["env_t_l"],
            render_config["env_b_r"],
            render_config["env_clr"],
            canvas,
            self.name,
        )

        # Return the final canvas with sub-canvas drawn on it
        return draw_sub_canvas(sub_canvas, canvas)

    @property
    def name(self) -> str:
        """Environment name."""
        return "MineSweeper"

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(5)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(
            # jnp.zeros((self.num_mines,)),
            # jnp.ones((self.num_mines,)),
            0,
            255,
            (self.board_size, self.board_size, 3),
            dtype=jnp.uint8,
        )


class MineSweeperEasy(MineSweeper):
    def __init__(self, **kwargs):
        super().__init__(board_size=4, num_mines=2, **kwargs)


class MineSweeperMedium(MineSweeper):
    def __init__(self, **kwargs):
        super().__init__(board_size=6, num_mines=6, **kwargs)


class MineSweeperHard(MineSweeper):
    def __init__(self, **kwargs):
        super().__init__(board_size=8, num_mines=10, **kwargs)
