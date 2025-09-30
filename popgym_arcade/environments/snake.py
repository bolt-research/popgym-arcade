"""JAX implementation of a Snake logic board environment."""

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


BOARD_SIZE = 15
MAX_LENGTH = BOARD_SIZE * BOARD_SIZE


@dataclass(frozen=True)
class EnvState(environment.EnvState):
	snake_body: jax.Array
	direction: jax.Array
	length: jax.Array
	apple_pos: jax.Array
	apple_flash: jax.Array
	time: jax.Array
	terminal: jax.Array
	score: jax.Array


@dataclass(frozen=True)
class EnvParams(environment.EnvParams):
	max_steps_in_episode: int = 1000


class Snake(environment.Environment[EnvState, EnvParams]):
	"""JAX implementation of a 15x15 Snake game."""

	color = {
		"frame": jnp.array([22, 26, 33], dtype=jnp.float32) / 255.0,
		"panel": jnp.array([37, 43, 52], dtype=jnp.float32) / 255.0,
		"board_bg": jnp.array([13, 17, 23], dtype=jnp.float32) / 255.0,
		"grid": jnp.array([55, 63, 76], dtype=jnp.float32) / 255.0,
		"body": jnp.array([108, 194, 135], dtype=jnp.float32) / 255.0,
		"head": jnp.array([56, 168, 96], dtype=jnp.float32) / 255.0,
		"apple": jnp.array([230, 90, 80], dtype=jnp.float32) / 255.0,
		"text": jnp.array([236, 240, 243], dtype=jnp.float32) / 255.0,
		"score_glow": jnp.array([92, 104, 120], dtype=jnp.float32) / 255.0,
	}

	size = {
		256: {
			"canvas_size": 256,
			"small_canvas_size": 200,
			"name_pos": {"top_left": (0, 230), "bottom_right": (256, 255)},
			"score": {"top_left": (86, 2), "bottom_right": (171, 30)},
		},
		128: {
			"canvas_size": 128,
			"small_canvas_size": 100,
			"name_pos": {"top_left": (0, 114), "bottom_right": (128, 127)},
			"score": {"top_left": (43, 1), "bottom_right": (85, 15)},
		},
	}

	def __init__(
		self,
		max_steps_in_episode: int = 1000,
		obs_size: int = 128,
		partial_obs: bool = False,
	) -> None:
		super().__init__()
		self.board_size = BOARD_SIZE
		self.max_length = MAX_LENGTH
		self.obs_shape = (self.board_size, self.board_size, 3)
		self.direction_vectors = jnp.array(
			[[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32
		)
		self.opposite_directions = jnp.array([1, 0, 3, 2], dtype=jnp.int32)
		self.full_action_set = jnp.arange(5, dtype=jnp.int32)
		self.minimal_action_set = self.full_action_set
		self.action_set = self.full_action_set

		self.win_score = 100
		self.reward_scale = 0.01
		self.max_steps_in_episode = max_steps_in_episode
		self.obs_size = obs_size
		self.partial_obs = partial_obs

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
		action = jnp.asarray(action, dtype=jnp.int32)
		a = self.action_set[action]

		new_direction = self._update_direction(state.direction, a)
		step_vector = self.direction_vectors[new_direction]
		head = state.snake_body[0]
		new_head = head + step_vector

		hit_wall = jnp.any(
			jnp.logical_or(new_head < 0, new_head >= self.board_size)
		)

		eat_apple = jnp.logical_and(
			jnp.logical_not(hit_wall), jnp.all(new_head == state.apple_pos)
		)
		grow = eat_apple.astype(jnp.int32)

		indices = jnp.arange(self.max_length)
		active_mask = indices < state.length
		tail_index = jnp.clip(state.length - 1, 0, self.max_length - 1)
		exclude_tail = jnp.logical_and(indices == tail_index, grow == 0)
		collision_mask = jnp.logical_and(active_mask, jnp.logical_not(exclude_tail))
		matches = jnp.all(state.snake_body == new_head, axis=1)
		self_collision = jnp.any(jnp.logical_and(matches, collision_mask))

		shifted_body = jnp.concatenate(
			(new_head[None, :], state.snake_body[:-1]), axis=0
		)
		new_body = jax.lax.cond(hit_wall, lambda _: state.snake_body, lambda _: shifted_body, operand=None)

		new_length = jnp.minimum(self.max_length, state.length + grow)

		key, apple_key = jax.random.split(key)
		new_apple = jax.lax.cond(
			eat_apple,
			lambda _: self._spawn_apple(apple_key, new_body, new_length),
			lambda _: state.apple_pos,
			operand=None,
		)

		new_score = jnp.minimum(self.win_score, state.score + grow)
		board_full = new_length == self.max_length
		collision = jnp.logical_or(hit_wall, self_collision)
		win_condition = jnp.logical_or(board_full, new_score >= self.win_score)
		terminal_flag = jnp.logical_or(collision, win_condition)
		jax.debug.print("collision: {}, terminal_flag: {}", collision, terminal_flag)
		new_flash = jnp.where(
			eat_apple,
			jnp.int32(1),
			jnp.maximum(state.apple_flash - 1, jnp.int32(0)),
		)

		updated_state = state.replace(
			snake_body=new_body,
			direction=new_direction,
			length=new_length,
			apple_pos=new_apple,
			apple_flash=new_flash,
			time=state.time + 1,
			terminal=terminal_flag,
			score=new_score,
		)

		done = self.is_terminal(updated_state, params)
		updated_state = updated_state.replace(terminal=done)

		base_reward = (grow.astype(jnp.float32)) * self.reward_scale
		death = collision
		reward = jnp.where(death, jnp.float32(-1.0), base_reward)
		# reward = jnp.clip(reward, -1.0, 1.0)
		info = {"discount": self.discount(updated_state, params)}
		jax.debug.print("timestep: {}, done: {}, reward: {}, grow: {}", state.time, done, reward, grow)
		
		return (
			# jax.lax.stop_gradient(self.get_obs(updated_state)),
			jax.lax.stop_gradient(self.render(updated_state)),
			jax.lax.stop_gradient(updated_state),
			reward,
			done,
			info,
		)

	def reset_env(
		self, key: jax.Array, params: EnvParams
	) -> tuple[jax.Array, EnvState]:
		key, apple_key = jax.random.split(key)
		mid = self.board_size // 2

		base_body = jnp.zeros((self.max_length, 2), dtype=jnp.int32)
		base_body = base_body.at[0].set(jnp.array([mid, mid + 1], dtype=jnp.int32))
		base_body = base_body.at[1].set(jnp.array([mid, mid], dtype=jnp.int32))
		base_body = base_body.at[2].set(jnp.array([mid, mid - 1], dtype=jnp.int32))

		length = jnp.int32(3)
		direction = jnp.int32(3)  # Start moving right

		apple_pos = self._spawn_apple(apple_key, base_body, length)

		state = EnvState(
			snake_body=base_body,
			direction=direction,
			length=length,
			apple_pos=apple_pos,
			apple_flash=jnp.int32(1),
			time=jnp.int32(0),
			terminal=jnp.bool_(False),
			score=jnp.int32(0),
		)
		return self.render(state), state

	def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
		body_layer = self._body_layer(state.snake_body, state.length)
		head_layer = jnp.zeros((self.board_size, self.board_size), dtype=jnp.float32)
		head_layer = jax.lax.cond(
			state.length > 0,
			lambda _: head_layer.at[tuple(state.snake_body[0])].set(1.0),
			lambda _: head_layer,
			operand=None,
		)

		apple_layer = jnp.zeros((self.board_size, self.board_size), dtype=jnp.float32)
		apple_layer = apple_layer.at[tuple(state.apple_pos)].set(1.0)

		return jnp.stack((head_layer, body_layer, apple_layer), axis=-1)

	def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
		done_steps = state.time >= params.max_steps_in_episode
		return jnp.logical_or(done_steps, state.terminal)

	@property
	def name(self) -> str:
		return "Snake"

	@functools.partial(jax.jit, static_argnums=(0,))
	def render(self, state: EnvState) -> jax.Array:
		size_cfg = self.size[self.obs_size]
		canvas_size = size_cfg["canvas_size"]
		small_canvas_size = size_cfg["small_canvas_size"]
		canvas = jnp.full((canvas_size, canvas_size, 3), self.color["frame"])
		small_canvas = jnp.full((small_canvas_size, small_canvas_size, 3), self.color["panel"])

		cell_size = max(1, small_canvas_size // self.board_size)
		board_pixels = cell_size * self.board_size
		ox = (small_canvas_size - board_pixels) // 2
		oy = (small_canvas_size - board_pixels) // 2

		y_coords, x_coords = jnp.meshgrid(
			jnp.arange(small_canvas_size),
			jnp.arange(small_canvas_size),
			indexing="ij",
		)

		in_board_x = jnp.logical_and(x_coords >= ox, x_coords < ox + board_pixels)
		in_board_y = jnp.logical_and(y_coords >= oy, y_coords < oy + board_pixels)
		board_mask = jnp.logical_and(in_board_x, in_board_y)

		small_canvas = jnp.where(board_mask[:, :, None], self.color["board_bg"], small_canvas)

		rel_x = (x_coords - ox) % cell_size
		rel_y = (y_coords - oy) % cell_size
		border_x = jnp.logical_or(x_coords == ox, x_coords == ox + board_pixels - 1)
		border_y = jnp.logical_or(y_coords == oy, y_coords == oy + board_pixels - 1)
		grid_mask = jnp.logical_and(
			board_mask,
			jnp.logical_or(jnp.logical_or(rel_x == 0, rel_y == 0), jnp.logical_or(border_x, border_y)),
		)
		small_canvas = jnp.where(grid_mask[:, :, None], self.color["grid"], small_canvas)

		cell_size_arr = jnp.int32(cell_size)
		board_y = jnp.clip(((y_coords - oy) // cell_size_arr).astype(jnp.int32), 0, self.board_size - 1)
		board_x = jnp.clip(((x_coords - ox) // cell_size_arr).astype(jnp.int32), 0, self.board_size - 1)

		body_layer = self._body_layer(state.snake_body, state.length)
		body_present = body_layer[board_y, board_x] > 0
		body_mask = jnp.logical_and(board_mask, body_present)

		valid_head = state.length > 0
		head_pos = jax.lax.cond(
			valid_head,
			lambda _: state.snake_body[0],
			lambda _: jnp.array([0, 0], dtype=jnp.int32),
			operand=None,
		)
		head_mask = jnp.logical_and(
			body_mask,
			jnp.logical_and(board_y == head_pos[0], board_x == head_pos[1]),
		)
		body_mask = jnp.logical_and(body_mask, jnp.logical_not(head_mask))

		apple_pos = state.apple_pos
		apple_mask = jnp.logical_and(
			board_mask,
			jnp.logical_and(board_y == apple_pos[0], board_x == apple_pos[1]),
		)

		apple_mask = jax.lax.cond(
			self.partial_obs,
			lambda _: jnp.logical_and(apple_mask, state.apple_flash > 0),
			lambda _: apple_mask,
			operand=None,
		)

		small_canvas = jnp.where(body_mask[:, :, None], self.color["body"], small_canvas)
		small_canvas = jnp.where(head_mask[:, :, None], self.color["head"], small_canvas)
		small_canvas = jnp.where(apple_mask[:, :, None], self.color["apple"], small_canvas)

		canvas = draw_number(
			size_cfg["score"]["top_left"],
			size_cfg["score"]["bottom_right"],
			self.color["score_glow"],
			canvas,
			state.score.astype(jnp.int32),
		)
		canvas = draw_str(
			size_cfg["name_pos"]["top_left"],
			size_cfg["name_pos"]["bottom_right"],
			self.color["text"],
			canvas,
			self.name,
		)

		return draw_sub_canvas(small_canvas, canvas)

	@property
	def num_actions(self) -> int:
		return int(self.action_set.shape[0])

	def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
		return spaces.Discrete(self.num_actions)

	def observation_space(self, params: EnvParams) -> spaces.Box:
		return spaces.Box(0.0, 1.0, self.obs_shape)

	def state_space(self, params: EnvParams) -> spaces.Dict:
		return spaces.Dict(
			{
				"snake_body": spaces.Box(
					0,
					self.board_size - 1,
					(self.max_length, 2),
					dtype=jnp.int32,
				),
				"direction": spaces.Discrete(4),
				"length": spaces.Discrete(self.max_length + 1),
				"apple_pos": spaces.Box(
					0,
					self.board_size - 1,
					(2,),
					dtype=jnp.int32,
				),
				"apple_flash": spaces.Discrete(2),
				"time": spaces.Discrete(params.max_steps_in_episode),
				"terminal": spaces.Discrete(2),
				"score": spaces.Discrete(self.max_length + 1),
			}
		)

	def _update_direction(self, direction: jax.Array, action: jax.Array) -> jax.Array:
		opposite = self.opposite_directions[direction]
		is_control = action < 4
		valid_turn = jnp.logical_and(is_control, action != opposite)
		return jax.lax.select(valid_turn, action, direction)

	def _body_layer(self, snake_body: jax.Array, length: jax.Array) -> jax.Array:
		flat = jnp.zeros(self.board_size * self.board_size, dtype=jnp.float32)
		indices = snake_body[:, 0] * self.board_size + snake_body[:, 1]
		mask = (jnp.arange(self.max_length) < length).astype(jnp.float32)
		flat = flat.at[indices].add(mask)
		flat = jnp.clip(flat, 0.0, 1.0)
		return flat.reshape(self.board_size, self.board_size)

	def _spawn_apple(
		self, key: jax.Array, snake_body: jax.Array, length: jax.Array
	) -> jax.Array:
		total_cells = self.max_length
		indices = snake_body[:, 0] * self.board_size + snake_body[:, 1]
		mask = (jnp.arange(self.max_length) < length).astype(jnp.float32)
		occupancy = jnp.zeros(total_cells, dtype=jnp.float32)
		occupancy = occupancy.at[indices].add(mask)
		occupancy = jnp.clip(occupancy, 0.0, 1.0)
		available = 1.0 - occupancy
		available_sum = jnp.sum(available)

		def sample(_: None) -> jax.Array:
			probs = available / available_sum
			idx = jax.random.choice(key, total_cells, p=probs)
			return jnp.stack((idx // self.board_size, idx % self.board_size)).astype(
				jnp.int32
			)

		def fallback(_: None) -> jax.Array:
			return snake_body[0]

		return jax.lax.cond(available_sum > 0, sample, fallback, operand=None)

class SnakeEasy(Snake):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode=1000, **kwargs)

class SnakeMedium(Snake):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode=1000, **kwargs)

class SnakeHard(Snake):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode=1000, **kwargs)


# import matplotlib.pyplot as plt

# env = Snake()
# key = jax.random.PRNGKey(0)
# key, re_key = jax.random.split(key)
# obs, state = env.reset_env(re_key, env.default_params)
# plt.imshow(obs)
# plt.show()
# while True:
# 	key, step_key = jax.random.split(key)
# 	action = int(input())
# 	obs, state, reward, done, info = env.step_env(key, state, action, env.default_params)
# 	print(reward, done)
# 	plt.imshow(obs)
# 	plt.show()
# 	if done:
# 		break