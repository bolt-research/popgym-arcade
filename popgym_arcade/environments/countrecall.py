import functools
from typing import Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from chex import dataclass
from gymnax.environments import environment, spaces
from jax import lax

from popgym_arcade.environments.draw_utils import (
    draw_club,
    draw_diamond,
    draw_heart,
    draw_number,
    draw_rectangle,
    draw_spade,
    draw_str,
    draw_sub_canvas,
)


@dataclass(frozen=True)
class EnvState:
    timestep: int
    value_cards: chex.Array
    query_cards: chex.Array
    running_count: chex.Array
    history: chex.Array
    default_action: int
    num_types: int
    score: int
    alreadyMove: int


@dataclass(frozen=True)
class EnvParams:
    pass


@jax.jit
def process_action(state: EnvState, action: int) -> Tuple[EnvState, bool]:
    """
    have 5 actions, 0: up(increase), 1: down(decrease), 2: left, 3: right, 4: fire(confirm)
    """
    new_default_action = jnp.where(
        action == 0,
        state.default_action + 1,
        jnp.where(action == 1, state.default_action - 1, state.default_action),
    )

    return state.replace(default_action=new_default_action), action == 4


class CountRecall(environment.Environment):
    """
    JAX compatible implementation of CountRecall environment.
    Source: https://github.com/proroklab/popgym/blob/master/popgym/envs/count_recall.py

    ### Description
    **CountRecall** is a game designed to test an agent’s **long-term, order-agnostic memory**—similar to
      remembering items in a **set**, rather than recalling their **sequence**.

    In each round:

    1. The agent is **dealt a card**.
    2. Later, it is asked: *“How many times have you seen this specific card before?”*
    3. The agent must **respond correctly** to earn a **reward**.

    ---

    **How to play**:
    - The **query card** appears in the **top-right corner** of the screen.
    - The **current value card** *(just dealt)* appears in the **top-left corner**.  
    - Green number is the **count**:  
    - **Increase** the count with the `up` action.  
    - **Decrease** the count with the `down` action.  
    - **Confirm** its answer with the `confirm` action.  
    - If the confirmed answer matches the **true count**, the **score**(red number) increases by **1**.

    ---

    **Difficulty Levels**:

    - **Easy** – `1 deck`, `2 types of cards`, 26 cards
    - **Medium** – `2 decks`, `2 types of cards`, 52 cards
    - **Hard** – `3 decks`, `4 types of cards`, 78 cards

    ### Observation
    The **observation space** is a `chex.Array` with shape `(256, 256, 3)`.  
    The **current state** is rendered as a visual matrix composed of multiple graphical elements.  

    The observation is structured as a **large canvas** containing a **smaller embedded canvas**:  

    - **Smaller Canvas**:  
    1. The **sequence of historical cards** observed by the agent *(only in the fully observable version)*.  

    - **Larger Canvas**:  
    1. Suit of the dealt card.  
    2. Suit of the query card.  
    3. Action counter – current action selected by the agent.  
    4. Score – total points accumulated so far.  
    5. Environment name.  
    ---
    **Observation Modes**:  
    - **Fully Observable** – Historical cards are visible in the smaller canvas.  
    - **Partially Observable** – Historical cards are hidden from the smaller canvas.
    
    ---

    ### Actions
    | Action | Description                         |
    |--------|-------------------------------------|
    | 0      | Up (Increase the count)             |
    | 1      | Down (Decrease the count)           |
    | 2      | Left (No-op)                        |
    | 3      | Right (No-op)                       |
    | 4      | Confirm (Lock in current selection) |

    ---

    ### Rewards
    The agent receives a reward of 1.0 / num_cards for each correct answer only when the agent confirms
    the selection (action 4). The agent receives a reward of 0.0 otherwise.

    ---

    ### Termination & Truncation
    The episode terminates when the agent has made `max_steps_in_episode` moves or when the agent has seen
    all the cards.
    
    ---
    
    ### Args
    num_decks: The number of decks of cards in the game. Easy: 1, Medium: 2, Hard: 3.
    num_types: The number of types of cards in the game. Easy: 2, Medium: 2, Hard: 4.
    partial_obs: Whether the environment is partially observable or not.
    max_steps_in_episode: The maximum number of steps in an episode.
    """

    color = {
        "red": jnp.array([255, 0, 0], dtype=jnp.uint8),
        "dark_red": jnp.array([191, 26, 26], dtype=jnp.uint8),
        "black": jnp.array([0, 0, 0], dtype=jnp.uint8),
        "white": jnp.array([255, 255, 255], dtype=jnp.uint8),
        "light_blue": jnp.array([74, 191, 240], dtype=jnp.uint8),
        "dark_blue": jnp.array([31, 120, 196], dtype=jnp.uint8),
        "soft_green": jnp.array([143, 237, 143], dtype=jnp.uint8),
        "green": jnp.array([143, 237, 143], dtype=jnp.uint8),
        "soft_beige": jnp.array([245, 224, 201], dtype=jnp.uint8),
        "olive_green": jnp.array([128, 161, 99], dtype=jnp.uint8),
        "maroon": jnp.array([176, 48, 97], dtype=jnp.uint8),
        "teal": jnp.array([0, 128, 128], dtype=jnp.uint8),
        "light_gray": jnp.array([245, 245, 245], dtype=jnp.uint8),
        "bright_blue": jnp.array([74, 191, 240], dtype=jnp.uint8),
        "muted_blue": jnp.array([153, 179, 204], dtype=jnp.uint8),
    }

    size = {
        256: {
            "canvas_size": 256,
            "small_canvas_size": 192,
            "value_pos": {
                "cards": {
                    "top_left": (0, 0),
                    "bottom_right": (20, 40),
                },
                "suit": {
                    "top_left": (0, 0),
                    "bottom_right": (20, 40),
                },
            },
            "query_pos": {
                "cards": {
                    "top_left": (234, 0),
                    "bottom_right": (254, 40),
                },
                "suit": {
                    "top_left": (234, 0),
                    "bottom_right": (254, 40),
                },
            },
            "action_pos": {
                "left_triangle": {
                    "top_left": (256 // 2 - 65, 228),
                    "bottom_right": (256 // 2 - 35, 254),
                },
                "action": {
                    "top_left": (136, 2),
                    "bottom_right": (246, 30),
                },
                "right_triangle": {
                    "top_left": (256 // 2 + 30, 228),
                    "bottom_right": (256 // 2 + 60, 254),
                },
            },
            "score_pos": {
                "top_left": (86, 2),
                "bottom_right": (171, 30),
            },
            "name_pos": {
                "top_left": (0, 256 - 25),
                "bottom_right": (256, 256),
            },
        },
        128: {
            "canvas_size": 128,
            "small_canvas_size": 96,
            "value_pos": {
                "cards": {
                    "top_left": (0, 0),
                    "bottom_right": (10, 20),
                },
                "suit": {
                    "top_left": (0, 0),
                    "bottom_right": (10, 20),
                },
            },
            "query_pos": {
                "cards": {
                    "top_left": (117, 0),
                    "bottom_right": (127, 20),
                },
                "suit": {
                    "top_left": (117, 0),
                    "bottom_right": (127, 20),
                },
            },
            "action_pos": {
                "left_triangle": {
                    "top_left": (128 // 2 - 32, 114),
                    "bottom_right": (128 // 2 - 17, 127),
                },
                "action": {
                    "top_left": (68, 1),
                    "bottom_right": (123, 15),
                },
                "right_triangle": {
                    "top_left": (128 // 2 + 15, 114),
                    "bottom_right": (128 // 2 + 30, 127),
                },
            },
            "score_pos": {
                "top_left": (43, 1),
                "bottom_right": (85, 15),
            },
            "name_pos": {
                "top_left": (0, 128 - 12),
                "bottom_right": (128, 128),
            },
        },
    }

    def __init__(
        self,
        num_decks=1,
        num_types=2,
        partial_obs: bool = False,
        obs_size: int = 128,
    ):
        self.partial_obs = partial_obs
        self.decksize = 26
        self.num_decks = num_decks
        self.num_types = num_types
        self.num_cards = self.decksize * self.num_decks
        self.max_num = self.num_cards // self.num_types  # number of every type of card
        self.reward_scale = 1.0 / self.num_cards

        self.max_steps_in_episode = 100 + self.num_cards

        self.canvas_size = self.size[obs_size]["canvas_size"]
        self.small_canvas_size = self.size[obs_size]["small_canvas_size"]
        self.canvas_color = self.color["light_gray"]
        self.large_canvas = (
            jnp.zeros((self.canvas_size, self.canvas_size, 3), dtype=jnp.uint8)
            + self.canvas_color
        )
        self.small_canvas = (
            jnp.zeros(
                (self.small_canvas_size, self.small_canvas_size, 3), dtype=jnp.uint8
            )
            + self.color["muted_blue"]
        )

        self.setup_render_templates()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @property
    def name(self) -> str:
        return "CountRecall"

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs stepping of environment.

        This method processes the action and updates the state accordingly.
        """
        new_state, fire_action = process_action(state, action)

        prev_count = state.running_count[state.query_cards[state.timestep]]
        reward = jnp.where(
            fire_action,
            jnp.where(new_state.default_action == prev_count, self.reward_scale, 0.0),
            0.0,
        )
        new_score = state.score + lax.cond(reward > 0, lambda _: 1, lambda _: 0, None)
        running_count = jnp.where(
            fire_action,
            state.running_count.at[state.value_cards[state.timestep]].add(1),
            state.running_count,
        )
        history = jnp.where(
            fire_action,
            state.history.at[state.timestep].set(state.value_cards[state.timestep]),
            state.history,
        )

        new_state = new_state.replace(
            timestep=new_state.timestep + fire_action,
            running_count=running_count,
            history=history,
            score=new_score,
            alreadyMove=new_state.alreadyMove + 1,
        )

        obs = self.get_obs(new_state)
        terminated = jnp.logical_or(
            new_state.timestep == self.num_cards,
            new_state.alreadyMove >= self.max_steps_in_episode,
        )
        infos = {
            "terminated": new_state.timestep >= self.num_cards,  # play all cards
            "truncated": new_state.alreadyMove
            >= self.max_steps_in_episode,  # timelimit
        }
        return obs, new_state, reward, terminated, infos

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        key, key_value, key_query = jax.random.split(key, 3)
        cards = jnp.arange(self.decksize * self.num_decks) % self.num_types
        value_cards = jax.random.permutation(key_value, cards)
        query_cards = jax.random.permutation(key_query, cards)
        running_count = jnp.zeros((self.num_types,))
        history = jnp.zeros(self.num_cards)
        state = EnvState(
            timestep=0,
            value_cards=value_cards,
            query_cards=query_cards,
            running_count=running_count,
            history=history,
            default_action=0,
            num_types=self.num_types,
            score=0,
            alreadyMove=0,
        )
        obs = self.get_obs(state)
        return obs, state

    def setup_render_templates(self):
        """Precompute all possible card templates once during init"""
        if self.canvas_size == 256:
            hist_positions_adjust = 20
            hist_endings_adjust = jnp.array([12, 20])
        elif self.canvas_size == 128:
            hist_positions_adjust = 10
            hist_endings_adjust = jnp.array([8, 12])
        else:
            pass
        # Value/Query card templates
        self.value_templates = self._create_card_templates(
            self.size[self.canvas_size]["value_pos"]["suit"]["top_left"],
            self.size[self.canvas_size]["value_pos"]["suit"]["bottom_right"],
        )
        self.query_templates = self._create_card_templates(
            self.size[self.canvas_size]["query_pos"]["suit"]["top_left"],
            self.size[self.canvas_size]["query_pos"]["suit"]["bottom_right"],
        )

        # History templates with proper vmap dimensions
        num_history = self.decksize * self.num_decks
        hist_positions = jnp.array(
            [
                ((i % 9) * hist_positions_adjust, (i // 9) * hist_positions_adjust)
                for i in range(num_history)
            ]
        )
        hist_endings = hist_positions + hist_endings_adjust

        # Create base canvas for each history position
        base_canvases = jnp.tile(self.small_canvas[None], (num_history, 1, 1, 1))

        # Create vmap with proper axis mapping
        vmap_draw = lambda fn: jax.vmap(fn, in_axes=(0, 0, None, 0))

        self.history_templates = jnp.stack(
            [
                vmap_draw(draw_heart)(
                    hist_positions, hist_endings, self.color["red"], base_canvases
                ),
                vmap_draw(draw_spade)(
                    hist_positions, hist_endings, self.color["black"], base_canvases
                ),
                vmap_draw(draw_club)(
                    hist_positions, hist_endings, self.color["black"], base_canvases
                ),
                vmap_draw(draw_diamond)(
                    hist_positions, hist_endings, self.color["red"], base_canvases
                ),
            ]
        )

    def _create_card_templates(self, top_left, bottom_red):
        """Create templates for a card position (value/query)"""
        bottom_black = (bottom_red[0], bottom_red[1] - 6)
        base = self.large_canvas.copy()
        return jnp.stack(
            [
                draw_heart(top_left, bottom_red, self.color["red"], base),
                draw_spade(top_left, bottom_black, self.color["black"], base),
                draw_club(top_left, bottom_black, self.color["black"], base),
                draw_diamond(top_left, bottom_red, self.color["red"], base),
            ]
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def render(self, state) -> chex.Array:
        large_canvas = self.large_canvas.copy()
        small_canvas = self.small_canvas.copy()

        valid_value = state.timestep < len(state.value_cards)
        value_idx = state.value_cards[state.timestep]
        value_template = self.value_templates[value_idx]
        value_mask = (value_template != self.large_canvas).any(axis=-1, keepdims=True)
        large_canvas = jnp.where(valid_value & value_mask, value_template, large_canvas)

        valid_query = state.timestep < len(state.query_cards)
        query_idx = state.query_cards[state.timestep]
        query_template = self.query_templates[query_idx]
        query_mask = (query_template != self.large_canvas).any(axis=-1, keepdims=True)
        large_canvas = jnp.where(valid_query & query_mask, query_template, large_canvas)

        if not self.partial_obs:
            valid_mask = jnp.arange(len(state.history)) < state.timestep
            card_indices = state.history.astype(int)

            selected = self.history_templates[
                card_indices, jnp.arange(len(state.history))
            ]
            valid_symbols = valid_mask[:, None, None] & jnp.any(
                selected != self.small_canvas[0, 0], axis=-1
            )

            priority = jnp.arange(len(state.history))[:, None, None] * valid_symbols
            last_idx = jnp.argmax(priority, axis=0)
            h, w = jnp.indices(small_canvas.shape[:2])
            small_canvas = jnp.where(
                jnp.any(valid_symbols, axis=0)[..., None],
                selected[last_idx, h, w],
                small_canvas,
            )

        score_top_left = self.size[self.canvas_size]["score_pos"]["top_left"]
        score_bottom_right = self.size[self.canvas_size]["score_pos"]["bottom_right"]
        action_top_left = self.size[self.canvas_size]["action_pos"]["action"][
            "top_left"
        ]
        action_bottom_right = self.size[self.canvas_size]["action_pos"]["action"][
            "bottom_right"
        ]
        large_canvas = draw_number(
            score_top_left,
            score_bottom_right,
            self.color["dark_red"],
            large_canvas,
            state.score,
        )
        large_canvas = draw_number(
            action_top_left,
            action_bottom_right,
            self.color["soft_green"],
            large_canvas,
            state.default_action,
        )
        large_canvas = draw_str(
            self.size[self.canvas_size]["name_pos"]["top_left"],
            self.size[self.canvas_size]["name_pos"]["bottom_right"],
            self.color["bright_blue"],
            large_canvas,
            self.name,
        )

        return draw_sub_canvas(small_canvas, large_canvas)

    def get_obs(self, state: EnvState) -> chex.Array:
        """Returns observation from the state."""
        obs = self.render(state)
        return obs

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(5)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 255, (256, 256, 3), dtype=jnp.uint8)


class CountRecallEasy(CountRecall):
    def __init__(self, partial_obs: bool = False, **kwargs):
        super().__init__(num_decks=1, num_types=2, partial_obs=partial_obs, **kwargs)


class CountRecallMedium(CountRecall):
    def __init__(self, partial_obs: bool = False, **kwargs):
        super().__init__(num_decks=2, num_types=2, partial_obs=partial_obs, **kwargs)


class CountRecallHard(CountRecall):
    def __init__(self, partial_obs: bool = False, **kwargs):
        super().__init__(num_decks=3, num_types=4, partial_obs=partial_obs, **kwargs)
