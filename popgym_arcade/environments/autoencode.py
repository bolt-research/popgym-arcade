import functools
from typing import Optional, Tuple

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
    draw_spade,
    draw_str,
    draw_sub_canvas,
)


@dataclass(frozen=True)
class EnvState:
    timestep: int
    cards: chex.Array
    score: int
    count: int
    default_action: int


@dataclass(frozen=True)
class EnvParams:
    pass


@jax.jit
def process_action(state: EnvState, action: int) -> Tuple[EnvState, bool]:
    new_default_action = jnp.where(
        action == 2,
        state.default_action + 1,
        jnp.where(action == 3, state.default_action - 1, state.default_action),
    )
    new_default_action = jnp.where(
        new_default_action < 0,
        3,
        jnp.where(new_default_action > 3, 0, new_default_action),
    )
    return state.replace(default_action=new_default_action), action == 4


class AutoEncode(environment.Environment):
    """
    JAX compilable environment for AutoEncode.
    Source: https://github.com/proroklab/popgym/blob/master/popgym/envs/autoencode.py

    ### Description
    **AutoEncode** is a game where the agent must **recall** and **output** a sequence of cards **in reverse order**.  
    - The sequence consists of cards from four suits: **Club (♣)**, **Spade (♠)**, **Heart (♥)**, and **Diamond (♦)**.  
    - Example: If the agent sees `[♣, ♠, ♥]` in the **watch stage**, it must output `[♥, ♠, ♣]` in the **play stage**.  

    ---

    **How to play**
    1. **Watch Stage** – The agent is shown a sequence of cards, one at a time.  
    2. **Play Stage** – The agent outputs the sequence **in reverse order**.  
    3. Reward: `1.0 / num_cards` for each correct card in the **play stage**.  
    4. No rewards are given during the **watch stage**.  

    
    **Difficulty Levels**  
    - **Easy** – `1 deck`, 26 cards
    - **Medium** – `2 decks`, 52 cards
    - **Hard** – `3 decks`, 78 cards

    ### Action Space
    | Action | Description                         |
    |--------|-------------------------------------|
    | `0`    | Up *(No-op)*                        |
    | `1`    | Down *(No-op)*                      |
    | `2`    | Left *(Cycle options left)*         |
    | `3`    | Right *(Cycle options right)*       |
    | `4`    | Confirm *(Lock in current selection)* |

    ### Observation Space
    - **Image Embeddings**:  
    - `256x256x3 (192x192x3)` or `128x128x3 (96x96x3)` depending on observation size.
    - **Query Suits**: Always one of `[♣, ♠, ♥, ♦]` at top-left during watch stage.
    - **History Suits**: Query Suits eventually becomes a sequence of history suits.
    - **Score**: Shown at **top-middle** of the screen.  
    - **Current Suit**: User's current action (chosen suit), shown at **top-right** in watch/play stage. 

    **MDP Version**  
    - Agent can see full sequence of **History Suits**, **Query Suits**, and **Current Suit**.

    **POMDP Version**  
    - Agent only sees the **Query Suits**, and **Current Suit**.


    ### Reward
    - Reward Scale: 1.0 / (num_cards)
    In watch stage, the agent will not receive any reward.
    In play stage, the agent will receive reward scale if the agent's action is correct.

    ### Termination & Truncation
    - Termination: The episode terminates when the agent has played all cards
    - Truncation: The episode will be truncated after 140 steps + num_cards

    ### Args
    num_decks: The number of decks of cards to use. Easy: 1, Medium: 2, Hard: 3
    partial_obs: Whether to use POMDP version of the environment or not.
    max_steps_in_episode: The maximum number of steps in an episode.
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
    }
    size = {
        256: {
            "canvas_size": 256,
            "small_canvas_size": 192,
            "value_cards_pos": {
                "top_left": (0, 0),
                "bottom_right": (20, 40),
            },
            "value_suit_pos": {
                "top_left": (0, 0),
                "bottom_right": (20, 40),
            },
            "left_triangle_pos": {
                "top_left": (92, 224),
                "bottom_right": (112, 256),
            },
            "current_suit_pos": {
                "top_left": (234, 0),
                "bottom_right": (254, 40),
            },
            "right_triangle_pos": {
                "top_left": (152, 224),
                "bottom_right": (172, 256),
            },
            "name_pos": {
                "top_left": (0, 256 - 25),
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
            "value_cards_pos": {
                "top_left": (0, 0),
                "bottom_right": (10, 20),
            },
            "value_suit_pos": {
                "top_left": (0, 0),
                "bottom_right": (10, 20),
            },
            "left_triangle_pos": {
                "top_left": (46, 112),
                "bottom_right": (56, 128),
            },
            "current_suit_pos": {
                "top_left": (117, 0),
                "bottom_right": (127, 20),
            },
            "right_triangle_pos": {
                "top_left": (76, 112),
                "bottom_right": (86, 128),
            },
            "name_pos": {
                "top_left": (0, 128 - 12),
                "bottom_right": (128, 128),
            },
            "score": {
                "top_left": (43, 1),
                "bottom_right": (85, 15),
            },
        },
    }

    def __init__(
        self,
        num_decks=1,
        partial_obs=False,
        obs_size: int = 128,
    ):
        super().__init__()
        self.partial_obs = partial_obs
        self.num_suits = 4
        self.decksize = 26
        self.num_decks = num_decks
        self.canvas_size = self.size[obs_size]["canvas_size"]
        self.canvas_color = self.color["light_blue"]
        self.large_canvas = jnp.full(
            (self.canvas_size, self.canvas_size, 3), self.canvas_color, dtype=jnp.uint8
        )
        self.small_canvas_size = self.size[obs_size]["small_canvas_size"]
        self.small_canvas = jnp.full(
            (self.small_canvas_size, self.small_canvas_size, 3),
            self.canvas_color,
            dtype=jnp.uint8,
        )

        self.max_steps_in_episode = self.decksize * self.num_decks * 2 * 5
        self.setup_render_templates()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @property
    def name(self) -> str:
        return "AutoEncode"

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step of the environment."""
        new_state, fire_action = process_action(state, action)
        num_cards = self.decksize * self.num_decks
        reward = 0

        fire_action = jnp.where(new_state.timestep <= num_cards, True, fire_action)

        reward_scale = 1.0 / (num_cards)

        terminated = jnp.logical_or(
            new_state.count >= num_cards,  # play all cards
            new_state.timestep >= self.max_steps_in_episode,  # timelimit
        )
        play = new_state.timestep >= num_cards

        reward = jnp.where(
            fire_action,
            jnp.where(
                jnp.flip(new_state.cards, axis=0)[new_state.count]
                == new_state.default_action,
                reward_scale,
                0,
            ),
            0,
        )

        reward = jnp.where(
            play,
            reward,
            0,
        )

        new_state = new_state.replace(
            timestep=new_state.timestep + 1,
            cards=new_state.cards,
            score=new_state.score
            + lax.cond(reward > 0, lambda _: 1, lambda _: 0, None),
            count=new_state.count + jnp.where(jnp.logical_and(fire_action, play), 1, 0),
            default_action=new_state.default_action,
        )
        obs = self.get_obs(new_state)
        infos = {
            "terminated": new_state.count >= num_cards,
            "truncated": new_state.timestep >= self.max_steps_in_episode,
        }
        return obs, new_state, reward, terminated, infos

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        cards = jnp.arange(self.decksize * self.num_decks) % self.num_suits
        cards = jax.random.permutation(key, cards)
        state = EnvState(
            timestep=0,
            cards=cards,
            score=0,
            count=0,
            default_action=0,
        )
        obs = self.get_obs(state)
        return obs, state

    def setup_render_templates(self):
        if self.canvas_size == 256:
            value_suit_adjust = 6
            hist_adjust = 20
            hist_endings_adjust = jnp.array([12, 20])
        elif self.canvas_size == 128:
            value_suit_adjust = 3
            hist_adjust = 10
            hist_endings_adjust = jnp.array([8, 14])
        else:
            pass
        base_large = self.large_canvas.copy()
        base_small = self.small_canvas.copy()

        value_suit_top_left = self.size[self.canvas_size]["value_suit_pos"]["top_left"]
        value_suit_bottom_right = self.size[self.canvas_size]["value_suit_pos"][
            "bottom_right"
        ]

        self.value_card_templates = jnp.stack(
            [
                draw_heart(
                    value_suit_top_left,
                    value_suit_bottom_right,
                    self.color["red"],
                    base_large,
                ),
                draw_spade(
                    value_suit_top_left,
                    (
                        value_suit_bottom_right[0],
                        value_suit_bottom_right[1] - value_suit_adjust,
                    ),
                    self.color["black"],
                    base_large,
                ),
                draw_club(
                    value_suit_top_left,
                    (
                        value_suit_bottom_right[0],
                        value_suit_bottom_right[1] - value_suit_adjust,
                    ),
                    self.color["black"],
                    base_large,
                ),
                draw_diamond(
                    value_suit_top_left,
                    value_suit_bottom_right,
                    self.color["red"],
                    base_large,
                ),
            ]
        )

        hist_positions = jnp.array(
            [
                ((i % 9) * hist_adjust, (i // 9) * hist_adjust)
                for i in range(self.decksize * self.num_decks)
            ]
        )

        hist_endings_red = hist_positions + hist_endings_adjust
        hist_endings_black = hist_positions + hist_endings_adjust

        vmap_draw = lambda fn: jax.vmap(fn, in_axes=(0, 0, None, None))
        self.history_card_templates = jnp.stack(
            [
                vmap_draw(draw_heart)(
                    hist_positions, hist_endings_red, self.color["red"], base_small
                ),
                vmap_draw(draw_spade)(
                    hist_positions, hist_endings_black, self.color["black"], base_small
                ),
                vmap_draw(draw_club)(
                    hist_positions, hist_endings_black, self.color["black"], base_small
                ),
                vmap_draw(draw_diamond)(
                    hist_positions, hist_endings_red, self.color["red"], base_small
                ),
            ],
            axis=1,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def render(self, state: EnvState) -> chex.Array:
        large_canvas = self.large_canvas.copy()
        small_canvas = self.small_canvas.copy()

        valid_current_card = state.timestep < self.decksize * self.num_decks
        current_suit = jax.lax.select(
            valid_current_card, state.cards[state.timestep].astype(int), 0
        )
        large_canvas = jnp.where(
            valid_current_card, self.value_card_templates[current_suit], large_canvas
        )

        def render_history(canvas):
            num_cards = self.decksize * self.num_decks
            valid_mask = jnp.arange(num_cards) < state.timestep

            card_indices = state.cards.astype(int)[:, None, None, None, None]

            selected = jnp.take_along_axis(
                self.history_card_templates, card_indices, axis=1
            ).squeeze(1)

            bg_color = self.small_canvas[0, 0]
            valid_symbol = valid_mask[:, None, None] & jnp.any(
                selected != bg_color, axis=-1
            )

            card_priority = jnp.arange(num_cards)[:, None, None] * valid_symbol
            last_valid_idx = jnp.argmax(card_priority, axis=0)
            any_valid = jnp.any(valid_symbol, axis=0)

            h, w = jnp.indices((self.small_canvas_size, self.small_canvas_size))
            final_colors = selected[last_valid_idx, h, w]

            return jnp.where(any_valid[..., None], final_colors, canvas)

        small_canvas = lax.cond(
            self.partial_obs, lambda: small_canvas, lambda: render_history(small_canvas)
        )

        a_pos = (
            self.size[self.canvas_size]["current_suit_pos"]["top_left"],
            self.size[self.canvas_size]["current_suit_pos"]["bottom_right"],
        )
        action_color = jnp.array(
            [
                self.color["red"],
                self.color["black"],
                self.color["black"],
                self.color["red"],
            ]
        )[state.default_action]

        large_canvas = jax.lax.switch(
            state.default_action,
            [
                lambda p0, p1, c, cnvs: draw_heart(p0, p1, c, cnvs),
                lambda p0, p1, c, cnvs: draw_spade(p0, p1, c, cnvs),
                lambda p0, p1, c, cnvs: draw_club(p0, p1, c, cnvs),
                lambda p0, p1, c, cnvs: draw_diamond(p0, p1, c, cnvs),
            ],
            a_pos[0],
            a_pos[1],
            action_color,
            large_canvas,
        )

        large_canvas = draw_number(
            self.size[self.canvas_size]["score"]["top_left"],
            self.size[self.canvas_size]["score"]["bottom_right"],
            self.color["bright_red"],
            large_canvas,
            state.score,
        )
        large_canvas = draw_str(
            self.size[self.canvas_size]["name_pos"]["top_left"],
            self.size[self.canvas_size]["name_pos"]["bottom_right"],
            self.color["neon_pink"],
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
        return spaces.Discrete(self.num_suits)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 255, (256, 256, 3), dtype=jnp.uint8)


class AutoEncodeEasy(AutoEncode):
    def __init__(self, **kwargs):
        super().__init__(num_decks=1, **kwargs)


class AutoEncodeMedium(AutoEncode):
    def __init__(self, **kwargs):
        super().__init__(num_decks=2, **kwargs)


class AutoEncodeHard(AutoEncode):
    def __init__(self, **kwargs):
        super().__init__(num_decks=3, **kwargs)
