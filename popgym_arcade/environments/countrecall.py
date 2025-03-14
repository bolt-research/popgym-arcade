from typing import Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces
import functools
from jax import lax
from popgym_arcade.environments.draw_utils import (draw_rectangle,
                                            draw_heart,
                                            draw_spade,
                                            draw_club,
                                            draw_diamond,
                                            draw_number,
                                            draw_str,
                                            draw_sub_canvas)


@struct.dataclass
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


@struct.dataclass
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
        jnp.where(
            action == 1, 
            state.default_action - 1,
            state.default_action
        )
    )
    
    return state.replace(default_action=new_default_action), action == 4


class CountRecall(environment.Environment):
    """
    JAX compatible implementation of CountRecall environment.
    Source: https://github.com/proroklab/popgym/blob/master/popgym/envs/count_recall.py

    ### Description
    A game where the agent is queried on past events.

    The agent is queried on how many times it has observed a specific event in the past. This tests long-term 
    order-agnostic memory like sets. The agent is dealt a card, and then asked how many times it has seen a 
    specific card in the past. The agent must answer correctly to receive a reward.
    
    CountRecall is a card counting game. The agent is dealt a card, and then asked how many times it has seen.
    There are three different levels of difficulty: easy, medium, and hard. The easy level has one deck of cards
    with two types of cards, the medium level has two decks of cards with two types of cards, and the hard level
    has three decks of cards with four types of cards. A query card is located in the top right corner of the 
    screen, and the value card is located in the top left corner of the screen. The agent can increase or decrease
    the count of the card by pressing the up or down action. The agent can lock in the count by pressing the 
    confirm action. The score will increase by one if the agent locks in the correct count. 


    ### Observation
    The observation space is a chex.Array with shape `(256, 256, 3)`
    Current state is rendered into a matrix using multiple graphical elements to form a visual observation.
    The entire observation consists of a large canvas with a smaller canvas embedded within it.
    The smaller canvas primarily displays the game interface,
    while the larger canvas shows additional information, such as the score.

    In the smaller canvas, the following elements are displayed:
    - The sequence of historical cards observed by the agent.

    In the larger canvas, the following elements are displayed:
    - The suit of the card dealt to the agent.
    - The suit of the card queried to the agent.
    - The action controller, which displays the current action selected by the agent.
    - The score of the agent has accumulated so far.
    - The name of the environment.

    In the fully observable version of the environment, the historical cards are displayed in the smaller canvas.
    In the partially observable version of the environment, the historical cards are not displayed.

    ### Actions
    | Action | Description                         |
    |--------|-------------------------------------|
    | 0      | Up (Increase the count)             |
    | 1      | Down (Decrease the count)           |
    | 2      | Left (No-op)                        |
    | 3      | Right (No-op)                       |
    | 4      | Confirm (Lock in current selection) |

    ### Rewards
    The agent receives a reward of 1.0 / num_cards for each correct answer only when the agent confirms 
    the selection (action 4). The agent receives a reward of 0.0 otherwise.

    ### Termination & Truncation
    The episode terminates when the agent has made `max_steps_in_episode` moves or when the agent has seen 
    all the cards.

    ### Args
    num_decks: The number of decks of cards in the game. Easy: 1, Medium: 2, Hard: 3.
    num_types: The number of types of cards in the game. Easy: 2, Medium: 2, Hard: 4.
    partial_obs: Whether the environment is partially observable or not.
    max_steps_in_episode: The maximum number of steps in an episode.
    """

    color = {
        "red": jnp.array([1, 0, 0]),
        "dark_red": jnp.array([0.75, 0.10, 0.10]),
        "black": jnp.array([0, 0, 0]),
        "white": jnp.array([1, 1, 1]),
        "light_blue": jnp.array([0.29, 0.75, 0.94]),
        "dark_blue": jnp.array([0.12, 0.47, 0.77]),
        "soft_green": jnp.array([0.56, 0.93, 0.56]),
        "green": jnp.array([0.56, 0.93, 0.56]),
        "soft_beige": jnp.array([0.96, 0.88, 0.79]),
        "olive_green": jnp.array([0.5, 0.63, 0.39]),
        "maroon": jnp.array([0.69, 0.19, 0.38]),
        "teal": jnp.array([0.0, 0.5, 0.5]),
        "light_gray": jnp.array([0.96, 0.96, 0.96]),
        "bright_blue": jnp.array([0.29, 0.75, 0.94]),
        "muted_blue": jnp.array([0.6, 0.7, 0.8]),
    }
    value_pos = {
        "cards": {
            "top_left": (0, 0),
            "bottom_right": (20, 40),
        },
        "suit": {
            "top_left": (0, 0),
            "bottom_right": (20, 40),
        },
    }
    query_pos = {
        "cards": {
            "top_left": (234, 0),
            "bottom_right": (254, 40),
        },
        "suit": {
            "top_left": (234, 0),
            "bottom_right": (254, 40),
        },
    }
    action_pos = {
        "left_triangle": {
            "top_left": (256//2 - 65, 228),
            "bottom_right": (256//2 - 35, 254),
        },
        "action": {
            "top_left": (136, 2),
            "bottom_right": (246, 30),
        },
        "right_triangle": {
            "top_left": (256//2 + 30, 228),
            "bottom_right": (256//2 + 60, 254),
        },
    }
    score_pos = {
        "top_left": (86, 2),
        "bottom_right": (171, 30),
    }
    name_pos = {
        "top_left": (0, 256-25),
        "bottom_right": (256, 256),
    }
    
    def __init__(
        self, 
        num_decks=1, 
        num_types=2, 
        partial_obs: bool = False
    ):
        self.partial_obs = partial_obs
        self.decksize = 26
        self.num_decks = num_decks
        self.num_types = num_types
        self.num_cards = self.decksize * self.num_decks
        self.max_num = self.num_cards // self.num_types  # number of every type of card
        self.reward_scale = 1.0 / self.num_cards

        self.max_steps_in_episode = 100 + self.num_cards

        self.canva_size = 256
        self.small_canva_size = 192
        self.canva_color = self.color["light_gray"]
        self.large_canva = jnp.zeros((self.canva_size, self.canva_size, 3)) + self.canva_color
        self.small_canva = jnp.zeros((self.small_canva_size, self.small_canva_size, 3)) + self.color["muted_blue"]

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
        params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs stepping of environment.

        This method processes the action and updates the state accordingly.
        """
        new_state, fire_action = process_action(state, action)

        prev_count = state.running_count[state.query_cards[state.timestep]]
        reward = jnp.where(
            fire_action,
            jnp.where(new_state.default_action == prev_count, self.reward_scale, 0.0),
            0.0
        )
        new_score = state.score + lax.cond(reward > 0, lambda _: 1, lambda _: 0, None)
        running_count = jnp.where(
            fire_action,
            state.running_count.at[state.value_cards[state.timestep]].add(1),
            state.running_count
        )
        history = jnp.where(
            fire_action,
            state.history.at[state.timestep].set(state.value_cards[state.timestep]),
            state.history
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
            new_state.alreadyMove >= self.max_steps_in_episode
        )
        return obs, new_state, reward, terminated, {}

    def reset_env(
        self,
        key: chex.PRNGKey, 
        params: EnvParams
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
        # Value/Query card templates
        self.value_templates = self._create_card_templates(
            self.value_pos["suit"]["top_left"],
            self.value_pos["suit"]["bottom_right"]
        )
        self.query_templates = self._create_card_templates(
            self.query_pos["suit"]["top_left"],
            self.query_pos["suit"]["bottom_right"]
        )

        # History templates with proper vmap dimensions
        num_history = self.decksize * self.num_decks
        hist_positions = jnp.array([((i%9)*20, (i//9)*20) for i in range(num_history)])
        hist_endings = hist_positions + jnp.array([12, 20])
        
        # Create base canvas for each history position
        base_canvases = jnp.tile(self.small_canva[None], (num_history, 1, 1, 1))

        # Create vmap with proper axis mapping
        vmap_draw = lambda fn: jax.vmap(fn, in_axes=(0, 0, None, 0))
        
        self.history_templates = jnp.stack([
            vmap_draw(draw_heart)(hist_positions, hist_endings, self.color["red"], base_canvases),
            vmap_draw(draw_spade)(hist_positions, hist_endings, self.color["black"], base_canvases),
            vmap_draw(draw_club)(hist_positions, hist_endings, self.color["black"], base_canvases),
            vmap_draw(draw_diamond)(hist_positions, hist_endings, self.color["red"], base_canvases)
        ])

    def _create_card_templates(self, top_left, bottom_red):
        """Create templates for a card position (value/query)"""
        bottom_black = (bottom_red[0], bottom_red[1] - 6)
        base = self.large_canva.copy()
        return jnp.stack([
            draw_heart(top_left, bottom_red, self.color["red"], base),
            draw_spade(top_left, bottom_black, self.color["black"], base),
            draw_club(top_left, bottom_black, self.color["black"], base),
            draw_diamond(top_left, bottom_red, self.color["red"], base)
        ])

    @functools.partial(jax.jit, static_argnums=(0,))
    def render(self, state) -> chex.Array:
        large_canva = self.large_canva.copy()
        small_canva = self.small_canva.copy()

        valid_value = state.timestep < len(state.value_cards)
        value_idx = state.value_cards[state.timestep]
        value_template = self.value_templates[value_idx]
        value_mask = (value_template != self.large_canva).any(axis=-1, keepdims=True)
        large_canva = jnp.where(
            valid_value & value_mask,
            value_template,
            large_canva
        )

        valid_query = state.timestep < len(state.query_cards)
        query_idx = state.query_cards[state.timestep]
        query_template = self.query_templates[query_idx]
        query_mask = (query_template != self.large_canva).any(axis=-1, keepdims=True)
        large_canva = jnp.where(
            valid_query & query_mask,
            query_template,
            large_canva
        )

        if not self.partial_obs:
            valid_mask = jnp.arange(len(state.history)) < state.timestep
            card_indices = state.history.astype(int)
            
            selected = self.history_templates[card_indices, jnp.arange(len(state.history))]
            valid_symbols = valid_mask[:, None, None] & jnp.any(selected != self.small_canva[0,0], axis=-1)
            
            priority = jnp.arange(len(state.history))[:, None, None] * valid_symbols
            last_idx = jnp.argmax(priority, axis=0)
            h, w = jnp.indices(small_canva.shape[:2])
            small_canva = jnp.where(
                jnp.any(valid_symbols, axis=0)[..., None],
                selected[last_idx, h, w],
                small_canva
            )

        score_top_left = self.score_pos["top_left"]
        score_bottom_right = self.score_pos["bottom_right"]
        action_top_left = self.action_pos["action"]["top_left"]
        action_bottom_right = self.action_pos["action"]["bottom_right"]
        large_canva = draw_number(
            score_top_left, 
            score_bottom_right, 
            self.color["dark_red"], 
            large_canva, 
            state.score
            )
        large_canva = draw_number(
            action_top_left, 
            action_bottom_right, 
            self.color["soft_green"], 
            large_canva, 
            state.default_action,
            )
        large_canva = draw_str(
            self.name_pos["top_left"],
            self.name_pos["bottom_right"],
            self.color["bright_blue"],
            large_canva,
            self.name,
            )
        large_canva = draw_sub_canvas(small_canva, large_canva)
        
        return large_canva

    def get_obs(self, state: EnvState) -> chex.Array:
        """Returns observation from the state."""
        obs = self.render(state)
        return obs

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(5)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(jnp.array(0,), jnp.array(1,), (256, 256, 3), dtype=jnp.float32)


class CountRecallEasy(CountRecall):
    def __init__(self, partial_obs: bool = False):
        super().__init__(num_decks=1, num_types=2, partial_obs=partial_obs)

class CountRecallMedium(CountRecall):
    def __init__(self, partial_obs: bool = False):
        super().__init__(num_decks=2, num_types=2, partial_obs=partial_obs)

class CountRecallHard(CountRecall):
    def __init__(self, partial_obs: bool = False):
        super().__init__(num_decks=3, num_types=4, partial_obs=partial_obs)
