"""JAX implementation of Tetris MinAtar environment."""
"""
Tetris
    Nonstationary: Random block order (shuffle the sequence of tetromino pieces)
    POMDP: Hide upcoming pieces or part of the board
"""
from typing import Any
import functools
import jax
import jax.numpy as jnp
from chex import dataclass

from gymnax.environments import environment, spaces
from popgym_arcade.environments.draw_utils import (
    draw_number,
    draw_str,
    draw_sub_canvas,
    draw_rectangle,
)


@dataclass(frozen=True)
class EnvState(environment.EnvState):
    board: jax.Array  # 20x10 board
    current_piece: int  # Current tetromino type (0-6)
    current_x: int  # Current piece x position
    current_y: int  # Current piece y position
    current_rotation: int  # Current piece rotation (0-3)
    next_pieces: jax.Array  # Array of next tetromino types for preview
    piece_queue: jax.Array  # Queue of upcoming pieces for nonstationarity
    queue_index: int  # Current index in the piece queue
    lines_cleared: int  # Total lines cleared
    score: int  # Game score
    time: int  # Time step
    terminal: bool  # Game over flag
    drop_timer: int  # Timer for automatic piece dropping
    lock_timer: int  # Timer for piece locking
    just_locked: bool = False
    frames_since_spawn: int = 0  # Track frames since new piece spawned

@dataclass(frozen=True)
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 5000
    drop_interval: int = 30  # Steps between automatic drops
    lock_delay: int = 30  # Steps before piece locks
    auto_drop_speed: int = 1  # Number of cells to drop automatically per step
    soft_drop_speed: int = 2  # Number of cells to drop when soft dropping


# Tetromino definitions: 7 pieces, 4 rotations each
# Each piece is defined as a 4x4 grid
TETROMINOES = jnp.array([
    # I-piece
    [[[0, 0, 0, 0],
      [1, 1, 1, 1],
      [0, 0, 0, 0],
      [0, 0, 0, 0]],
     [[0, 0, 1, 0],
      [0, 0, 1, 0],
      [0, 0, 1, 0],
      [0, 0, 1, 0]],
     [[0, 0, 0, 0],
      [0, 0, 0, 0],
      [1, 1, 1, 1],
      [0, 0, 0, 0]],
     [[0, 1, 0, 0],
      [0, 1, 0, 0],
      [0, 1, 0, 0],
      [0, 1, 0, 0]]],
    # O-piece
    [[[0, 0, 0, 0],
      [0, 1, 1, 0],
      [0, 1, 1, 0],
      [0, 0, 0, 0]],
     [[0, 0, 0, 0],
      [0, 1, 1, 0],
      [0, 1, 1, 0],
      [0, 0, 0, 0]],
     [[0, 0, 0, 0],
      [0, 1, 1, 0],
      [0, 1, 1, 0],
      [0, 0, 0, 0]],
     [[0, 0, 0, 0],
      [0, 1, 1, 0],
      [0, 1, 1, 0],
      [0, 0, 0, 0]]],
    # T-piece
    [[[0, 0, 0, 0],
      [0, 1, 0, 0],
      [1, 1, 1, 0],
      [0, 0, 0, 0]],
     [[0, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 1, 1, 0],
      [0, 1, 0, 0]],
     [[0, 0, 0, 0],
      [0, 0, 0, 0],
      [1, 1, 1, 0],
      [0, 1, 0, 0]],
     [[0, 0, 0, 0],
      [0, 1, 0, 0],
      [1, 1, 0, 0],
      [0, 1, 0, 0]]],
    # S-piece
    [[[0, 0, 0, 0],
      [0, 1, 1, 0],
      [1, 1, 0, 0],
      [0, 0, 0, 0]],
     [[0, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 1, 1, 0],
      [0, 0, 1, 0]],
     [[0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 1, 1, 0],
      [1, 1, 0, 0]],
     [[0, 0, 0, 0],
      [1, 0, 0, 0],
      [1, 1, 0, 0],
      [0, 1, 0, 0]]],
    # Z-piece
    [[[0, 0, 0, 0],
      [1, 1, 0, 0],
      [0, 1, 1, 0],
      [0, 0, 0, 0]],
     [[0, 0, 0, 0],
      [0, 0, 1, 0],
      [0, 1, 1, 0],
      [0, 1, 0, 0]],
     [[0, 0, 0, 0],
      [0, 0, 0, 0],
      [1, 1, 0, 0],
      [0, 1, 1, 0]],
     [[0, 0, 0, 0],
      [0, 1, 0, 0],
      [1, 1, 0, 0],
      [1, 0, 0, 0]]],
    # J-piece
    [[[0, 0, 0, 0],
      [1, 0, 0, 0],
      [1, 1, 1, 0],
      [0, 0, 0, 0]],
     [[0, 0, 0, 0],
      [0, 1, 1, 0],
      [0, 1, 0, 0],
      [0, 1, 0, 0]],
     [[0, 0, 0, 0],
      [0, 0, 0, 0],
      [1, 1, 1, 0],
      [0, 0, 1, 0]],
     [[0, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 1, 0, 0],
      [1, 1, 0, 0]]],
    # L-piece
    [[[0, 0, 0, 0],
      [0, 0, 1, 0],
      [1, 1, 1, 0],
      [0, 0, 0, 0]],
     [[0, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 1, 0, 0],
      [0, 1, 1, 0]],
     [[0, 0, 0, 0],
      [0, 0, 0, 0],
      [1, 1, 1, 0],
      [1, 0, 0, 0]],
     [[0, 0, 0, 0],
      [1, 1, 0, 0],
      [0, 1, 0, 0],
      [0, 1, 0, 0]]]
])


def check_collision(board: jax.Array, piece: jax.Array, x: int, y: int) -> bool:
    """Check if piece placement would cause collision."""
    # Create coordinate grids for the 4x4 piece
    py_grid, px_grid = jnp.meshgrid(jnp.arange(4), jnp.arange(4), indexing='ij')
    
    # Calculate board positions for each piece cell
    board_x = x + px_grid
    board_y = y + py_grid
    
    # Check bounds violations
    out_of_bounds = jnp.logical_or(
        jnp.logical_or(board_x < 0, board_x >= 10),
        jnp.logical_or(board_y < 0, board_y >= 20)
    )
    
    # For valid positions, check collision with existing blocks
    valid_positions = jnp.logical_not(out_of_bounds)
    
    # Clamp coordinates to valid range for safe indexing
    safe_x = jnp.clip(board_x, 0, 9)
    safe_y = jnp.clip(board_y, 0, 19)
    
    # Check collisions only where piece has blocks and position is valid
    piece_blocks = piece == 1
    board_occupied = board[safe_y, safe_x] > 0  # Any non-zero value means occupied
    
    # Collision occurs if piece block overlaps with occupied board cell or goes out of bounds
    collisions = piece_blocks * (jnp.logical_or(out_of_bounds, board_occupied))
    
    # Return True if any collision exists
    return jnp.any(collisions)


def place_piece(board: jax.Array, piece: jax.Array, x: int, y: int, piece_type: int) -> jax.Array:
    """Place piece on board."""
    # Create coordinate grids for the 4x4 piece
    py_grid, px_grid = jnp.meshgrid(jnp.arange(4), jnp.arange(4), indexing='ij')
    
    # Calculate board positions for each piece cell
    board_x = x + px_grid
    board_y = y + py_grid
    
    # Check bounds
    within_bounds = jnp.logical_and(
        jnp.logical_and(board_x >= 0, board_x < 10),
        jnp.logical_and(board_y >= 0, board_y < 20)
    )
    
    # Only place blocks where piece has a block AND position is within bounds
    should_place = jnp.logical_and(piece == 1, within_bounds)
    
    # Clamp coordinates to valid range for safe indexing
    safe_x = jnp.clip(board_x, 0, 9)
    safe_y = jnp.clip(board_y, 0, 19)
    
    # Create update values - piece_type + 1 where we should place piece (so we can distinguish from empty)
    # We add 1 because piece types are 0-6, but we want board values 1-7 (0 = empty)
    updates = jnp.where(should_place, piece_type + 1, 0)
    
    # Flatten everything for scatter operations
    flat_y = safe_y.flatten()
    flat_x = safe_x.flatten()
    flat_updates = updates.flatten()
    flat_should_place = should_place.flatten()
    
    # Create linear indices
    linear_indices = flat_y * 10 + flat_x
    
    # Use scatter_max to place pieces (handles overlaps by taking max)
    flat_board = board.flatten()
    updated_flat = flat_board.at[linear_indices].max(flat_updates * flat_should_place.astype(jnp.int32))
    
    new_board = updated_flat.reshape(20, 10)
    
    return new_board


def clear_lines(board: jax.Array) -> tuple[jax.Array, int]:
    """Clear completed lines and return new board and number of lines cleared."""
    # Check which lines are complete (all cells non-zero, since board stores piece types 1-7)
    line_complete = jnp.all(board > 0, axis=1)
    lines_cleared = jnp.sum(line_complete)
    
    # Create new board by filtering out complete lines using JAX operations
    # We'll reconstruct the board by copying lines that aren't complete
    def copy_line(i, new_board_info):
        new_board, write_idx = new_board_info
        # If this line is not complete, copy it to the new position
        should_copy = ~line_complete[i]
        new_line = jax.lax.select(should_copy, board[i], jnp.zeros(10, dtype=jnp.int32))
        new_board = new_board.at[write_idx].set(new_line)
        # Only increment write index if we actually copied a line
        new_write_idx = write_idx + should_copy.astype(jnp.int32)
        return new_board, new_write_idx
    
    # Start with empty board and copy incomplete lines from bottom to top
    empty_board = jnp.zeros_like(board)
    final_board, _ = jax.lax.fori_loop(
        0, 20, copy_line, (empty_board, lines_cleared)
    )
    
    return final_board, lines_cleared


def step_piece(
    state: EnvState,
    action: jax.Array,
    params: EnvParams,
) -> tuple[EnvState, jax.Array]:
    """Step the current piece based on action."""
    current_piece_shape = TETROMINOES[state.current_piece, state.current_rotation]
    
    # Check if we should use enhanced soft drop on second frame or later AND action is down (1)
    should_double_soft_drop = jnp.logical_and(
        state.frames_since_spawn >= 1,
        action == 1
    )

    # Handle different actions
    new_x = state.current_x
    new_y = state.current_y
    new_rotation = state.current_rotation

    # Action 0 (up): Rotate piece 90° clockwise
    new_rotation = jax.lax.select(action == 0, (state.current_rotation + 1) % 4, new_rotation)
    # Action 1 (down): Soft drop (move piece down) OR enhanced soft drop if second frame or later
    # For enhanced soft drop, try soft_drop_speed cells first, then fallback to 1 cell if collision
    def handle_down_action():
        # Try moving soft_drop_speed cells down
        try_max_cells = state.current_y + params.soft_drop_speed
        # Try moving 1 cell down
        try_1_cell = state.current_y + 1
        
        # Check if soft_drop_speed cells is valid
        can_move_max = jnp.logical_not(check_collision(
            state.board, 
            TETROMINOES[state.current_piece, state.current_rotation], 
            state.current_x, 
            try_max_cells
        ))
        
        # If can move soft_drop_speed cells, use that; otherwise try 1 cell
        return jax.lax.select(can_move_max, try_max_cells, try_1_cell)
    
    new_y = jax.lax.select(
        should_double_soft_drop,
        handle_down_action(),
        jax.lax.select(action == 1, state.current_y + 1, new_y)
    )
    # Action 2 (left): Move piece left
    new_x = jax.lax.select(action == 2, state.current_x - 1, new_x)
    # Action 3 (right): Move piece right  
    new_x = jax.lax.select(action == 3, state.current_x + 1, new_x)
    # Action 4 (noop): Do nothing - handled by default values above

    # Get new piece shape after potential rotation
    new_piece_shape = TETROMINOES[state.current_piece, new_rotation]

    # Check if move is valid
    move_valid = jnp.logical_not(check_collision(state.board, new_piece_shape, new_x, new_y))

    # Apply move only if valid
    final_x = jax.lax.select(move_valid, new_x, state.current_x)
    final_y = jax.lax.select(move_valid, new_y, state.current_y)
    final_rotation = jax.lax.select(move_valid, new_rotation, state.current_rotation)

    # Automatic falling: piece falls down based on auto_drop_speed
    def try_auto_drop(current_y, drop_amount):
        """Try to drop piece by drop_amount, fallback to 1 cell if blocked."""
        target_y = current_y + drop_amount
        can_drop_full = jnp.logical_not(check_collision(
            state.board, 
            TETROMINOES[state.current_piece, final_rotation], 
            final_x, 
            target_y
        ))
        # If can't drop full amount, try 1 cell
        fallback_y = current_y + 1
        return jax.lax.select(can_drop_full, target_y, fallback_y)
    
    auto_drop_y = try_auto_drop(final_y, params.auto_drop_speed)
    auto_drop_valid = jnp.logical_not(check_collision(
        state.board, 
        TETROMINOES[state.current_piece, final_rotation], 
        final_x, 
        auto_drop_y
    ))

    # Apply automatic drop if valid (unless player already moved down with action 1)
    already_moved_down = jnp.logical_or(action == 1, should_double_soft_drop)
    final_y = jax.lax.select(
        jnp.logical_and(auto_drop_valid, jnp.logical_not(already_moved_down)),
        auto_drop_y,
        final_y
    )

    # Check if piece should lock (can't move down anymore)
    piece_shape = TETROMINOES[state.current_piece, final_rotation]
    can_drop_further = jnp.logical_not(check_collision(state.board, piece_shape, final_x, final_y + 1))

    should_lock = jnp.logical_not(can_drop_further)

    return (
        state.replace(
            current_x=final_x,
            current_y=final_y,
            current_rotation=final_rotation,
            drop_timer=0,  # Not used in new mechanics
            lock_timer=0,  # Not used in new mechanics
        ),
        should_lock,
    )


def spawn_new_piece(state: EnvState, key: jax.Array, preview_num: int = 1) -> EnvState:
    """Spawn a new piece and update the piece queue for nonstationarity."""
    # Get next piece from the first position in next_pieces array
    current_piece = state.next_pieces[0]
    
    # Update queue index and get pieces for the preview
    new_queue_index = (state.queue_index + 1) % len(state.piece_queue)
    
    # Check if we need to generate a new shuffled queue
    should_reshuffle = new_queue_index == 0
    
    # Generate new shuffled queue if needed
    new_queue = jax.lax.select(
        should_reshuffle,
        jax.random.permutation(key, jnp.arange(7)),
        state.piece_queue
    )
    
    # Create new next_pieces array by shifting and adding new piece
    # Shift all pieces one position up and add new piece at the end
    def get_next_piece(i):
        # For positions 0 to preview_num-2, take from next position
        shifted_piece = jax.lax.select(
            i < preview_num - 1,
            state.next_pieces[i + 1],
            0  # placeholder
        )
        
        # For the last position, get from queue
        queue_pos = (new_queue_index + i) % len(new_queue)
        new_piece = new_queue[queue_pos]
        
        return jax.lax.select(
            i < preview_num - 1,
            shifted_piece,
            new_piece
        )
    
    # Build new next_pieces array
    new_next_pieces = jnp.array([get_next_piece(i) for i in range(preview_num)])
    
    # Check if new piece causes game over
    initial_piece_shape = TETROMINOES[current_piece, 0]
    game_over = check_collision(state.board, initial_piece_shape, 3, 0)
    
    return state.replace(
        current_piece=current_piece,
        current_x=3,  # Start in middle of board
        current_y=0,  # Start at top
        current_rotation=0,
        next_pieces=new_next_pieces,
        piece_queue=new_queue,
        queue_index=new_queue_index,
        lock_timer=0,
        drop_timer=0,
        terminal=game_over,
        frames_since_spawn=0,  # Reset frame counter when spawning new piece
    )


class Tetris(environment.Environment[EnvState, EnvParams]):
    """JAX implementation of Tetris MinAtar environment.

    ENVIRONMENT DESCRIPTION - 'Tetris-MinAtar'
    - Player controls falling tetromino pieces on a 20x10 board.
    - Goal is to clear horizontal lines by filling them completely.
    - Game ends when pieces stack to the top of the board.
    - Reward is given for clearing lines (1 point per line).
    - Nonstationarity: Random piece order via shuffled queues.
    - Actions: [rotate (up), soft drop (down), left, right, noop]
    - Observation has dimensionality (20, 10, 2) - board and current piece
    """
    
    color = {
        "black": jnp.array([0, 0, 0], dtype=jnp.uint8),
        "white": jnp.array([255, 255, 255], dtype=jnp.uint8),
        "gray": jnp.array([128, 128, 128], dtype=jnp.uint8),
        "red": jnp.array([255, 0, 0], dtype=jnp.uint8),
        "green": jnp.array([0, 255, 0], dtype=jnp.uint8),
        "blue": jnp.array([0, 0, 255], dtype=jnp.uint8),
        "yellow": jnp.array([255, 255, 0], dtype=jnp.uint8),
        "cyan": jnp.array([0, 255, 255], dtype=jnp.uint8),
        "magenta": jnp.array([255, 0, 255], dtype=jnp.uint8),
        "orange": jnp.array([255, 128, 0], dtype=jnp.uint8),
        "bright_red": jnp.array([255, 48, 71], dtype=jnp.uint8),
    }
    
    # Colors for different tetromino types
    piece_colors = jnp.array([
        [0, 203, 225],  # I - cyan
        [225, 206, 0],  # O - yellow
        [187, 0, 222],  # T - magenta
        [0, 226, 64],  # S - green
        [208, 0, 0],  # Z - red
        [0, 119, 220],  # J - blue
        [215, 144, 0],  # L - orange
    ], dtype=jnp.uint8)
    
    size = {
        256: {
            "canvas_size": 256,
            "small_canvas_size": 200,
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
            "small_canvas_size": 100,
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

    def __init__(self, obs_size: int = 128, partial_obs=False, max_steps_in_episode=5000, preview_num=1, auto_drop_speed=1, soft_drop_speed=2):
        super().__init__()
        self.obs_shape = (20, 10, 2)  # board + current piece
        # Action set: [rotate (up), soft drop (down), left, right, noop]
        self.action_set = jnp.array([0, 1, 2, 3, 4])
        self.max_steps_in_episode = max_steps_in_episode
        self.obs_size = obs_size
        self.partial_obs = partial_obs
        self.preview_num = preview_num
        self.auto_drop_speed = auto_drop_speed
        self.soft_drop_speed = soft_drop_speed

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters."""
        return EnvParams(
            max_steps_in_episode=self.max_steps_in_episode,
            auto_drop_speed=self.auto_drop_speed,
            soft_drop_speed=self.soft_drop_speed
        )

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jnp.ndarray, jnp.ndarray, dict[Any, Any]]:
        """Perform single timestep state transition."""
        # Handle spawning new piece and clearing lines from previous frame
        def spawn_and_clear(state_and_key):
            state, key = state_and_key
            new_board, lines_cleared = clear_lines(state.board)
            
            reward = lines_cleared.astype(jnp.int32) * 0.1

            new_lines_cleared = state.lines_cleared + lines_cleared
            state = state.replace(
                board=new_board,
                lines_cleared=new_lines_cleared,
                score=state.score + lines_cleared,
                just_locked=False,
            )
            
            success = new_lines_cleared >= 10
            state = state.replace(terminal=jnp.logical_or(state.terminal, success))
            
            new_state = spawn_new_piece(state, key, self.preview_num)
            return new_state, reward
        
        def no_spawn_clear(state_and_key):
            state, key = state_and_key
            return state, 0.0
        
        state, reward = jax.lax.cond(
            state.just_locked,
            spawn_and_clear,
            no_spawn_clear,
            (state, key)
        )

        state, should_lock = step_piece(state, action, params)
        
        state = state.replace(
            frames_since_spawn=jnp.minimum(state.frames_since_spawn + 1, 10)
        )
        

        def lock_piece(state_and_key):
            state, key = state_and_key
            piece_shape = TETROMINOES[state.current_piece, state.current_rotation]
            new_board = place_piece(state.board, piece_shape, state.current_x, state.current_y, state.current_piece)
            
            state = state.replace(
                board=new_board,
                just_locked=True,
            )
            
            return state
        
        def no_lock(state_and_key):
            state, key = state_and_key
            return state
        
        state = jax.lax.cond(
            should_lock,
            lock_piece,
            no_lock,
            (state, key)
        )
        

        state = state.replace(time=state.time + 1)

        done = self.is_terminal(state, params)
        state = state.replace(terminal=done)
        
        done_steps = state.time >= params.max_steps_in_episode  # Truncation
        done_lines = state.lines_cleared >= 10  # Success
        done_collision = jnp.logical_and(done, jnp.logical_not(jnp.logical_or(done_steps, done_lines)))  # Collision
        
        reward = jax.lax.select(done_collision, reward - 1.0, reward)
        info = {"discount": self.discount(state, params),
                "truncated": done_steps,
                "terminated": state.terminal}
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state."""
        key1, key2 = jax.random.split(key)
        
        piece_queue = jax.random.permutation(key1, jnp.arange(7))
        initial_piece = piece_queue[0]
        
        next_pieces = jnp.array([piece_queue[(i + 1) % 7] for i in range(self.preview_num)])
        
        state = EnvState(
            board=jnp.zeros((20, 10), dtype=jnp.int32),
            current_piece=initial_piece,
            current_x=3,
            current_y=0,
            current_rotation=0,
            next_pieces=next_pieces,
            piece_queue=piece_queue,
            queue_index=self.preview_num,  # Start after the preview pieces
            lines_cleared=0,
            score=0,
            time=0,
            terminal=False,
            drop_timer=0,
            lock_timer=0,
            frames_since_spawn=0,  # Initialize frame counter
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Return observation from raw state transformation."""
        return self.render(state)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        done_steps = state.time >= params.max_steps_in_episode
        done_lines = state.lines_cleared >= 10
        return jnp.logical_or(jnp.logical_or(done_steps, done_lines), state.terminal)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Tetris"

    @functools.partial(jax.jit, static_argnums=(0,))
    def render(self, state: EnvState) -> jax.Array:
        """Render the current state."""
        canvas = jnp.zeros(
            (self.size[self.obs_size]["canvas_size"], self.size[self.obs_size]["canvas_size"], 3), 
            dtype=jnp.uint8
        ) + self.color["gray"]
        
        small_canvas = jnp.full(
            (self.size[self.obs_size]["small_canvas_size"], self.size[self.obs_size]["small_canvas_size"], 3),
            self.color["black"], 
            dtype=jnp.uint8
        )
        
        # Calculate cell size for rendering
        cell_size = self.size[self.obs_size]["small_canvas_size"] // 20
        
        # Render board pieces
        y_coords, x_coords = jnp.meshgrid(
            jnp.arange(self.size[self.obs_size]["small_canvas_size"]), 
            jnp.arange(self.size[self.obs_size]["small_canvas_size"]), 
            indexing='ij'
        )
        
        board_y = y_coords // cell_size
        board_x = (x_coords // cell_size) - 5  # Center the 10-wide board
        
        # Ensure indices are within bounds
        board_y = jnp.clip(board_y, 0, 19)
        board_x = jnp.clip(board_x, 0, 9)
        
        # Create mask for valid board positions
        valid_x = jnp.logical_and(x_coords >= 5 * cell_size, x_coords < 15 * cell_size)
        board_values = state.board[board_y, board_x]
        board_mask = jnp.logical_and(valid_x, board_values > 0)
        
        # POMDP: Hide locked pieces when partial_obs is True
        # Only show board pieces if partial_obs is False
        show_board_pieces = jnp.logical_and(board_mask, jnp.logical_not(self.partial_obs))
        
        # Get piece type (subtract 1 because board stores piece_type + 1)
        piece_types = jnp.clip(board_values - 1, 0, 6)
        board_colors = self.piece_colors[piece_types]
        
        # Check for completed lines
        line_complete = jnp.all(state.board > 0, axis=1)  # Check which lines are complete
        
        # Draw board pieces with their original colors (only if not partial_obs)
        draw_normal_pieces = jnp.logical_and(show_board_pieces, jnp.logical_not(self.partial_obs))
        small_canvas = jnp.where(draw_normal_pieces[:, :, None], board_colors, small_canvas)
        
        # Draw borders for board pieces to show individual squares (only if not partial_obs)
        # Create masks for borders
        cell_y_in_cell = y_coords % cell_size
        cell_x_in_cell = x_coords % cell_size
        
        # Border masks (1 pixel borders) - apply to all visible pieces
        all_visible_pieces = jnp.logical_and(show_board_pieces, jnp.logical_not(self.partial_obs))
        top_border = jnp.logical_and(all_visible_pieces, cell_y_in_cell == 0)
        bottom_border = jnp.logical_and(all_visible_pieces, cell_y_in_cell == cell_size - 1)
        left_border = jnp.logical_and(all_visible_pieces, cell_x_in_cell == 0)
        right_border = jnp.logical_and(all_visible_pieces, cell_x_in_cell == cell_size - 1)
        
        # Apply white borders for normal pieces
        border_mask = jnp.logical_or(
            jnp.logical_or(top_border, bottom_border),
            jnp.logical_or(left_border, right_border)
        )
        small_canvas = jnp.where(border_mask[:, :, None], self.color["white"], small_canvas)
        
        # Draw white squares on walls for completed lines (line clearing indicators)
        # Create mask for completed lines positions on walls
        board_y_in_bounds = jnp.logical_and(board_y >= 0, board_y < 20)
        line_to_indicate = jnp.logical_and(board_y_in_bounds, line_complete[board_y])
        
        # Left wall indicator (at x = 5 * cell_size - 2 to 5 * cell_size - 1)
        # left_wall_indicator_mask = jnp.logical_and(
        #     jnp.logical_and(x_coords >= 5 * cell_size - 2, x_coords < 15 * cell_size + 2),
        #     line_to_indicate
        # )
        # small_canvas = jnp.where(left_wall_indicator_mask[:, :, None], self.color["white"], small_canvas)
        
        # Draw visual walls/boundaries
        # Left wall (at x = 5 * cell_size)
        left_wall_mask = x_coords == 5 * cell_size - 1
        small_canvas = jnp.where(left_wall_mask[:, :, None], self.color["white"], small_canvas)
        
        # Right wall (at x = 15 * cell_size)  
        right_wall_mask = x_coords == 15 * cell_size
        small_canvas = jnp.where(right_wall_mask[:, :, None], self.color["white"], small_canvas)
        
        # Show lines cleared on the left side (left of left wall)
        lines_text_x = 1 * cell_size  # Position in left margin
        lines_text_y = 2 * cell_size  # Near the top

        next_piece_x, next_piece_y, next_label_height_cells = jnp.int32(16 * cell_size), jnp.int32(2 * cell_size), jnp.int32(1)

        def draw_preview_piece(canvas, piece_idx):
            # Calculate vertical offset for each preview piece
            piece_offset_y = next_piece_y + next_label_height_cells * cell_size + 2 + piece_idx * (3 * cell_size)
            
            # Get the piece type for this preview position
            preview_piece_type = state.next_pieces[piece_idx]
            preview_piece_shape = TETROMINOES[preview_piece_type, 0]  # Always show in default rotation
            preview_piece_color = self.piece_colors[preview_piece_type]
            
            # Render preview piece in a 4x4 grid
            preview_py_grid, preview_px_grid = jnp.meshgrid(jnp.arange(4), jnp.arange(4), indexing='ij')
            preview_render_x = next_piece_x + preview_px_grid * (cell_size // 2)  # Smaller cells for preview
            preview_render_y = piece_offset_y + preview_py_grid * (cell_size // 2)
            preview_should_render = preview_piece_shape == 1
            
            # Draw preview piece blocks with borders
            def draw_preview_piece_block(canvas, args):
                py, px = args
                cond = preview_should_render[py, px]
                x1 = preview_render_x[py, px]
                y1 = preview_render_y[py, px]
                x2 = x1 + (cell_size // 2)
                y2 = y1 + (cell_size // 2)
                
                def draw_preview_square_with_border(c):
                    # Draw filled square
                    c = draw_rectangle((x1, y1), (x2, y2), preview_piece_color, c)
                    
                    # Draw border only if preview cells are big enough (>=4 pixels)
                    # to avoid covering the entire colored area
                    preview_cell_size = cell_size // 2
                    should_draw_border = preview_cell_size >= 4
                    
                    def draw_borders(canvas):
                        border_color = self.color["white"]
                        border_size = 1  # Always 1 pixel border
                        # Top border
                        canvas = draw_rectangle((x1, y1), (x2, y1 + border_size), border_color, canvas)
                        # Bottom border  
                        canvas = draw_rectangle((x1, y2 - border_size), (x2, y2), border_color, canvas)
                        # Left border
                        canvas = draw_rectangle((x1, y1), (x1 + border_size, y2), border_color, canvas)
                        # Right border
                        canvas = draw_rectangle((x2 - border_size, y1), (x2, y2), border_color, canvas)
                        return canvas
                    
                    return jax.lax.cond(should_draw_border, draw_borders, lambda c: c, c)
                
                return jax.lax.cond(
                    cond,
                    draw_preview_square_with_border,
                    lambda c: c,
                    canvas
                )
            
            # Apply drawing for all preview piece positions
            canvas = jax.lax.fori_loop(
                0, 16,
                lambda i, canvas: draw_preview_piece_block(canvas, (i // 4, i % 4)),
                canvas
            )
            
            return canvas
        
        # Draw all preview pieces
        small_canvas = jax.lax.fori_loop(
            0, self.preview_num,
            lambda i, canvas: draw_preview_piece(canvas, i),
            small_canvas
        )
        
        # Render current piece
        current_piece_shape = TETROMINOES[state.current_piece, state.current_rotation]
        piece_color = self.piece_colors[state.current_piece]
        
        # Use JAX-compatible operations for rendering current piece
        # Create coordinate grids for the 4x4 piece
        py_grid, px_grid = jnp.meshgrid(jnp.arange(4), jnp.arange(4), indexing='ij')
        
        # Calculate render positions for each piece cell
        render_x = (state.current_x + px_grid + 5) * cell_size  # +5 to center
        render_y = (state.current_y + py_grid) * cell_size
        
        # Check which cells should be rendered (where piece has blocks)
        should_render = current_piece_shape == 1
        
        # For each piece block, draw a square with border
        # We need to iterate through the piece blocks in a JAX-compatible way
        def draw_piece_block(canvas, args):
            py, px = args
            cond = should_render[py, px]
            x1 = render_x[py, px]
            y1 = render_y[py, px]
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            
            def draw_square_with_border(c):
                # Draw filled square with piece color
                c = draw_rectangle((x1, y1), (x2, y2), piece_color, c)
                
                # Draw border around the square (1 pixel border)
                border_color = self.color["white"]
                # Top border
                c = draw_rectangle((x1, y1), (x2, y1 + 1), border_color, c)
                # Bottom border  
                c = draw_rectangle((x1, y2 - 1), (x2, y2), border_color, c)
                # Left border
                c = draw_rectangle((x1, y1), (x1 + 1, y2), border_color, c)
                # Right border
                c = draw_rectangle((x2 - 1, y1), (x2, y2), border_color, c)
                
                return c
            
            return jax.lax.cond(
                cond,
                draw_square_with_border,
                lambda c: c,
                canvas
            )
        
        small_canvas = jax.lax.fori_loop(
            0, 16,
            lambda i, canvas: draw_piece_block(canvas, (i // 4, i % 4)),
            small_canvas
        )

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
        left_wall_indicator_mask = jnp.logical_and(
            jnp.logical_and(x_coords >= 5 * cell_size - 2, x_coords < 15 * cell_size + 2),
            line_to_indicate
        )
        small_canvas = jnp.where(left_wall_indicator_mask[:, :, None], self.color["white"], small_canvas)
        
        canvas = draw_sub_canvas(small_canvas, canvas)
        return canvas

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(self.action_set))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 255, (self.obs_size, self.obs_size, 3), dtype=jnp.uint8)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "board": spaces.Box(0, 1, (20, 10)),
                "current_piece": spaces.Discrete(7),
                "current_x": spaces.Discrete(10),
                "current_y": spaces.Discrete(20),
                "current_rotation": spaces.Discrete(4),
                "next_pieces": spaces.Box(0, 6, (self.preview_num,)),
                "piece_queue": spaces.Box(0, 6, (7,)),
                "queue_index": spaces.Discrete(7),
                "lines_cleared": spaces.Discrete(1000),
                "score": spaces.Discrete(1000),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "terminal": spaces.Discrete(2),
                "drop_timer": spaces.Discrete(params.drop_interval + 1),
                "lock_timer": spaces.Discrete(params.lock_delay + 1),
                "frames_since_spawn": spaces.Discrete(11),  # 0-10 frames
            }
        )


class TetrisEasy(Tetris):
    def __init__(self, **kwargs):
        super().__init__(
            max_steps_in_episode=3000, 
            preview_num=3, 
            auto_drop_speed=1,
            soft_drop_speed=2,
            **kwargs
        )


class TetrisMedium(Tetris):
    def __init__(self, **kwargs):
        super().__init__(
            max_steps_in_episode=3000, 
            preview_num=2, 
            auto_drop_speed=1,
            soft_drop_speed=2,
            **kwargs
        )


class TetrisHard(Tetris):
    def __init__(self, **kwargs):
        super().__init__(
            max_steps_in_episode=3000, 
            preview_num=1, 
            auto_drop_speed=2,
            soft_drop_speed=3,
            **kwargs
        )