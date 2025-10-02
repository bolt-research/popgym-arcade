from functools import partial
from typing import Tuple

import chex
import dm_pix as dm
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array


def log_normal(value: chex.Array, minimum: int) -> chex.Array:
    return minimum + 2 * jnp.log(value + 1)


def draw_rectangle(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    color: chex.Array,
    canvas: chex.Array,
) -> chex.Array:
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right

    x_start = jnp.clip(top_x, 0, canvas.shape[0])
    y_start = jnp.clip(top_y, 0, canvas.shape[1])
    x_end = jnp.clip(bottom_x, 0, canvas.shape[0])
    y_end = jnp.clip(bottom_y, 0, canvas.shape[1])

    mask_x = jnp.logical_and(
        jnp.arange(canvas.shape[0]) >= x_start, jnp.arange(canvas.shape[0]) < x_end
    )
    mask_y = jnp.logical_and(
        jnp.arange(canvas.shape[1]) >= y_start, jnp.arange(canvas.shape[1]) < y_end
    )

    mask = jnp.outer(mask_y, mask_x)

    colored_canvas = jnp.where(mask[:, :, None], color, canvas)

    return colored_canvas


def draw_circle(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    radius: float,
    color: chex.Array,
    canvas: chex.Array,
) -> chex.Array:
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right
    center_x = (top_x + bottom_x) // 2
    center_y = (top_y + bottom_y) // 2
    y, x = jnp.ogrid[: canvas.shape[1], : canvas.shape[0]]
    dist_from_center = jnp.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    mask = dist_from_center <= radius
    colored_canvas = jnp.where(mask[:, :, None], color, canvas)

    return colored_canvas


def draw_triangle(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    color: chex.Array,
    canvas: chex.Array,
    direction: int,
) -> chex.Array:
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right

    def upward(top_x, top_y, bottom_x, bottom_y) -> Tuple:
        return ((top_x + bottom_x) // 2, top_y, bottom_x, bottom_y, top_x, bottom_y)

    def downward(top_x, top_y, bottom_x, bottom_y) -> Tuple:
        return (top_x, top_y, bottom_x, top_y, (top_x + bottom_x) // 2, bottom_y)

    def leftward(top_x, top_y, bottom_x, bottom_y) -> Tuple:
        return (top_x, (top_y + bottom_y) // 2, bottom_x, top_y, bottom_x, bottom_y)

    def rightward(top_x, top_y, bottom_x, bottom_y) -> Tuple:
        return (top_x, top_y, bottom_x, (top_y + bottom_y) // 2, top_x, bottom_y)

    x1, y1, x2, y2, x3, y3 = lax.cond(
        direction == 1,
        lambda _: upward(top_x, top_y, bottom_x, bottom_y),
        lambda _: lax.cond(
            direction == 2,
            lambda _: downward(top_x, top_y, bottom_x, bottom_y),
            lambda _: lax.cond(
                direction == 3,
                lambda _: leftward(top_x, top_y, bottom_x, bottom_y),
                lambda _: rightward(top_x, top_y, bottom_x, bottom_y),
                operand=None,
            ),
            operand=None,
        ),
        operand=None,
    )

    def edge_function(x, y, x1, y1, x2, y2):
        return (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)

    x, y = jnp.meshgrid(jnp.arange(canvas.shape[0]), jnp.arange(canvas.shape[1]))

    edge1 = edge_function(x, y, x1, y1, x2, y2) >= 0
    edge2 = edge_function(x, y, x2, y2, x3, y3) >= 0
    edge3 = edge_function(x, y, x3, y3, x1, y1) >= 0

    mask = edge1 & edge2 & edge3
    colored_canvas = jnp.where(mask[:, :, None], color, canvas)

    return colored_canvas


def draw_o(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    thickness: int,
    color: chex.Array,
    canvas: chex.Array,
) -> chex.Array:
    """Draws a hollow circle (O) using vectorized operations."""
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right

    cx = (top_x + bottom_x) // 2
    cy = (top_y + bottom_y) // 2
    outer_radius = (bottom_x - top_x) // 2
    inner_radius = outer_radius - thickness

    y, x = jnp.ogrid[: canvas.shape[0], : canvas.shape[1]]

    dx = x - cx
    dy = y - cy
    squared_dist = dx**2 + dy**2

    annulus_mask = (squared_dist <= outer_radius**2) & (squared_dist >= inner_radius**2)

    return annulus_mask[..., None] * color + (~annulus_mask[..., None]) * canvas


def draw_x(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    thickness: int,
    color: chex.Array,
    canvas: chex.Array,
) -> chex.Array:
    """Draws an 'X' inside a rectangle using vectorized operations."""
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right
    height, width, _ = canvas.shape
    y, x = jnp.ogrid[:height, :width]

    dx = bottom_x - top_x
    dy = bottom_y - top_y
    line_length = jnp.sqrt(dx**2 + dy**2)
    norm_thickness = thickness * line_length / 2  # Normalized thickness

    dist_line1 = jnp.abs((y - top_y) * dx - (x - top_x) * dy)
    dist_line2 = jnp.abs((y - bottom_y) * dx - (x - top_x) * (-dy))

    in_rect = (x >= top_x) & (x <= bottom_x) & (y >= top_y) & (y <= bottom_y)

    mask = ((dist_line1 <= norm_thickness) | (dist_line2 <= norm_thickness)) & in_rect

    return mask[..., None] * color + (~mask[..., None]) * canvas


def draw_grid(
    square_size: int, thickness: int, color: chex.Array, canvas: chex.Array
) -> chex.Array:
    height, width, _ = canvas.shape

    # Generate grid coordinates
    y, x = jnp.ogrid[:height, :width]

    # Calculate the period of the grid
    period = square_size + thickness

    # Create masks for vertical and horizontal lines
    vertical_mask = x % period < thickness
    horizontal_mask = y % period < thickness

    # Combine masks to create the grid mask
    grid_mask = vertical_mask | horizontal_mask

    # Use the mask as a weight to blend the grid color and the canvas color
    grid_canvas = grid_mask[:, :, None] * color + (1 - grid_mask[:, :, None]) * canvas

    return grid_canvas


def draw_sub_canvas(sub_canvas: chex.Array, canvas: chex.Array) -> chex.Array:
    """
    Draws a sub-canvas on a blank (256x256x3) canvas.
    Examples:
    - sub_canvas = jnp.ones((192, 192, 3))
    - large_canvas = jnp.zeros((256, 256, 3))
    """
    sub_canvas_width = sub_canvas.shape[0]
    sub_canvas_height = sub_canvas.shape[1]
    canvas_width = canvas.shape[0]
    canvas_height = canvas.shape[1]
    margin_width = (canvas_width - sub_canvas_width) // 2
    margin_height = (canvas_height - sub_canvas_height) // 2
    # print(margin_width, margin_height)
    merged_canvas = canvas.at[
        margin_width : canvas_width - margin_width,
        margin_height : canvas_height - margin_height,
        :,
    ].set(sub_canvas)
    return merged_canvas


def draw_heart(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    color: chex.Array,
    canvas: chex.Array,
) -> chex.Array:
    """
    Draws a heart shape defined by top_left and bottom_right on the canvas.
    The heart is oriented upward like a standard playing card symbol ♥.
    """
    if canvas.shape[0] == 256 or canvas.shape[0] == 192:
        adjust = 6
    elif canvas.shape[0] == 128 or canvas.shape[0] == 96:
        adjust = 4
    else:
        pass
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right

    # Calculate width and height
    width = bottom_x - top_x
    height = bottom_y - top_y
    size = jnp.minimum(width, height)

    # Calculate center and scale
    center_x = (top_x + bottom_x) // 2
    center_y = (top_y + bottom_y) // 2
    scale = size // 2 + adjust

    # canvas = jnp.zeros_like(canva)
    y, x = jnp.ogrid[: canvas.shape[0], : canvas.shape[1]]

    # Normalize coordinates to [-2, 2] range within the specified rectangle
    x = (x - center_x) / scale * 2
    y = -((y - center_y) / scale * 2)  # Note the negative sign for upward orientation

    # Heart shape equation (modified for upward orientation)
    heart_mask = ((x**2 + y**2 - 1) ** 3 - (x**2 * y**3)) < 0

    # Apply the mask only within the specified rectangle
    within_rect = (
        (x >= (top_x - center_x) / scale * 2)
        & (x <= (bottom_x - center_x) / scale * 2)
        & (y >= (top_y - center_y) / scale * 2)
        & (y <= (bottom_y - center_y) / scale * 2)
    )

    heart_mask = heart_mask & within_rect

    colored_canvas = jnp.where(heart_mask[:, :, None], color, canvas)

    return colored_canvas


def draw_spade(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    color: chex.Array,
    canvas: chex.Array,
) -> chex.Array:
    """
    Draws a spade shape (♠) defined by top_left and bottom_right on the canvas.
    The spade consists of a heart shape rotated 180 degrees and a triangle at the bottom.
    """
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right

    # Calculate width and height
    width = bottom_x - top_x
    height = bottom_y - top_y

    # Calculate center and scale
    center_x = (top_x + bottom_x) / 2

    y, x = jnp.ogrid[: canvas.shape[0], : canvas.shape[1]]
    # Split the height: 2/3 for heart, 1/3 for triangle
    heart_height = height * 0.7

    # Heart part
    # Normalize coordinates for heart shape
    x_heart = (x - (top_x + width / 2)) / (width / 2) * 1.404  # Scale x from -1 to 1
    y_heart = (
        -(y - (top_y + heart_height / 2)) / (heart_height / 2) * 1.404
    )  # Scale y and invert

    # Heart equation
    inverted_heart_mask = (
        (x_heart**2 + y_heart**2 - 1) ** 3 - (x_heart**2 * (-y_heart) ** 3)
    ) < 0
    # Triangle mask
    x1, y1 = (
        center_x,
        bottom_y - height * 0.8,
    )  # Top vertex at the center bottom of the heart
    x2, y2 = center_x + width * 0.4, bottom_y + height * 1.8
    x3, y3 = center_x - width * 0.4, bottom_y + height * 1.8

    def edge_function(x, y, x1, y1, x2, y2):
        return (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)

    edge1 = edge_function(x, y, x1, y1, x2, y2) >= 0
    edge2 = edge_function(x, y, x2, y2, x3, y3) >= 0
    edge3 = edge_function(x, y, x3, y3, x1, y1) >= 0

    triangle_mask = edge1 & edge2 & edge3
    # Combine masks for spade shape
    boundary_mask = (x >= top_x) & (x <= bottom_x) & (y >= top_y) & (y <= bottom_y)
    spade_mask = (inverted_heart_mask | triangle_mask) & boundary_mask

    # Color the canvas
    colored_canvas = jnp.where(spade_mask[:, :, None], color, canvas)

    return colored_canvas


def draw_club(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    color: chex.Array,
    canvas: chex.Array,
) -> chex.Array:
    """
    Draws a club shape (♣) within the specified boundary.
    The club consists of three circles and a stem with triangle.
    """
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right

    # Calculate dimensions
    width = bottom_x - top_x
    height = bottom_y - top_y

    y, x = jnp.ogrid[: canvas.shape[0], : canvas.shape[1]]
    # Calculate center and radius
    center_x = (top_x + bottom_x) // 2
    center_y = (top_y + bottom_y) // 2
    radius = jnp.minimum(width, height) // 3.12

    # Create the three circles
    # Top circle
    circle1_x = center_x
    circle1_y = center_y - radius * 0.8
    circle1 = ((x - circle1_x) ** 2 + (y - circle1_y) ** 2) <= radius**2

    # Bottom left circle
    circle2_x = center_x - radius * 0.8
    circle2_y = center_y + radius
    circle2 = ((x - circle2_x) ** 2 + (y - circle2_y) ** 2) <= radius**2

    # Bottom right circle
    circle3_x = center_x + radius * 0.8
    circle3_y = center_y + radius
    circle3 = ((x - circle3_x) ** 2 + (y - circle3_y) ** 2) <= radius**2

    # Create stem
    stem_width = radius * 0.8
    stem_height = height * 0.4
    stem_top_y = center_y + radius
    stem = (
        (jnp.abs(x - center_x) <= stem_width // 2) & (y >= stem_top_y) & (y <= bottom_y)
    )
    # Create triangle at the bottom
    triangle_top_y = bottom_y - stem_height * 0.5

    def edge_function(px, py, x1, y1, x2, y2):
        return (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)

    # Triangle vertices
    x1, y1 = center_x, triangle_top_y  # Top point
    x2, y2 = center_x - stem_width, bottom_y - 6  # Bottom left
    x3, y3 = center_x + stem_width, bottom_y - 6  # Bottom right

    # Calculate triangle edges
    edge1 = edge_function(x, y, x1, y1, x2, y2) >= 0
    edge2 = edge_function(x, y, x2, y2, x3, y3) >= 0
    edge3 = edge_function(x, y, x3, y3, x1, y1) >= 0

    triangle = edge1 & edge2 & edge3

    # Boundary mask
    boundary_mask = (x >= top_x) & (x <= bottom_x) & (y >= top_y) & (y <= bottom_y)
    # Combine all shapes
    club_mask = (circle1 | circle2 | circle3 | stem | triangle) & boundary_mask
    # Color the canvas
    colored_canvas = jnp.where(club_mask[:, :, None], color, canvas)

    return colored_canvas


def draw_diamond(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    color: chex.Array,
    canvas: chex.Array,
) -> chex.Array:
    """
    Draws a diamond shape (♦) within the specified boundary.
    The diamond is created using coordinate transformations and masks.
    """
    if canvas.shape[0] == 256 or canvas.shape[0] == 192:
        adjust = 1.48
    elif canvas.shape[0] == 128 or canvas.shape[0] == 96:
        adjust = 1.2
    else:
        pass
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right

    # Calculate dimensions
    width = bottom_x - top_x
    height = bottom_y - top_y

    y, x = jnp.ogrid[: canvas.shape[0], : canvas.shape[1]]
    # Calculate center
    center_x = (top_x + bottom_x) // 2
    center_y = (top_y + bottom_y) // 2

    # Scale factors for x and y directions
    scale_x = width // 2
    scale_y = height // 2

    # Normalize coordinates to [-1, 1] range
    x_norm = (x - center_x) / scale_x * adjust
    y_norm = (y - center_y) / scale_y * adjust

    # Create diamond mask using normalized coordinates
    diamond_mask = (jnp.abs(x_norm) + jnp.abs(y_norm)) <= 1.0

    # Apply boundary mask
    boundary_mask = (x >= top_x) & (x <= bottom_x) & (y >= top_y) & (y <= bottom_y)

    final_mask = diamond_mask & boundary_mask

    # Color the canvas
    colored_canvas = jnp.where(final_mask[:, :, None], color, canvas)

    return colored_canvas


def draw_hexagon(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    color: chex.Array,
    canvas: chex.Array,
) -> chex.Array:
    """
    Draws a vertical hexagon shape (⬡) within the specified boundary.
    """
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right

    # Calculate center and size
    width = bottom_x - top_x
    height = bottom_y - top_y
    size = jax.lax.min(width, height) / 2
    center_x = (top_x + bottom_x) / 2
    center_y = (top_y + bottom_y) / 2

    # Create grid
    y_indices = jnp.arange(canvas.shape[0])
    x_indices = jnp.arange(canvas.shape[1])
    yy, xx = jnp.meshgrid(y_indices, x_indices, indexing="ij")

    # Normalize coordinates and rotate
    x = (xx - center_x) / size
    y = (yy - center_y) / size
    x_rot = x * jnp.cos(jnp.pi / 6) - y * jnp.sin(jnp.pi / 6)
    y_rot = x * jnp.sin(jnp.pi / 6) + y * jnp.cos(jnp.pi / 6)

    # Hexagon mask
    sqrt3 = jnp.sqrt(3)

    hex_mask = (
        (jnp.abs(x_rot) <= 1)
        & (jnp.abs(y_rot) <= sqrt3 / 2)
        & (sqrt3 * jnp.abs(x_rot) + jnp.abs(y_rot) <= sqrt3)
    )

    # Apply mask
    colored_canvas = jnp.where(hex_mask[:, :, None], color, canvas)

    return colored_canvas


def draw_matchstick_man(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    color: chex.Array,
    canvas: chex.Array,
) -> chex.Array:
    """
    Draws a simple matchstick man figure within the specified boundary.

    Args:
        top_left: Top-left coordinates of the drawing boundary
        bottom_right: Bottom-right coordinates of the drawing boundary
        color: Color array for drawing the matchstick man
        canvas: Input canvas to draw on

    Returns:
        Updated canvas with matchstick man drawn
    """

    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right

    # Calculate dimensions and center
    width = bottom_x - top_x
    height = bottom_y - top_y
    size = jax.lax.min(width, height)
    center_x = (top_x + bottom_x) / 2
    center_y = (top_y + bottom_y) / 2

    # Create grid
    y_indices = jnp.arange(canvas.shape[0])
    x_indices = jnp.arange(canvas.shape[1])
    yy, xx = jnp.meshgrid(y_indices, x_indices, indexing="ij")

    # Head (circular)
    head_radius = size * 0.15
    head_mask = (
        (xx - center_x) ** 2 + (yy - (center_y - size * 0.3)) ** 2
    ) <= head_radius**2
    # Body (vertical line)
    body_width = size * 0.1
    body_mask = (
        (jnp.abs(xx - center_x) <= body_width / 2)
        & (yy >= center_y - size * 0.3 + head_radius)
        & (yy <= center_y + size * 0.2)
    )
    # Arms (horizontal lines)
    arm_length = size * 0.2
    arm_width = size * 0.05

    left_arm_mask = (
        (jnp.abs(yy - (center_y - size * 0.1)) <= arm_width / 2)
        & (xx >= center_x - arm_length)
        & (xx <= center_x)
    )
    right_arm_mask = (
        (jnp.abs(yy - (center_y - size * 0.1)) <= arm_width / 2)
        & (xx >= center_x)
        & (xx <= center_x + arm_length)
    )

    # Legs (two angled lines)
    leg_length = size * 0.25
    leg_width = size * 0.1
    leg_angle = jnp.pi / 4
    left_leg_mask = (
        (
            jnp.abs(yy - (center_y + size * 0.2 + (center_x - xx) * jnp.tan(leg_angle)))
            <= leg_width / 2
        )
        & (xx >= center_x - leg_length)
        & (xx <= center_x)
    )

    right_leg_mask = (
        (
            jnp.abs(yy - (center_y + size * 0.2 + (xx - center_x) * jnp.tan(leg_angle)))
            <= leg_width / 2
        )
        & (xx >= center_x)
        & (xx <= center_x + leg_length)
    )

    # Combine masks
    matchstick_mask = (
        head_mask
        | body_mask
        | left_arm_mask
        | right_arm_mask
        | left_leg_mask
        | right_leg_mask
    )

    # Apply mask
    colored_canvas = jnp.where(matchstick_mask[:, :, None], color, canvas)

    return colored_canvas


def draw_tnt_block(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    canvas: chex.Array,
) -> chex.Array:
    """
    Draws a Minecraft-style TNT block on a canvas.

    Args:
        top_left: Top-left coordinates of the TNT block
        bottom_right: Bottom-right coordinates of the TNT block
        canvas: Input canvas to draw on

    Returns:
        Updated canvas with TNT block drawn
    """
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right

    # Clip coordinates to canvas boundaries
    x_start = jnp.clip(top_x, 0, canvas.shape[0])
    y_start = jnp.clip(top_y, 0, canvas.shape[1])
    x_end = jnp.clip(bottom_x, 0, canvas.shape[0])
    y_end = jnp.clip(bottom_y, 0, canvas.shape[1])

    # Create masks for different parts of the TNT block
    mask_x = jnp.logical_and(
        jnp.arange(canvas.shape[0]) >= x_start, jnp.arange(canvas.shape[0]) < x_end
    )
    mask_y = jnp.logical_and(
        jnp.arange(canvas.shape[1]) >= y_start, jnp.arange(canvas.shape[1]) < y_end
    )
    # Full block mask
    full_block_mask = jnp.outer(mask_y, mask_x)

    # Inner white rectangle (slightly smaller)

    inner_x_start = jnp.clip(top_x + (bottom_x - top_x) // 16, 0, canvas.shape[0])
    inner_y_start = jnp.clip(top_y + (bottom_y - top_y) // 3, 0, canvas.shape[1])
    inner_x_end = jnp.clip(bottom_x - (bottom_x - top_x) // 16, 0, canvas.shape[0])
    inner_y_end = jnp.clip(bottom_y - (bottom_y - top_y) // 3, 0, canvas.shape[1])

    inner_mask_x = jnp.logical_and(
        jnp.arange(canvas.shape[0]) >= inner_x_start,
        jnp.arange(canvas.shape[0]) < inner_x_end,
    )
    inner_mask_y = jnp.logical_and(
        jnp.arange(canvas.shape[1]) >= inner_y_start,
        jnp.arange(canvas.shape[1]) < inner_y_end,
    )
    inner_block_mask = jnp.outer(inner_mask_y, inner_mask_x)

    red_color = jnp.array([255, 0, 0], dtype=jnp.uint8)
    white_color = jnp.array([255, 255, 255], dtype=jnp.uint8)
    black_color = jnp.array([0, 0, 0], dtype=jnp.uint8)

    # Create a preliminary canvas with the red background
    red_canvas = jnp.where(full_block_mask[:, :, None], red_color, canvas)

    # Add white inner rectangle
    white_canvas = jnp.where(inner_block_mask[:, :, None], white_color, red_canvas)

    # Function to draw 'TNT' text
    def draw_text(canvas):
        # Determine text placement and size based on block dimensions
        width = bottom_x - top_x
        height = bottom_y - top_y

        # Create text mask for each letter
        y_indices = jnp.arange(canvas.shape[0])
        x_indices = jnp.arange(canvas.shape[1])
        yy, xx = jnp.meshgrid(y_indices, x_indices, indexing="ij")

        # T mask (left part)
        t_mask_x = (xx >= top_x + width * 0.13) & (xx < top_x + width * 0.2)
        t_mask_y = (yy >= top_y + height * 0.35) & (yy < top_y + height * 0.65)
        t_horizontal_mask_x = (xx >= top_x + width * 0.05) & (xx < top_x + width * 0.3)
        t_horizontal_mask_y = (yy >= top_y + height * 0.32) & (
            yy < top_y + height * 0.4
        )
        t_mask = (t_mask_x & t_mask_y) | (t_horizontal_mask_x & t_horizontal_mask_y)

        # N mask (middle part)
        n_mask_x1 = (xx >= top_x + width * 0.325) & (xx < top_x + width * 0.4)
        n_mask_x2 = (xx >= top_x + width * 0.6) & (xx < top_x + width * 0.65)
        n_mask_y = (yy >= top_y + height * 0.34) & (yy < top_y + height * 0.65)

        n_diag_mask = (
            jnp.abs(
                yy
                - (top_y + height * 0.35)
                - (xx - (top_x + width * 0.35))
                * (
                    (top_y + height * 0.65 - (top_y + height * 0.3))
                    / ((top_x + width * 0.65) - (top_x + width * 0.35))
                )
            )
            <= height * 0.05
        )

        n_diag_mask = (
            n_diag_mask
            & (xx >= top_x + width * 0.35)
            & (xx < top_x + width * 0.6)
            & (yy >= top_y + height * 0.35)
            & (yy < top_y + height * 0.65)
        )

        n_mask = ((n_mask_x1 | n_mask_x2) & n_mask_y) | n_diag_mask

        # T mask (right part)

        t2_mask_x = (xx >= top_x + width * 0.76) & (xx < top_x + width * 0.83)
        t2_mask_y = (yy >= top_y + height * 0.35) & (yy < top_y + height * 0.65)
        t2_horizontal_mask_x = (xx >= top_x + width * 0.67) & (
            xx < top_x + width * 0.93
        )
        t2_horizontal_mask_y = (yy >= top_y + height * 0.32) & (
            yy < top_y + height * 0.4
        )
        t2_mask = (t2_mask_x & t2_mask_y) | (
            t2_horizontal_mask_x & t2_horizontal_mask_y
        )

        # Combine text masks
        text_mask = t_mask | n_mask | t2_mask

        # Apply text to canvas
        return jnp.where(text_mask[:, :, None], black_color, canvas)

    # Final canvas with text
    final_canvas = draw_text(white_canvas)

    return final_canvas


def draw_crooked_tail(
    top_left: Tuple,
    bottom_right: Tuple,
    color: chex.Array,
    thickness: int,
    canvas: chex.Array,
) -> chex.Array:
    """Draws a quarter circle ring on a blank (256x256x3) canvas."""
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right
    center_x = (top_x + bottom_x) // 2
    center_y = (top_y + bottom_y) // 2
    y, x = jnp.ogrid[: canvas.shape[0], : canvas.shape[1]]
    dist_from_center = jnp.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    radius = (bottom_x - top_x) // 2

    # Create masks for the outer and inner circles
    outer_mask = dist_from_center <= radius
    inner_mask = dist_from_center <= (radius - thickness)

    # Combine masks to get the ring
    ring_mask = outer_mask & ~inner_mask

    # Apply the mask to the canvas
    crooked_tail = jnp.where(ring_mask[:, :, None], color, canvas)
    quarter_mask = (x >= center_x) & (y <= center_y)

    #  Mask out the three-quarters of the circle to get a quarter circle
    crooked_tail = jnp.where(quarter_mask[:, :, None], crooked_tail, canvas)

    return crooked_tail


def draw_stick(
    top_left: Tuple,
    bottom_right: Tuple,
    angle: float,
    thickness: int,
    color: chex.Array,
    canvas: chex.Array,
) -> chex.Array:
    """Draws a vertical stick on a blank (256x256x3) canvas."""
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right
    bottom_stick_x = (bottom_x + top_x) // 2
    bottom_stick_y = bottom_y

    length = bottom_x - top_x

    # Calculate the top point of the stick
    top_stick_x = bottom_stick_x + length * jnp.sin(angle * jnp.pi)
    top_stick_y = bottom_stick_y - length * jnp.cos(angle * jnp.pi)

    y, x = jnp.ogrid[: canvas.shape[0], : canvas.shape[1]]

    # Calculate the distance from each point to the line
    dist_to_line = jnp.abs(
        (top_stick_y - bottom_stick_y) * x
        - (top_stick_x - bottom_stick_x) * y
        + top_stick_x * bottom_stick_y
        - top_stick_y * bottom_stick_x
    ) / jnp.sqrt(
        (top_stick_y - bottom_stick_y) ** 2 + (top_stick_x - bottom_stick_x) ** 2
    )

    # Create a mask for the stick
    mask = (
        (dist_to_line <= thickness / 2)
        & (x >= jnp.minimum(bottom_x, top_x))
        & (x <= jnp.maximum(bottom_x, top_x))
        & (y >= jnp.minimum(bottom_y, top_y))
        & (y <= jnp.maximum(bottom_y, top_y))
    )

    # Apply the color to the canvas
    colored_canvas = jnp.where(mask[:, :, None], color, canvas)

    return colored_canvas


def return_digit_patterns(index: int) -> chex.Array:
    """According to the number index, return the corresponding digit boolean array."""
    digit_patterns = jnp.array(
        [
            # 0
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            # 1
            [
                [False, False, True, False, False],
                [False, True, True, False, False],
                [False, False, True, False, False],
                [False, False, True, False, False],
                [False, True, True, True, False],
            ],
            # 2
            [
                [True, True, True, True, True],
                [False, False, False, False, True],
                [True, True, True, True, True],
                [True, False, False, False, False],
                [True, True, True, True, True],
            ],
            # 3
            [
                [True, True, True, True, True],
                [False, False, False, False, True],
                [True, True, True, True, True],
                [False, False, False, False, True],
                [True, True, True, True, True],
            ],
            # 4
            [
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
                [False, False, False, False, True],
                [False, False, False, False, True],
            ],
            # 5
            [
                [True, True, True, True, True],
                [True, False, False, False, False],
                [True, True, True, True, True],
                [False, False, False, False, True],
                [True, True, True, True, True],
            ],
            # 6
            [
                [True, True, True, True, True],
                [True, False, False, False, False],
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            # 7
            [
                [True, True, True, True, True],
                [False, False, False, False, True],
                [False, False, False, False, True],
                [False, False, False, False, True],
                [False, False, False, False, True],
            ],
            # 8
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            # 9
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
                [False, False, False, False, True],
                [True, True, True, True, True],
            ],
        ]
    )
    return digit_patterns[index]


def draw_digit(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    color: chex.Array,
    canvas: chex.Array,
    digit: int,
) -> chex.Array:
    """
    Draws a specified digit defined by top_left and bottom_right on the canvas.
    The digit is represented by a 5x5 grid of rectangles.
    """
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right

    # Calculate width and height of each rectangle
    width = (bottom_x - top_x) // 5
    height = (bottom_y - top_y) // 5

    # Get the digit pattern
    digit_pattern = return_digit_patterns(digit)  # Shape (5, 5)

    # Create a grid of coordinates
    y_indices = jnp.arange(canvas.shape[0])
    x_indices = jnp.arange(canvas.shape[1])
    yy, xx = jnp.meshgrid(y_indices, x_indices, indexing="ij")

    # Compute the cell indices for each pixel
    cell_x = ((xx - top_x) // width).astype(jnp.uint8)
    cell_y = ((yy - top_y) // height).astype(jnp.uint8)

    # Create a mask for valid cells
    valid_cells = (cell_x >= 0) & (cell_x < 5) & (cell_y >= 0) & (cell_y < 5)

    # Create a mask for where to draw
    mask = valid_cells & digit_pattern[cell_y, cell_x]

    # Apply the mask to the canvas
    canvas = jnp.where(mask[:, :, None], color, canvas)

    return canvas


def draw_number(
    top_left: Tuple[int | Array, int],
    bottom_right: Tuple[int | Array, int | Array],
    color: chex.Array,
    canvas: chex.Array,
    number: int,
) -> chex.Array:
    """
    Draws a multi-digit number on the canvas.
    Each digit is drawn with some horizontal offset.

    We need to define a couple of default args:
    - digit_width: the pixel width of very digit.
    - margin: the top and bottom margin.
    - space: the space between different digits.
    """
    if canvas.shape[0] == 256:
        digit_width = 15
        margin = 1
        space = 2
    else:
        digit_width = 10
        margin = 1
        space = 2

    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right

    num_digits = jnp.where(
        number == 0,
        1,
        jnp.floor(jnp.log10(jnp.maximum(1, number)) + 1e-7).astype(int) + 1,
    )

    total_width = num_digits * digit_width + (num_digits - 1) * space
    start_col = top_x + (bottom_x - top_x - total_width) // 2

    divisor = 10 ** (num_digits - 1 + (number == 0))  # Handle 0 case

    def body_fun(i, carry):
        canvas, current_divisor = carry

        digit = (number // current_divisor) % 10
        digit = jnp.where(number == 0, 0, digit)

        pos_x = start_col + i * (digit_width + space)
        digit_tl = (pos_x, top_y + margin)
        digit_br = (pos_x + digit_width, bottom_y - margin)

        return (
            draw_digit(digit_tl, digit_br, color, canvas, digit),
            current_divisor // 10,
        )

    initial_carry = (canvas, divisor)

    final_canvas, _ = lax.fori_loop(0, num_digits, body_fun, initial_carry)
    return final_canvas


def draw_single_digit(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    color: chex.Array,
    canvas: chex.Array,
    digit: int,
    digit_width: int = 15,
    margin: int = 1,
) -> chex.Array:
    """
    Draws a specified digit defined by top_left and bottom_right on the canvas.
    The digit is represented by a 5x5 grid of rectangles, with padding and alignment.
    """
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right

    # Calculate the actual drawing area with padding
    available_width = bottom_x - top_x
    available_height = bottom_y - top_y

    text_width = digit_width
    text_height = available_height  # Assume height is given

    # Center the digit horizontally and apply vertical margin
    text_x = top_x + (available_width - text_width) // 2
    text_y = top_y + margin

    # Calculate the actual top-left and bottom-right for the digit
    real_top_left = (text_x, text_y)
    real_bottom_right = (text_x + text_width, text_y + available_height - 2 * margin)

    # Calculate width and height of each rectangle
    width = (text_width) // 5
    height = (available_height - 2 * margin) // 5

    # Get the digit pattern
    digit_pattern = return_digit_patterns(digit)  # Shape (5, 5)

    # Create a grid of coordinates
    y_indices = jnp.arange(canvas.shape[0])
    x_indices = jnp.arange(canvas.shape[1])
    yy, xx = jnp.meshgrid(y_indices, x_indices, indexing="ij")

    # Compute the cell indices for each pixel
    cell_x = ((xx - real_top_left[0]) // width).astype(int)
    cell_y = ((yy - real_top_left[1]) // height).astype(int)

    # Create a mask for valid cells
    valid_cells = (cell_x >= 0) & (cell_x < 5) & (cell_y >= 0) & (cell_y < 5)

    # Create a mask for where to draw
    mask = valid_cells & digit_pattern[cell_y, cell_x]

    # Apply the mask to the canvas
    canvas = jnp.where(mask[:, :, None], color, canvas)

    return canvas


def draw_horizontal_tail(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    thickness: int,
    color: chex.Array,
    canvas: chex.Array,
) -> chex.Array:
    """Draws a horizontal line on a canvas with specified thickness."""
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right
    start_x = top_x
    start_y = (top_y + bottom_y) // 2
    end_x = bottom_x
    end_y = (top_y + bottom_y) // 2

    x_start = jnp.clip(start_x, 0, canvas.shape[0])
    x_end = jnp.clip(end_x, 0, canvas.shape[0])
    y_start = jnp.clip(start_y - thickness // 2, 0, canvas.shape[1])
    y_end = jnp.clip(start_y + thickness // 2, 0, canvas.shape[1])

    mask_x = jnp.logical_and(
        jnp.arange(canvas.shape[0]) >= x_start, jnp.arange(canvas.shape[0]) < x_end
    )

    mask_y = jnp.logical_and(
        jnp.arange(canvas.shape[1]) >= y_start, jnp.arange(canvas.shape[1]) < y_end
    )
    mask = jnp.outer(mask_y, mask_x)

    colored_canvas = jnp.where(mask[:, :, None], color, canvas)

    return colored_canvas


def draw_vertical_tail(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    thickness: int,
    color: chex.Array,
    canvas: chex.Array,
) -> chex.Array:
    """Draws a vertical line on a canvas with specified thickness."""
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right
    start_x = (top_x + bottom_x) // 2
    start_y = top_y
    end_x = (top_x + bottom_x) // 2
    end_y = bottom_y

    x_start = jnp.clip(start_x - thickness // 2, 0, canvas.shape[0])
    x_end = jnp.clip(start_x + thickness // 2, 0, canvas.shape[0])
    y_start = jnp.clip(start_y, 0, canvas.shape[1])
    y_end = jnp.clip(end_y, 0, canvas.shape[1])
    mask_x = jnp.logical_and(
        jnp.arange(canvas.shape[0]) >= x_start, jnp.arange(canvas.shape[0]) < x_end
    )
    mask_y = jnp.logical_and(
        jnp.arange(canvas.shape[1]) >= y_start, jnp.arange(canvas.shape[1]) < y_end
    )

    mask = jnp.outer(mask_y, mask_x)

    colored_canvas = jnp.where(mask[:, :, None], color, canvas)

    return colored_canvas


def draw_horizontal_arrow(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    color: chex.Array,
    velocity: chex.Array,
    canvas: chex.Array,
) -> chex.Array:
    """Draws a straight arrow on a blank (256x256x3) canvas."""

    if canvas.shape[0] == 192:
        mini_th = 2
        mini_hmargin = 8
    else:
        mini_th = 2
        mini_hmargin = 6

    thickness = log_normal(jnp.abs(velocity), mini_th).astype(int)

    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right
    mid_y = (top_y + bottom_y) // 2

    head_margin = log_normal(jnp.abs(velocity), mini_hmargin).astype(int)

    def left_velocity(canvas):
        tail_top_left = (top_x + head_margin, mid_y - head_margin // 2)
        tail_bottom_right = (bottom_x, mid_y + head_margin // 2)
        canvas = draw_horizontal_tail(
            tail_top_left,
            tail_bottom_right,
            thickness,
            color,
            canvas,
        )
        head_top_left = (top_x, mid_y - head_margin // 2)
        head_bottom_right = (top_x + head_margin, mid_y + head_margin // 2)
        canvas = draw_triangle(
            head_top_left, head_bottom_right, color, canvas, direction=3
        )
        return canvas

    def right_velocity(canvas):
        tail_top_left = (top_x, mid_y - head_margin // 2)
        tail_bottom_right = (bottom_x - head_margin, mid_y + head_margin // 2)
        canvas = draw_horizontal_tail(
            tail_top_left, tail_bottom_right, thickness, color, canvas
        )
        head_top_left = (bottom_x - head_margin, mid_y - head_margin // 2)
        head_bottom_right = (bottom_x, mid_y + head_margin // 2)
        canvas = draw_triangle(
            head_top_left, head_bottom_right, color, canvas, direction=4
        )
        return canvas

    canvas = lax.select(velocity > 0, right_velocity(canvas), left_velocity(canvas))
    return canvas


def draw_vertical_arrow(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    color: chex.Array,
    velocity: chex.Array,
    canvas: chex.Array,
) -> chex.Array:
    """Draws a vertical arrow on a blank (256x256x3) canvas."""

    if canvas.shape[0] == 192:
        mini_th = 2
        mini_hmargin = 8
    else:
        mini_th = 2
        mini_hmargin = 6

    thickness = log_normal(jnp.abs(velocity), mini_th).astype(int)
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right
    mid_x = (top_x + bottom_x) // 2

    head_margin = log_normal(jnp.abs(velocity), mini_hmargin).astype(int)

    def up_velocity(canvas):
        tail_top_left = (mid_x - head_margin // 2, top_y + head_margin)
        tail_bottom_right = (mid_x + head_margin // 2, bottom_y)

        canvas = draw_vertical_tail(
            tail_top_left, tail_bottom_right, thickness, color, canvas
        )

        head_top_left = (mid_x - head_margin // 2, top_y)
        head_bottom_right = (mid_x + head_margin // 2, top_y + head_margin)

        canvas = draw_triangle(
            head_top_left, head_bottom_right, color, canvas, direction=1
        )
        return canvas

    def down_velocity(canvas):
        tail_top_left = (mid_x - head_margin // 2, top_y)
        tail_bottom_right = (mid_x + head_margin // 2, bottom_y - head_margin)

        canvas = draw_vertical_tail(
            tail_top_left, tail_bottom_right, thickness, color, canvas
        )

        head_top_left = (mid_x - head_margin // 2, bottom_y - head_margin)
        head_bottom_right = (mid_x + head_margin // 2, bottom_y)

        canvas = draw_triangle(
            head_top_left, head_bottom_right, color, canvas, direction=2
        )

        return canvas

    canvas = lax.select(velocity > 0, down_velocity(canvas), up_velocity(canvas))

    return canvas


def draw_crooked_arrow(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    color: chex.Array,
    angular_velocity: chex.Array,
    canvas: chex.Array,
) -> chex.Array:
    """Draws a crooked arrow on a blank (256x256x3) canvas."""

    if canvas.shape[0] == 192:
        mini_th = 2
        mini_hmargin = 8
    else:
        mini_th = 2
        mini_hmargin = 6

    thickness = log_normal(jnp.abs(angular_velocity), mini_th).astype(int)

    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right
    canvas = draw_crooked_tail(top_left, bottom_right, color, thickness, canvas)

    head_margin = log_normal(abs(angular_velocity), mini_hmargin).astype(int)

    mid_x = (top_x + bottom_x) // 2
    mid_y = (top_y + bottom_y) // 2

    def right_velocity(canvas):
        mid_ring = (bottom_x - thickness // 2, mid_y)
        head_top_left = (mid_ring[0] - head_margin // 2, mid_y)
        head_bottom_right = (mid_ring[0] + head_margin // 2, mid_y + head_margin)
        canvas = draw_triangle(
            head_top_left, head_bottom_right, color, canvas, direction=2
        )
        return canvas

    def left_velocity(canvas):
        mid_ring = (mid_x, top_y + thickness // 2)

        head_top_left = (mid_x - head_margin, mid_ring[1] - head_margin // 2)
        head_bottom_right = (mid_x, mid_ring[1] + head_margin // 2)
        canvas = draw_triangle(
            head_top_left, head_bottom_right, color, canvas, direction=3
        )
        return canvas

    canvas = lax.select(
        angular_velocity > 0, right_velocity(canvas), left_velocity(canvas)
    )
    return canvas


def rotate(
    image: chex.Array,
    angle: float,
    center: Tuple[int | Array, int | Array],
    order: int = 1,
    mode: str = "nearest",
    cval: float = 0.0,
) -> chex.Array:
    """Rotate an image around a center point."""
    c = jnp.cos(-angle)
    s = jnp.sin(-angle)
    matrix = jnp.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    # Use the offset to place the rotation at the image center.
    center = jnp.asarray(center)
    image_center = (jnp.asarray(image.shape) - 1.0) / 2.0
    image_center = image_center.at[:2].set(center)
    offset = image_center - matrix @ image_center

    return dm.affine_transform(
        image, matrix, offset=offset, order=order, mode=mode, cval=cval
    )


def draw_pole(
    start: Tuple[int | Array, int | Array],
    end: Tuple[int | Array, int | Array],
    color: chex.Array,
    angle: float,
    thickness: int,
    canvas: chex.Array,
) -> chex.Array:
    start = jnp.array(start, dtype=jnp.float32)
    end = jnp.array(end, dtype=jnp.float32)
    color = jnp.asarray(color, dtype=canvas.dtype)

    def rotate_point(point, angle, center):
        x, y = point
        cx, cy = center
        x_rot = cx + (x - cx) * jnp.cos(angle) - (y - cy) * jnp.sin(angle)
        y_rot = cy + (x - cx) * jnp.sin(angle) + (y - cy) * jnp.cos(angle)
        return jnp.array([x_rot, y_rot])

    def vectorized_circle(canvas, center, radius):
        h, w = canvas.shape[0], canvas.shape[1]
        y_grid, x_grid = jnp.mgrid[:h, :w]
        dist_sq = (x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2
        mask = (
            (dist_sq <= radius**2)
            & (x_grid >= 0)
            & (x_grid < w)
            & (y_grid >= 0)
            & (y_grid < h)
        )
        return canvas + mask[..., None] * color

    def vectorized_line(canvas, start, end, thickness):
        h, w = canvas.shape[0], canvas.shape[1]
        y_grid, x_grid = jnp.mgrid[:h, :w].astype(jnp.float32)

        x0, y0 = start[0], start[1]
        x1, y1 = end[0], end[1]
        dx = x1 - x0
        dy = y1 - y0
        line_len_sq = dx**2 + dy**2 + 1e-8

        t = ((x_grid - x0) * dx + (y_grid - y0) * dy) / line_len_sq
        t_clipped = jnp.clip(t, 0.0, 1.0)
        proj_x = x0 + t_clipped * dx
        proj_y = y0 + t_clipped * dy

        dist_sq = (x_grid - proj_x) ** 2 + (y_grid - proj_y) ** 2
        line_mask = dist_sq <= (thickness**2)

        valid = (x_grid >= 0) & (x_grid < w) & (y_grid >= 0) & (y_grid < h)
        return canvas + (line_mask & valid)[..., None] * color

    new_end = rotate_point(end, angle, start)
    canvas = vectorized_line(canvas, start, new_end, thickness)
    canvas = vectorized_circle(canvas, start, thickness)
    canvas = vectorized_circle(canvas, new_end, thickness)

    return jnp.clip(canvas, 0, 255)


def return_letter_patterns(letter: int) -> chex.Array:
    patterns = {
        65: [
            [False, True, True, True, False],
            [True, False, False, False, True],
            [True, True, True, True, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
        ],
        66: [
            [True, True, True, True, False],
            [True, False, False, False, True],
            [True, True, True, True, False],
            [True, False, False, False, True],
            [True, True, True, True, False],
        ],
        67: [
            [False, True, True, True, True],
            [True, False, False, False, False],
            [True, False, False, False, False],
            [True, False, False, False, False],
            [False, True, True, True, True],
        ],
        68: [
            [True, True, True, True, False],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, True, True, True, False],
        ],
        69: [
            [True, True, True, True, True],
            [True, False, False, False, False],
            [True, True, True, True, True],
            [True, False, False, False, False],
            [True, True, True, True, True],
        ],
        70: [
            [True, True, True, True, True],
            [True, False, False, False, False],
            [True, True, True, True, False],
            [True, False, False, False, False],
            [True, False, False, False, False],
        ],
        71: [
            [False, True, True, True, False],
            [True, False, False, False, False],
            [True, False, True, True, True],
            [True, False, False, False, True],
            [False, True, True, True, False],
        ],
        72: [
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, True, True, True, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
        ],
        73: [
            [True, True, True, True, True],
            [False, False, True, False, False],
            [False, False, True, False, False],
            [False, False, True, False, False],
            [True, True, True, True, True],
        ],
        74: [
            [False, False, False, True, False],
            [False, False, False, True, False],
            [False, False, False, True, False],
            [True, False, False, True, False],
            [True, True, True, False, False],
        ],
        75: [
            [True, False, False, False, True],
            [True, False, False, True, False],
            [True, True, True, False, False],
            [True, False, False, True, False],
            [True, False, False, False, True],
        ],
        76: [
            [True, False, False, False, False],
            [True, False, False, False, False],
            [True, False, False, False, False],
            [True, False, False, False, False],
            [True, True, True, True, True],
        ],
        77: [
            [True, False, False, False, True],
            [True, True, False, True, True],
            [True, False, True, False, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
        ],
        78: [
            [True, False, False, False, True],
            [True, True, False, False, True],
            [True, False, True, False, True],
            [True, False, False, True, True],
            [True, False, False, False, True],
        ],
        79: [
            [False, True, True, True, False],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [False, True, True, True, False],
        ],
        80: [
            [True, True, True, True, False],
            [True, False, False, False, True],
            [True, True, True, True, False],
            [True, False, False, False, False],
            [True, False, False, False, False],
        ],
        81: [
            [False, True, True, True, False],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, False, False, True, False],
            [False, True, True, False, True],
        ],
        82: [
            [True, True, True, True, False],
            [True, False, False, False, True],
            [True, True, True, True, False],
            [True, False, False, True, False],
            [True, False, False, False, True],
        ],
        83: [
            [True, True, True, True, True],
            [True, False, False, False, False],
            [True, True, True, True, True],
            [False, False, False, False, True],
            [True, True, True, True, True],
        ],
        84: [
            [True, True, True, True, True],
            [False, False, True, False, False],
            [False, False, True, False, False],
            [False, False, True, False, False],
            [False, False, True, False, False],
        ],
        85: [
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [False, True, True, True, False],
        ],
        86: [
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [False, True, False, True, False],
            [False, False, True, False, False],
        ],
        87: [
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, False, True, False, True],
            [True, True, False, True, True],
            [True, False, False, False, True],
        ],
        88: [
            [True, False, False, False, True],
            [False, True, False, True, False],
            [False, False, True, False, False],
            [False, True, False, True, False],
            [True, False, False, False, True],
        ],
        89: [
            [True, False, False, False, True],
            [False, True, False, True, False],
            [False, False, True, False, False],
            [False, False, True, False, False],
            [False, False, True, False, False],
        ],
        90: [
            [True, True, True, True, True],
            [False, False, False, True, False],
            [False, False, True, False, False],
            [False, True, False, False, False],
            [True, True, True, True, True],
        ],
    }

    patterns_stack = jnp.stack(
        [
            jnp.array(patterns[65], dtype=bool),
            jnp.array(patterns[66], dtype=bool),
            jnp.array(patterns[67], dtype=bool),
            jnp.array(patterns[68], dtype=bool),
            jnp.array(patterns[69], dtype=bool),
            jnp.array(patterns[70], dtype=bool),
            jnp.array(patterns[71], dtype=bool),
            jnp.array(patterns[72], dtype=bool),
            jnp.array(patterns[73], dtype=bool),
            jnp.array(patterns[74], dtype=bool),
            jnp.array(patterns[75], dtype=bool),
            jnp.array(patterns[76], dtype=bool),
            jnp.array(patterns[77], dtype=bool),
            jnp.array(patterns[78], dtype=bool),
            jnp.array(patterns[79], dtype=bool),
            jnp.array(patterns[80], dtype=bool),
            jnp.array(patterns[81], dtype=bool),
            jnp.array(patterns[82], dtype=bool),
            jnp.array(patterns[83], dtype=bool),
            jnp.array(patterns[84], dtype=bool),
            jnp.array(patterns[85], dtype=bool),
            jnp.array(patterns[86], dtype=bool),
            jnp.array(patterns[87], dtype=bool),
            jnp.array(patterns[88], dtype=bool),
            jnp.array(patterns[89], dtype=bool),
            jnp.array(patterns[90], dtype=bool),
        ]
    )
    return patterns_stack[letter - 65]


def draw_letter(
    top_left: chex.Array,
    bottom_right: chex.Array,
    color: chex.Array,
    canvas: chex.Array,
    letter_code: int,
) -> chex.Array:
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right

    width = (bottom_x - top_x) // 5
    height = (bottom_y - top_y) // 5

    letter_pattern = return_letter_patterns(letter_code)

    H, W = canvas.shape[:2]

    xx = jnp.arange(W)  # (W,)
    yy = jnp.arange(H)  # (H,)

    valid_x = (xx >= top_x) & (xx < top_x + 5 * width)
    valid_y = (yy >= top_y) & (yy < top_y + 5 * height)
    valid_mask = valid_y[:, None] & valid_x[None, :]  # (H, W)

    cell_x = ((xx - top_x) // width).astype(int)
    cell_y = ((yy - top_y) // height).astype(int)

    pattern_mask = letter_pattern[cell_y[:, None], cell_x[None, :]]

    final_mask = valid_mask & pattern_mask
    return jnp.where(final_mask[..., None], color, canvas)


def draw_words_h(
    top_left: chex.Array,
    bottom_right: chex.Array,
    color: chex.Array,
    canvas: chex.Array,
    letters: chex.Array,
) -> chex.Array:

    if canvas.shape[0] == 256:
        letter_width = 20
        margin = 2
        space = 3
    else:
        letter_width = 10
        margin = 1
        space = 1

    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right

    num_letters = letters.shape[0]
    total_width = num_letters * letter_width + (num_letters - 1) * space
    start_col = top_x + (bottom_x - top_x - total_width) // 2

    indices = jnp.arange(num_letters, dtype=jnp.uint8)
    current_cols = start_col + indices * (letter_width + space)
    top_left_letters = jnp.stack(
        [current_cols, jnp.full(num_letters, top_y + margin, dtype=jnp.uint8)], axis=1
    )
    bottom_right_letters = jnp.stack(
        [
            current_cols + letter_width,
            jnp.full(num_letters, bottom_y - margin, dtype=jnp.uint8),
        ],
        axis=1,
    )

    def body_fn(i, canvas_acc):
        return draw_letter(
            top_left_letters[i], bottom_right_letters[i], color, canvas_acc, letters[i]
        )

    return lax.fori_loop(0, num_letters, body_fn, canvas)


def draw_words_v(
    top_left: chex.Array,
    bottom_right: chex.Array,
    color: chex.Array,
    canvas: chex.Array,
    letters: chex.Array,
) -> chex.Array:

    if canvas.shape[0] == 256:
        letter_width = 20
        margin = 2
        space = 3
    else:
        letter_width = 10
        margin = 1
        space = 2

    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right

    num_letters = letters.shape[0]
    total_height = num_letters * letter_width + (num_letters - 1) * space
    start_row = top_y + (bottom_y - top_y - total_height) // 2

    indices = jnp.arange(num_letters)
    current_rows = start_row + indices * (letter_width + space)
    top_left_letters = jnp.stack(
        [jnp.full(num_letters, top_x + margin, dtype=jnp.uint8), current_rows], axis=1
    )
    bottom_right_letters = jnp.stack(
        [
            jnp.full(num_letters, bottom_x - margin, dtype=jnp.uint8),
            current_rows + letter_width,
        ],
        axis=1,
    )

    def body_fn(i, canvas_acc):
        return draw_letter(
            top_left_letters[i], bottom_right_letters[i], color, canvas_acc, letters[i]
        )

    return lax.fori_loop(0, num_letters, body_fn, canvas)

@partial(jax.jit, static_argnums=(4, 5))
def draw_str(
    top_left: Tuple[int | Array, int | Array],
    bottom_right: Tuple[int | Array, int | Array],
    color: chex.Array,
    canvas: chex.Array,
    word: str,
    horizontal: bool = True,
) -> chex.Array:
    """
    Draw a string on the canvas with optimized vectorized operations
    """
    # Convert string to ASCII codes with vectorized uppercase conversion
    arr = jnp.frombuffer(word.encode("ascii"), dtype=jnp.uint8)
    mask = (arr >= ord("a")) & (arr <= ord("z"))
    letter = jnp.where(mask, arr - 32, arr).astype(jnp.uint8)

    return lax.cond(
        horizontal,
        lambda: draw_words_h(top_left, bottom_right, color, canvas, letter),
        lambda: draw_words_v(top_left, bottom_right, color, canvas, letter),
    )
