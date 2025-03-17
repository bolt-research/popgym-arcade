import popgym_arcade
import jax
import pygame
import numpy as np

# Change these to play other games
ENV_NAME = "BattleShipEasy"
IS_POMDP = True
OBS_SIZE = 128

def to_surf(arr):
    # Convert jax arry to pygame surface
    return np.transpose(np.array(observation * 255).astype(np.uint8), (1, 0, 2))


# Create env env variant
env, env_params = popgym_arcade.make(
    ENV_NAME, partial_obs=IS_POMDP, obs_size=OBS_SIZE
)

# Vectorize and compile the env
env_reset = jax.jit(env.reset)
env_step = jax.jit(env.step)

# Initialize environment
key = jax.random.PRNGKey(0)
key, reset_key = jax.random.split(key)
observation, env_state = env_reset(reset_key, env_params)
done = False

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((OBS_SIZE, OBS_SIZE))
clock = pygame.time.Clock()
running = True

# Convert numpy array to Pygame surface
surface = pygame.surfarray.make_surface(to_surf(observation))

# Action mappings (modify based on your environment's action space)
ACTION_MEANINGS = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 3,
    pygame.K_SPACE: 4,
    # Add more keys if needed
}
print("Controls: up, down, left, right, spacebar")




while running:
    # Handle events
    action = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in ACTION_MEANINGS:
                action = ACTION_MEANINGS[event.key]
    
    
    # Render to screen
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    
    # Take action if key pressed
    if action is not None:
        key, step_key = jax.random.split(key)
        observation, env_state, reward, done, info = env_step(step_key, env_state, action, env_params)
        surface = pygame.surfarray.make_surface(to_surf(observation))
        
        # Render to screen
        if done:
            print("Game over")
            break
    
    clock.tick(30)  # Control FPS

pygame.quit()