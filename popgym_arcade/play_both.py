#!/usr/bin/env python3

import argparse

import jax
import numpy as np
import pygame

import popgym_arcade
import popgym_arcade.registration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env", help="The env you want to play, for example BattleShipEasy"
    )
    parser.add_argument(
        "--obs-size",
        "-o",
        help="Pixel size of observations, can be 128 or 256",
        type=int,
        default=256,
    )
    args = parser.parse_args()

    assert (
        args.env in popgym_arcade.registration.REGISTERED_ENVIRONMENTS
    ), f"Invalid env: {args.env}, must be in {popgym_arcade.registration.REGISTERED_ENVIRONMENTS}"
    assert args.obs_size in [
        128,
        256,
    ], f"Invalid obs-size: {args.obs_size}, must be in [128, 256]"
    return args


def to_surf(arr):
    # Convert jax arry to pygame surface
    return np.transpose(np.array(arr, dtype=np.uint8), (1, 0, 2))


def play(args):
    # Create MDP env variant
    env_mdp, env_params_mdp = popgym_arcade.make(
        args.env, partial_obs=False, obs_size=args.obs_size
    )
    
    # Create POMDP env variant
    env_pomdp, env_params_pomdp = popgym_arcade.make(
        args.env, partial_obs=True, obs_size=args.obs_size
    )

    # Vectorize and compile the envs
    env_reset_mdp = jax.jit(env_mdp.reset)
    env_step_mdp = jax.jit(env_mdp.step)
    env_reset_pomdp = jax.jit(env_pomdp.reset)
    env_step_pomdp = jax.jit(env_pomdp.step)

    # Initialize environments with same seed
    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    observation_mdp, env_state_mdp = env_reset_mdp(reset_key, env_params_mdp)
    observation_pomdp, env_state_pomdp = env_reset_pomdp(reset_key, env_params_pomdp)
    done_mdp = False
    done_pomdp = False

    # Pygame setup - side by side display
    pygame.init()
    screen_width = args.obs_size * 2
    screen_height = args.obs_size
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    running = True

    # Convert numpy arrays to Pygame surfaces
    print(observation_mdp.dtype)
    surface_mdp = pygame.surfarray.make_surface(to_surf(observation_mdp))
    surface_pomdp = pygame.surfarray.make_surface(to_surf(observation_pomdp))

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
    print("MDP (left) | POMDP (right)")

    while running:
        # Handle events
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in ACTION_MEANINGS:
                    action = ACTION_MEANINGS[event.key]

        # Render both environments to screen
        screen.blit(surface_mdp, (0, 0))  # Left side
        screen.blit(surface_pomdp, (args.obs_size, 0))  # Right side
        pygame.display.flip()

        # Take action if key pressed
        if action is not None:
            key, step_key = jax.random.split(key)
            
            # Step both environments with the same action and key
            if not done_mdp:
                observation_mdp, env_state_mdp, reward_mdp, done_mdp, info_mdp = env_step_mdp(
                    step_key, env_state_mdp, action, env_params_mdp
                )
                surface_mdp = pygame.surfarray.make_surface(to_surf(observation_mdp))
            
            if not done_pomdp:
                observation_pomdp, env_state_pomdp, reward_pomdp, done_pomdp, info_pomdp = env_step_pomdp(
                    step_key, env_state_pomdp, action, env_params_pomdp
                )
                surface_pomdp = pygame.surfarray.make_surface(to_surf(observation_pomdp))

            # Check if both games are done
            if done_mdp and done_pomdp:
                print("Both games over")
                break

        clock.tick(30)  # Control FPS

    pygame.quit()


def main():
    args = parse_args()
    play(args)


if __name__ == "__main__":
    main()
