import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import requests
import base64
import io
from PIL import Image
from tqdm import tqdm

import popgym_arcade

output_filename = "vlm_game_results.csv"
seeds_per_env = [0] # seeds to run
environments_to_play = popgym_arcade.registration.REGISTERED_ENVIRONMENTS 
max_context = 80 # some models give errors if you go beyond max context, so truncate
    

# --- Configuration ---

# Fetch API key from environment variables
# To set it, run: export OPENROUTER_API_KEY="YOUR_KEY"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

# VLM model to use via OpenRouter.
# google/gemini-pro-vision is a strong, commonly used choice.
VLM_MODEL = "mistralai/mistral-small-3.1-24b-instruct"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Helper Functions ---

def image_to_base64(image_array: np.ndarray) -> str:
    """Converts a NumPy array (RGB) to a base64 encoded string."""
    img = Image.fromarray(np.array(image_array))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def clean_vlm_output(text: str) -> str | None:
    """Extracts the first valid action word from the VLM's text output."""
    if not text:
        return None

    try:
        a = int(text)
        assert (a >= 0) and (a <= 4)
        return a
    except (ValueError, AssertionError):
        print(f"Invalid action: {text}")
        return None

def query_vlm_for_action_conversational(
    messages_history: list,
    current_observation: np.ndarray
) -> tuple[str | None, list]:
    """
    Queries the VLM with a full conversation history and the new observation.

    Args:
        messages_history: The list of messages so far (system, user, assistant).
        current_observation: The current screen image as a NumPy array.

    Returns:
        A tuple containing:
        - The single action string (e.g., "left") or None.
        - The updated messages list including the new user query and assistant response.
    """
    base64_image = image_to_base64(current_observation)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # The new user message containing the latest observation
    user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Here is the next observation. What is your next single move? Output an integer in (0, 1, 2, 3, 4) and nothing else."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            },
        ],
    }

    # Append the new user message to the history to form the request
    payload_messages = messages_history + [user_message]
    # Truncate history if necessary
    payload_messages = payload_messages[0:1] + payload_messages[:-max_context + 1]

    payload = {
        "model": VLM_MODEL,
        "messages": payload_messages,
        "max_tokens": 20, # output tokens
    }

    try:
        response = requests.post(
            OPENROUTER_API_URL, headers=headers, json=payload, timeout=60 # Increased timeout
        )
        response.raise_for_status()
        response_json = response.json()
        
        # Extract the VLM's response content
        vlm_response_content = response_json["choices"][0]["message"]["content"]
        
        # Create the assistant message to be added to the history
        assistant_message = {"role": "assistant", "content": vlm_response_content}
        
        # The final, updated history for the next turn
        updated_history = payload_messages + [assistant_message]
        
        cleaned_action = clean_vlm_output(vlm_response_content)
        
        return cleaned_action, vlm_response_content, updated_history

    except requests.exceptions.RequestException as e:
        print(f"\nAPI Error: {e}")
    except (KeyError, IndexError) as e:
        print(f"\nError parsing VLM response: {e}")
        print(f"Full response: {response.text}")
        
    # On error, return None and the history *before* this failed attempt
    return None, None, messages_history

def play_episode(env_name: str, seed: int, pomdp: bool) -> float:
    """
    Plays one full episode of a Gymnasium environment using the VLM.

    Args:
        env_name: The name of the environment to play.
        seed: The seed for the environment's random number generator.

    Returns:
        The total undiscounted return for the episode.
    """
    print(f"\n--- Playing {env_name} with seed {seed} and POMDP={pomdp} ---")

    env, env_params = popgym_arcade.make(env_name, partial_obs=pomdp)
    step = jax.jit(env.step)
    reset = jax.jit(env.reset)
    docstring = env.__class__.__bases__[0].__doc__
    # Construct the full system prompt for the VLM
    system_prompt = (
        "You are playing a game described by the below prompt:\n"
        + docstring 
        + "\nYour goal is to achieve a high-score in the game. Note that the game may be"
        + " partially observable. At each timestep, you will"
        + " receive an observation and must output an action as a single integer from (0, 1, 2, 3, 4)."
        + " Do not provide any explanation or other text, just a single integer."
    )

    # Initialize environment
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    observation, env_state = reset(reset_key, env_params)

    total_reward = 0.0
    done = False
    t = 0
    messages = [{"role": "system", "content": system_prompt}]
    num_rand_actions = 0

    while not done:
        t += 1
        
        # Get action from VLM
        vlm_action_int, vlm_action_str, messages = query_vlm_for_action_conversational(messages, observation)

        if vlm_action_int is not None:
            action = jnp.array(vlm_action_int)
            action_source = "VLM"
        else:
            # If VLM fails or gives an invalid/unmapped action, take a random one
            key, action_key = jax.random.split(key)
            action = jax.random.randint(action_key, shape=(), minval=0, maxval=5)
            #action = env.action_space.sample()
            action_source = "Random"
            print(f"  [Step {t}] VLM output '{vlm_action_str}' was invalid/unmapped. Taking random action.")
            num_rand_actions += 1

        # Step the environment
        key, step_key = jax.random.split(key)
        observation, env_state, reward, done, info = step(
            step_key, env_state, action, env_params
        )
        total_reward += reward

        print(f"  [Step {t}] Source: {action_source}, Action: {vlm_action_int or 'N/A'}, Mapped: {action}, Reward: {reward:.4f}, Total Return: {total_reward:.2f}")

    print(f"--- Episode Finished ---")
    print(f"Final Return for {env_name} (seed={seed}): {total_reward}")
    return total_reward, num_rand_actions, t


def check_run(result_df, name, pomdp, seed):
    if result_df is None:
        return False
    search = (result_df['env_name'] == name) & (result_df['pomdp'] == pomdp) & (result_df['seed'] == seed)
    if not any(search):
        return False
    
    if any(result_df[search]["episode_length"] == -1):
        return False

    if any(result_df[search]["num_rand_actions"] == result_df[search]["episode_length"]):
        return False
    
    return True


def main():
    """Main function to run experiments and save results."""
    try:
        results_df = pd.read_csv(output_filename)
        results = results_df.to_dict(orient='records')
        print("Loaded CSV")
        print(results_df)
    except Exception:
        print("Failed to load CSV")
        results_df = None
        results = []

    for env_name in environments_to_play:
        for pomdp in [True, False]:
            for seed in tqdm(seeds_per_env, desc=f"Running {env_name}"):
                if check_run(results_df, env_name, pomdp, seed):
                    print(f"Found in dataset, skipping: {env_name} - {pomdp} - {seed}")
                    continue
                try:
                    episode_return, num_rand_actions, length = play_episode(env_name, seed, pomdp)
                    results.append({
                        "env_name": env_name,
                        "seed": seed,
                        "return": episode_return,
                        "pomdp": pomdp,
                        "num_rand_actions": num_rand_actions,
                        "episode_length": length,
                    })
                except Exception as e:
                    print(f"An error occurred while playing {env_name} with seed {seed}: {e}")
                    # Optionally, record the failure
                    results.append({
                        "env_name": env_name,
                        "seed": seed,
                        "return": float('nan'), # Use NaN to indicate failure
                        "pomdp": pomdp,
                        "num_rand_actions": -1,
                        "episode_length": -1,
                    })


                # --- Save Results ---
                if results:
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(output_filename, index=False)
                    print("Results saved to vlm_game_results.csv")
                    print(results_df)
                else:
                    print("Failed to run experiments, nothing saved")


if __name__ == "__main__":
    main()
