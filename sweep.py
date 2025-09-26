import subprocess
import itertools
import os
from wandb.apis.public import Api
from typing import Dict, Any

# Configuration - customize these mappings
WANDB_PROJECT = "Arcade-NIPS"
WANDB_ENTITY = "bolt-um"  # Optional, unless you're in a team
MAX_JOBS = 7 # Maximum number of jobs to run before terminating (useful for HPC/SLURM)
TRAIN_PATH = "/home/mc45189/breakouttetris/popgym-arcade/popgym_arcade/train.py"

algorithm_families = ['PQN']
models = ['mlp', 'lru', 'mingru', 'gru']
seeds = [0,1,2,3,4]
environments_config = {
    "CartPoleEasy": {
        "PPO": int(1e7),
        "PQN": int(1e7),
        "TOTAL_TIMESTEPS_DECAY": int(1e6) # can also be 2e6
    },
    "CartPoleMedium": {
        "PPO": int(1e7),
        "PQN": int(1e7),
        "TOTAL_TIMESTEPS_DECAY": int(1e6)
    },
    "CartPoleHard": {
        "PPO": int(1e7),
        "PQN": int(1e7),
        "TOTAL_TIMESTEPS_DECAY": int(1e6)
    },
    "NoisyCartPoleEasy": {
        "PPO": int(1e7),
        "PQN": int(1e7),
        "TOTAL_TIMESTEPS_DECAY": int(1e6)
    },
    "NoisyCartPoleMedium": {
        "PPO": int(1e7),
        "PQN": int(1e7),
        "TOTAL_TIMESTEPS_DECAY": int(1e6)
    },
    "NoisyCartPoleHard": {
        "PPO": int(1e7),
        "PQN": int(1e7),
        "TOTAL_TIMESTEPS_DECAY": int(1e6)
    },
    "BattleShipEasy": {
        "PPO": int(2e7),  # Different timesteps for PPO
        "PQN": int(2e7),  # Different timesteps for PQN
        "TOTAL_TIMESTEPS_DECAY": int(2e6)  # New decay parameter for PQN
    },
    "BattleShipMedium": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "BattleShipHard": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "CountRecallEasy": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "CountRecallMedium": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "CountRecallHard": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "NavigatorEasy": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "NavigatorMedium": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "NavigatorHard": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "MineSweeperEasy": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "MineSweeperMedium": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "MineSweeperHard": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "AutoEncodeEasy": {
        "PPO": int(1e7),
        "PQN": int(1e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "AutoEncodeMedium": {
        "PPO": int(1e7),
        "PQN": int(1e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "AutoEncodeHard": {
        "PPO": int(1e7),
        "PQN": int(1e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "TetrisEasy": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "TetrisMedium": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "TetrisHard": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "SkittlesEasy": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "SkittlesMedium": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "SkittlesHard": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "BreakoutEasy": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "BreakoutMedium": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
    "BreakoutHard": {
        "PPO": int(2e7),
        "PQN": int(2e7),
        "TOTAL_TIMESTEPS_DECAY": int(2e6)
    },
}
partial_flags = [True, False]

def is_rnn(model_str):
    return "mlp" not in model_str

def generate_experiment_key(experiment: Dict[str, Any]) -> str:
    """Create a unique key for an experiment configuration"""
    return (f"{experiment['algorithm']}_{experiment['model']}_"
            f"{experiment['seed']}_{experiment['environment']}_"
            f"{experiment['partial']}")


def get_wandb_runs() -> set:
    """Get completed or running experiments from WandB"""
    api = Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}") if WANDB_ENTITY else api.runs(WANDB_PROJECT)

    existing = set()
    for run in runs:
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        key = generate_experiment_key({
            "algorithm": config["TRAIN_TYPE"].replace("_RNN", ""),
            "model": config.get("MEMORY_TYPE", "mlp").lower(),
            "seed": config["SEED"],
            "environment": config["ENV_NAME"],
            "partial": config["PARTIAL"]
        })
        if run.state in ["finished", "running"]:
            existing.add(key)
    return existing


def build_base_command(experiment: Dict[str, Any]) -> list:
    """Construct the appropriate command based on model type"""

    algo = experiment['algorithm']
    algo += "_RNN" if is_rnn(experiment["model"]) else ""

    base_cmd = [
        "python", TRAIN_PATH,
        algo,
        "--PROJECT", WANDB_PROJECT,
        "--SEED", str(experiment["seed"]),
        "--ENV_NAME", experiment['environment'],
        "--TOTAL_TIMESTEPS", str(experiment["total_timesteps"]),
    ]

    base_cmd += ["--PARTIAL"] if experiment['partial'] else []

    if experiment['algorithm'] in ["PQN", "PQN_RNN"]:
        base_cmd += [
            "--TOTAL_TIMESTEPS_DECAY", str(experiment["total_timesteps_decay"])
        ]

    if is_rnn(experiment['model']):
        base_cmd += ["--MEMORY_TYPE", experiment['model']]

    return base_cmd


def get_all_experiments():
    """Return all possible experiments"""
    all_experiments = []
    for env, config in environments_config.items():
        combinations = itertools.product(seeds, algorithm_families, models, partial_flags)
        for seed, family, model, partial in combinations:

            # Get timesteps specific to algorithm family
            total_timesteps = config[family]  # PPO or PQN

            all_experiments.append({
                "algorithm": family,
                "model": model,
                "total_timesteps": total_timesteps,
                "total_timesteps_decay": config["TOTAL_TIMESTEPS_DECAY"],  # Include in config
                "seed": seed,
                "environment": env,
                "partial": partial
            })
    return all_experiments

def get_pending_experiments(all_experiments):
    """Return experiments that we plan to run"""
    # Generate all possible experiment combinations

    # Get completed/running experiments from WandB
    completed_or_running = get_wandb_runs()
    # Find pending experiments
    pending_experiments = [
        exp for exp in all_experiments
        if generate_experiment_key(exp) not in completed_or_running
    ]
    return completed_or_running, pending_experiments

def main():
    all_experiments = get_all_experiments()

    # Run experiments sequentially
    for i in range(MAX_JOBS):
    #for i, experiment in enumerate(pending_experiments):
        completed_or_running, pending_experiments = get_pending_experiments(all_experiments)
        print("Currently running or completed experiments:")
        print(completed_or_running)

        if not pending_experiments:
            print("All experiments have been completed or are running!")
            break

        # Run experiment
        experiment = pending_experiments[0]

        print(f"Found {len(pending_experiments)} pending experiments out of {len(all_experiments)} total")
        print(f"\n=== Starting experiment {i + 1}/{len(pending_experiments)} ===")
        print("Configuration:", experiment)

        # Build command
        base_cmd = build_base_command(experiment)

        # Run experiment
        print("Command:", " ".join(base_cmd))
        result = subprocess.run(base_cmd, check=False, env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})

        if result.returncode != 0:
            print(f"Experiment failed with exit code {result.returncode}")
            # Handle failure as needed

        if i + 1 == MAX_JOBS:
            print(f"Reached maximum number of jobs ({MAX_JOBS}), terminating")
            break
        i += 1


if __name__ == "__main__":
    main()
