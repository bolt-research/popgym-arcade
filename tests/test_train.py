import pytest
import subprocess

# TODO: Add DQN tests when replay buffer library uses latest python version


def test_pqn_train():
    """Test that the training script runs without errors."""
    result = subprocess.run(
        ["python", "popgym_arcade/train.py", "PQN", "--NUM_STEPS", "2", "--TOTAL_TIMESTEPS", "8", "--NUM_ENVS", "2", "--WANDB_MODE", "disabled", "--NUM_MINIBATCHES", "1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"PQN training script failed with error: {result.stderr}"

def test_pqn_rnn_train():
    """Test that the training script runs without errors."""
    result = subprocess.run(
        ["python", "popgym_arcade/train.py", "PQN_RNN", "--NUM_STEPS", "2", "--TOTAL_TIMESTEPS", "8", "--NUM_ENVS", "2", "--MEMORY_TYPE", "LRU", "--WANDB_MODE", "disabled", "--NUM_MINIBATCHES", "1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"PQN_RNN training script failed with error: {result.stderr}"

def test_ppo_rnn_train():
    """Test that the training script runs without errors."""
    result = subprocess.run(
        ["python", "popgym_arcade/train.py", "PPO_RNN", "--NUM_STEPS", "2", "--TOTAL_TIMESTEPS", "8", "--NUM_ENVS", "2", "--MEMORY_TYPE", "LRU", "--WANDB_MODE", "disabled", "--NUM_MINIBATCHES", "1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"PPO_RNN training script failed with error: {result.stderr}"

def test_ppo_train():
    """Test that the training script runs without errors."""
    result = subprocess.run(
        ["python", "popgym_arcade/train.py", "PPO", "--NUM_STEPS", "2", "--TOTAL_TIMESTEPS", "8", "--NUM_ENVS", "2", "--WANDB_MODE", "disabled", "--NUM_MINIBATCHES", "1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"PPO training script failed with error: {result.stderr}"

if __name__ == "__main__":
    pytest.main()
