#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8        #maximum cpu limit for each v100 GPU is 6 , each a100 GPU is 8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G      #maximum memory limit for each v100 GPU is 90G , each a100 GPU is 40G
#SBATCH --partition=gbunchQ2 #default batch is a100
#SBATCH --output=output/output_%j.txt

set -e
export WANDB_API_KEY=7a8e49e11981e6fd8f3b4b0640616c6e93a05c86 # FILL THIS OUT 
eval "$(conda shell.bash hook)"
conda activate jaxenv
python sweep.py
#python popgym_arcade/train.py PQN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "BreakoutEasy"
#python popgym_arcade/train.py PQN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "TetrisEasy"
#python popgym_arcade/train.py PQN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "SKittlesEasy"

# python popgym_arcade/train.py PQN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "TetrisEasy"
#python popgym_arcade/train.py PQN_RNN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "BreakoutEasy" --PARTIAL --MEMORY_TYPE lru
#python popgym_arcade/train.py PQN_RNN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "TetrisEasy" --PARTIAL --MEMORY_TYPE lru
#python popgym_arcade/train.py PQN_RNN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "SkittlesEasy" --PARTIAL --MEMORY_TYPE lru



# python popgym_arcade/train.py PQN_RNN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "TetrisEasy" --PARTIAL --MEMORY_TYPE lru

# python popgym_arcade/train.py PQN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 10000000 --ENV_NAME "SkittlesEasy"

# python popgym_arcade/train.py PQN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "SkittlesEasy"

# python popgym_arcade/train.py PQN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 10000000 --ENV_NAME "SkittlesHard"

# python popgym_arcade/train.py PQN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "SkittlesHard"

# python popgym_arcade/train.py PQN_RNN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 10000000 --ENV_NAME "SkittlesEasy" --PARTIAL --MEMORY_TYPE lru

# python popgym_arcade/train.py PQN_RNN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "SkittlesEasy" --PARTIAL --MEMORY_TYPE lru

# python popgym_arcade/train.py PQN_RNN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 10000000 --ENV_NAME "SkittlesHard" --PARTIAL --MEMORY_TYPE lru

# python popgym_arcade/train.py PQN_RNN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "SkittlesHard" --PARTIAL --MEMORY_TYPE lru



# python popgym_arcade/train.py PQN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 10000000 --ENV_NAME "BreakoutEasy"

# python popgym_arcade/train.py PQN_RNN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 10000000 --ENV_NAME "BreakoutEasy" --MEMORY_TYPE lru --PARTIAL

# python popgym_arcade/train.py PQN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "BreakoutEasy"

# python popgym_arcade/train.py PQN_RNN --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "BreakoutEasy" --MEMORY_TYPE lru --PARTIAL

if [[ $SLURM_JOB_ID ]]; then
    echo "Resubmitting job"
    NEW_JOBID=$(sbatch --dependency=afterok:$SLURM_JOB_ID $0 | awk '{print $4}')
    echo "Job $NEW_JOBID resubmitted"
fi
