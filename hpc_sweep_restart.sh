#!/bin/bash
#SBATCH --job-name=arcade
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --partition=h800_batch
#SBATCH --output=output/output_%j.txt


set -e
export WANDB_API_KEY=7a8e49e11981e6fd8f3b4b0640616c6e93a05c86
eval "$(conda shell.bash hook)"
conda activate nanojax

python sweep.py
# python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "lstm" --ENV_NAME "MineSweeperEasy" --PARTIAL  --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000
# python popgym_arcade/train.py PQN_RNN --MEMORY_TYPE "lstm" --ENV_NAME "CountRecallEasy" --PARTIAL  --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000
# for env in halfcheetah hopper walker2d pusher; do
#     python PDPG/pdpg_brax.py PDPG --ENV_NAME $env --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000
# done
# python PDPG/pdpg_brax.py PDPG
# ENV_NAMES=("NavigatorEasy" "BattleShipEasy" "CountRecallEasy" "MineSweeperEasy")
# END_EPSILONS=(0.05)

#for env_name in "${ENV_NAMES[@]}"; do
#    for end_epsilon in "${END_EPSILONS[@]}"; do
#        echo "Running env=$env_name, end_epsilon=$end_epsilon"
#        python collect.py --env_name "$env_name" --end_epsilon "$end_epsilon"
#    done
#done

if [[ $SLURM_JOB_ID ]]; then
   echo "Resubmitting job..."
   sbatch "$0"
fi
