#!/bin/bash
#SBATCH --job-name=LRU_PQNBSM    # job name
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16        #maximum cpu limit for each v100 GPU is 6 , each a100 GPU is 8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G      #maximum memory limit for each v100 GPU is 90G , each a100 GPU is 40G
#SBATCH --output=./output/output_PQNLRUBattleShipMedium.txt # change this to a output file
#SBATCH --partition=a100_batch    #default batch is a100 

# source ~/.bashrc    # this 2 lines need to uncomment 
# source activate jaxenv           #need change 'myenv' to your environment # this 2 lines need to uncomment 

# CartPole
for dif in Easy Medium Hard; do python popgym_arcade/train.py PQN --SEED 0 --TOTAL_TIMESTEPS 10000000 --TOTAL_TIMESTEPS_DECAY 100000 --ENV_NAME "CartPole${dif}" --PROJECT "CartPoleTest"; done
for dif in Easy Medium Hard; do python popgym_arcade/train.py PQN_RNN --SEED 0 --TOTAL_TIMESTEPS 10000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "CartPole${dif}" --PROJECT "CartPoleTest" --PARTIAL "True" --MEMORY_TYPE "lru"; done
# Below is using MLP to train POMDP env
# for dif in Easy Medium Hard; do python popgym_arcade/train.py PQN --SEED 0 --TOTAL_TIMESTEPS 10000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "CartPole${dif}" --PROJECT "CartPoleTest" --PARTIAL "True"; done
# Below is using Memory model to train MDP env, in this term is "lru"
# for dif in Easy Medium Hard; do python popgym_arcade/train.py PQN_RNN --SEED 0 --TOTAL_TIMESTEPS 10000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "CartPole${dif}" --PROJECT "CartPoleTest" --PARTIAL "False" --MEMORY_TYPE "lru"; done


# BattleShip
for dif in Easy Medium Hard; do python popgym_arcade/train.py PQN --SEED 0 --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "BattleShip${dif}" --PROJECT "CartPoleTest"; done
for dif in Easy Medium Hard; do python popgym_arcade/train.py PQN_RNN --SEED 0 --TOTAL_TIMESTEPS 20000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "BattleShip${dif}" --PROJECT "CartPoleTest" --PARTIAL "True" --MEMORY_TYPE "lru"; done

# MineSweeper
for dif in Easy Medium Hard; do python popgym_arcade/train.py PQN --SEED 0 --TOTAL_TIMESTEPS 10000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "MineSweeper${dif}" --PROJECT "CartPoleTest"; done
for dif in Easy Medium Hard; do python popgym_arcade/train.py PQN_RNN --SEED 0 --TOTAL_TIMESTEPS 10000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "MineSweeper${dif}" --PROJECT "CartPoleTest" --PARTIAL "True" --MEMORY_TYPE "lru"; done

# Navigator
for dif in Easy Medium Hard; do python popgym_arcade/train.py PQN --SEED 0 --TOTAL_TIMESTEPS 10000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "Navigator${dif}" --PROJECT "CartPoleTest"; done
for dif in Easy Medium Hard; do python popgym_arcade/train.py PQN_RNN --SEED 0 --TOTAL_TIMESTEPS 10000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "Navigator${dif}" --PROJECT "CartPoleTest" --PARTIAL "True" --MEMORY_TYPE "lru"; done

# AutoEncode
for dif in Easy Medium Hard; do python popgym_arcade/train.py PQN --SEED 0 --TOTAL_TIMESTEPS 10000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "AutoEncode${dif}" --PROJECT "CartPoleTest"; done
for dif in Easy Medium Hard; do python popgym_arcade/train.py PQN_RNN --SEED 0 --TOTAL_TIMESTEPS 10000000 --TOTAL_TIMESTEPS_DECAY 2000000 --ENV_NAME "AutoEncode${dif}" --PROJECT "CartPoleTest" --PARTIAL "True" --MEMORY_TYPE "lru"; done



wait
