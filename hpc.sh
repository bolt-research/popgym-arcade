#!/bin/bash
#SBATCH --job-name=arcade
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

wandb agent bolt-um/obsgaplru/jc3thme9
# wandb agent bolt-um/obsgapmingru/502iv9xx
