#!/bin/bash
#SBATCH --job-name=popgym-sweep # job name
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8        #maximum cpu limit for each v100 GPU is 6 , each a100 GPU is 8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G      #maximum memory limit for each v100 GPU is 90G , each a100 GPU is 40G
#SBATCH --partition=a100_batch #default batch is a100 
set -e
export WANDB_API_KEY=cb3f47217a6fc585868ade5a936e8667bfbeb015 # FILL THIS OUT 
eval "$(conda shell.bash hook)"
conda activate arcade
python ~/code/popgym_arcade/sweep.py & 
sleep 60
python ~/code/popgym_arcade/sweep.py &
wait
