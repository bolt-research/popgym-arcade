#!/bin/bash
#SBATCH --job-name=popgym-sweep # job name
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8        #maximum cpu limit for each v100 GPU is 6 , each a100 GPU is 8
#SBATCH --gres=gpu:1
#SBATCH --mem=20G      #maximum memory limit for each v100 GPU is 90G , each a100 GPU is 40G
#SBATCH --partition=gbunchQ1 #default batch is a100 
set -e
export WANDB_API_KEY= # FILL THIS OUT 
eval "$(conda shell.bash hook)"
conda activate arcade
python ~/code/popgym_arcade/sweep.py

if [[ $SLURM_JOB_ID ]]; then
    echo "Resubmitting job"
    NEW_JOBID=$(sbatch --dependency=afterok:$SLURM_JOB_ID $0 | awk '{print $4}')
    echo "Job $NEW_JOBID resubmitted"
fi