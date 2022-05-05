#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1

module purge
module load nvidia-hpc-sdk/21.7

source /home1/vrundhas/environments/alex_env/bin/activate
cd ../trial
python shuffle_signers.py

 # 8223958