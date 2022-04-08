#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

module purge
module load nvidia-hpc-sdk/21.7

source /home1/svadiraj/environments/neccam_slt_3.7.6/bin/activate
cd ../slt
python -m signjoey train configs/sign.yaml
