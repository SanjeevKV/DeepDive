#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1

module purge
module load gcc/8.3.0
module load python/3.7.6
module load nvidia-hpc-sdk/21.7

# nvcc cuda_program.co -o cuda_program
# ./cuda_program

source env/bin/activate
python -m signjoey train configs/sign.yaml
deactivate
