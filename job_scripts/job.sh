#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1

module purge
module load nvidia-hpc-sdk/21.7

nvcc cuda_program.cu -o cuda_program

./cuda_program
source ~/environments/neccam_slt_3.7.6/bin/activate
cd /home1/svadiraj/projects/DeepDive/trial
python img2vec.py /scratch2/svadiraj/data/rwth_phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T /scratch2/svadiraj/data/rwth_phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/alex_out/