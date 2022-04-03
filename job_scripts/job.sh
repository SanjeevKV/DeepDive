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

source ~/slt/venv/bin/activate
cd /home1/maiyaupp/DeepDive/trial
python img2vec.py --dataset=Phoenix --base_folder=/scratch2/maiyaupp/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T --out_folder=/scratch2/maiyaupp/phoenix/phoenix_alexnet --subset=train --start_ind=0 --end_ind=1 --batch_size=100
