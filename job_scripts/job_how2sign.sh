#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

module purge
module load nvidia-hpc-sdk/21.7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

source /home1/maiyaupp/slt/venv/bin/activate
cd ../trial

python img2vec.py --dataset=How2Sign --base_folder=/scratch2/maiyaupp/how2sign/ --out_folder=/scratch2/maiyaupp/how2sign/how2sign_vitb16/ --subset=train --start_ind=2001 --end_ind=4999 --batch_size=100
