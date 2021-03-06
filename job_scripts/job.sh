#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1

module purge
module load nvidia-hpc-sdk/21.7

source ~/slt/venv/bin/activate
cd ../trial

python img2vec.py --dataset=How2Sign --base_folder=/scratch2/maiyaupp/how2sign/ --out_folder=/scratch1/maiyaupp/how2sign/how2sign_embedding_mimic/ --subset=train --start_ind=14000 --end_ind=15999 --batch_size=100