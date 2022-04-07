#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40GB
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

module purge
module load nvidia-hpc-sdk/21.7

source /home1/svadiraj/environments/neccam_slt_3.7.6/bin/activate
cd ../trial

python img2vec.py --dataset=Phoenix --base_folder=/scratch2/maiyaupp/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/ --out_folder=/scratch2/maiyaupp/phoenix/phoenix_resnet --subset=train --start_ind=0 --end_ind=10 --batch_size=100
