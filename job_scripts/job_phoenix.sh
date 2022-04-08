#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

module purge
module load nvidia-hpc-sdk/21.7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

source /home1/svadiraj/environments/neccam_slt_3.7.6/bin/activate
cd ../trial

python img2vec.py --dataset=Phoenix --base_folder=/scratch2/maiyaupp/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/ --out_folder=/scratch2/maiyaupp/phoenix/phoenix_vgg16/ --subset=test --start_ind=0 --end_ind=-1 --batch_size=10
