#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1

module purge
module load nvidia-hpc-sdk/21.7

source ~/environments/openpose_3.7.6/bin/activate
cd ../trial

python img2pose.py --dataset=Phoenix --base_folder=/scratch1/maiyaupp/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/ --out_folder=/scratch1/maiyaupp/phoenix/PHOENIX-POSE/ --subset=test --start_ind=0 --end_ind=-1