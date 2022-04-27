#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=12:00:00

source /home1/maiyaupp/slt/venv/bin/activate
cd ../trial

python img2vec.py --dataset=Phoenix --base_folder=/scratch1/maiyaupp/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/ --out_folder=/scratch1/maiyaupp/phoenix/phoenix_compressed_flatten/ --subset=dev --start_ind=0 --end_ind=-1 --batch_size=10
