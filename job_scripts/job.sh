#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH --time=12:00:00

source ~/slt/venv/bin/activate
cd ../trial

python phoenix_video_converter.py