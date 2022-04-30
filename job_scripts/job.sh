#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50GB
#SBATCH --time=10:00:00

source /home1/maiyaupp/slt/venv/bin/activate
cd ../trial

python asl_pos_trials.py