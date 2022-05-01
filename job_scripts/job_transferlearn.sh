#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

module purge
module load nvidia-hpc-sdk/21.7

source ~/env/bin/activate
cd ../slt
# python -m signjoey train configs/transfer_first.yaml
python -m signjoey train configs/transfer_second.yaml
deactivate

#8248371 - error with gloss vs no gloss
#8258721 - printed full model & state_dict
#8258722 - printed encoder, decoder & state_dict sizes - errored out
#8258725 - printed decoder & state_dict sizes
#8258728 - just running it with the gloss params deleted