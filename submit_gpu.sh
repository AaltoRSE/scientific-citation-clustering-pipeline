#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1 

module load mamba
module load model-huggingface
source activate env-gpu/

python3 -u src/main.py $1 --distance_threshold 0.05
