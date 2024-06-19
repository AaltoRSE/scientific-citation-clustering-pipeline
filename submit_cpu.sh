#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

module load mamba
module load model-huggingface
source activate env-cpu/

python3 -u src/main.py $1 --distance_threshold 0.05
