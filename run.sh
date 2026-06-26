#!/bin/bash
#SBATCH --job-name=NCA.sh
#SBATCH --array=0-9
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/NCA_%A_%a.out
#SBATCH --error=logs/NCA_%A_%a.err


source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate wm

python3 -u run.py 
