#!/bin/bash

##SBATCH -q ebrainccs
#SBATCH -p nvidia
#SBATCH -C 80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=71:59:59
#SBATCH --mem=32GB
#SBATCH --job-name=IT-icc

module purge

# Overlay the singularity
singularity shell --overlay ../../../../data_hallu.ext3 /share/apps/admin/singularity-images/centos-8.2.2004.sif

# Load the Conda module
source ~/.bashrc

# Activate your Conda environment
conda activate cipherword

# Lead to huggingface cache
export HF_HOME='/scratch/bc3194/huggingface_cache'

# Call python script with the model variable
python -u run.py \
    --subset="icc"