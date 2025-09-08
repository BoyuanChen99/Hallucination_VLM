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

# Run inside the container (read-only overlay), with GPU support and /scratch visible.
singularity exec \
  --nv \
  --bind /scratch \
  --overlay ../../../data_hallu.ext3:ro \
  /share/apps/admin/singularity-images/centos-8.2.2004.sif \
  bash -lc '
    # Load Conda and activate env (from your home, which is mounted in the container)
    source ~/.bashrc
    conda activate vlm
    cd Hallucination_VLM/scripts/experiments

    # Hugging Face cache (ensure this directory exists and is writable on the host)
    export HF_HOME="/scratch/bc3194/huggingface_cache"

    # Run your script
    python -u run.py --subset="icc"
  '
