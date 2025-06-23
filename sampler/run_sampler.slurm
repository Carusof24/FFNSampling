#!/bin/bash
#SBATCH --job-name=hmc_sampler
#SBATCH --output=results/hmc_sampler_out.txt
#SBATCH --error=results/hmc_sampler_err.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

module purge
module load sw/miniconda3
module load cuda/12.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env

# Launch the sampler script directly
python3 -u sampler/pl_tgpu.py
