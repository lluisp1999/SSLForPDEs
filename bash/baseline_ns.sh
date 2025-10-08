#!/bin/bash
#SBATCH -J baseline_ns
#SBATCH --partition=accelerated,accelerated-h100
#SBATCH -n 16
#SBATCH --mem=124G
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH -o outputs/%x-%j.out

set -euo pipefail

########################
# 1 Software stack     #
########################
echo "Purging modules and activating Conda environment..."
module purge
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ssl

python navier_stokes.py --logging-folder log --exp-name baseline_ns --data-root "pdearena/NavierStokes-2D-conditioned"