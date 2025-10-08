#!/bin/bash

# Parameters
#SBATCH --constraint=volta32gb
#SBATCH --cpus-per-task=10
#SBATCH --error=/hkfs/home/project/hk-project-pai00052/st_ac148019/SSLForPDEs/log/baseline_ns/%j_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=baseline_ns
#SBATCH --mem=40GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/hkfs/home/project/hk-project-pai00052/st_ac148019/SSLForPDEs/log/baseline_ns/%j_0_log.out
#SBATCH --partition=PARTITION
#SBATCH --signal=USR2@120
#SBATCH --time=4320
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /hkfs/home/project/hk-project-pai00052/st_ac148019/SSLForPDEs/log/baseline_ns/%j_%t_log.out --error /hkfs/home/project/hk-project-pai00052/st_ac148019/SSLForPDEs/log/baseline_ns/%j_%t_log.err /hkfs/home/project/hk-project-pai00052/st_ac148019/miniforge3/envs/ssl/bin/python -u -m submitit.core._submit /hkfs/home/project/hk-project-pai00052/st_ac148019/SSLForPDEs/log/baseline_ns
