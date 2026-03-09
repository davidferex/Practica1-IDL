#!/bin/bash
#SBATCH --job-name=distilbert_gpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=./logs/distilbert_job_train_gpu_batch32_fp16_%j.out
#SBATCH --err=./logs/distilbert_job_train_gpu_batch32_fp16_%j.err

set -euo pipefail

# Default value if not provided from outside
NUM_PROCESSES=${NUM_PROCESSES:-1}

echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

nvidia-smi

apptainer exec --nv --cleanenv \
  /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
  bash -lc "
    echo 'Inside container on ' \$(hostname) ' at ' \$(date)
    time accelerate launch --config_file config_gpubase.yaml ${HOME}/distilbert/distilbert_train_gpu_batch32_fp16.py
  "