#!/bin/bash
#SBATCH --job-name=distilbert_cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --output=./logs/distilbert_job_train_cpu_batch16_grad_acc_ipex_%j.out
#SBATCH --err=./logs/distilbert_job_train_cpu_batch16_grad_acc_ipex_%j.err

set -euo pipefail


# Build accelerate arguments conditionally
echo "No-GPU mode"

apptainer exec --cleanenv \
  /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
  bash -lc "
    echo 'Inside container on ' \$(hostname) ' at ' \$(date)
    time accelerate launch --config_file config_ipexbase.yaml ${HOME}/distilbert/distilbert_train_cpu_batch16_grad_acc.py
  "

