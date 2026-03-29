#!/bin/bash
#SBATCH --job-name=inf_gpu_fp16
#SBATCH --partition=atlasv2_mia_gpu01_1t4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12GB
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

nvidia-smi

apptainer exec --nv --cleanenv \
  /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
  bash -lc "
    echo 'Inside container on ' \$(hostname) ' at ' \$(date)
    mkdir -p \${HOME}/whisper_tiny/results
    accelerate launch --num_machines=1 --num_processes=1 \${HOME}/whisper_tiny/accelerate_scripts/whisper-tiny-inf_gpu_fp16.py
  "
