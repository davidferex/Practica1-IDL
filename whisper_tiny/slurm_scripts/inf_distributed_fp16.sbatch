#!/bin/bash
#SBATCH --job-name=inf_distributed_fp16
#SBATCH --partition=atlasv2_mia_gpu02_4t4
#SBATCH --gres=gpu:2
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12GB
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

nvidia-smi

# Puerto único para evitar colisiones entre jobs concurrentes
MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 10000) ))

apptainer exec --nv --cleanenv \
  /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
  bash -lc "
    echo 'Inside container on ' \$(hostname) ' at ' \$(date)
    mkdir -p \${HOME}/whisper_tiny/results
    accelerate launch --multi_gpu --main_process_port ${MASTER_PORT} --num_machines=1 --num_processes=2 \${HOME}/whisper_tiny/accelerate_scripts/whisper-tiny-inf_distributed_fp16.py
  "
