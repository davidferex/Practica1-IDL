#!/bin/bash
#SBATCH --job-name=inf_cpu_batch32
#SBATCH --partition=atlasv2_mia_cpu01
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12GB
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

apptainer exec --cleanenv \
  /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
  bash -lc "
    echo 'Inside container on ' \$(hostname) ' at ' \$(date)
    mkdir -p \${HOME}/whisper_tiny/results
    accelerate launch --cpu --num_machines=1 --num_processes=1 \${HOME}/whisper_tiny/accelerate_scripts/whisper-tiny-inf_cpu_batch32.py
  "
