#!/bin/bash
set -euo pipefail

NUM_PROCESSES="${NUM_GPUs:-1}"

# Default partition (single GPU)
PARTITION_DEFAULT="atlasv2_mia_gpu01_1t4"
# Multi-GPU partition
PARTITION_MULTI="atlasv2_mia_gpu02_4t4"

if [ "${NUM_PROCESSES}" -gt 1 ]; then
  PARTITION="${PARTITION_MULTI}"

  echo "Submitting with NUM_GPUs=${NUM_PROCESSES} to partition=${PARTITION}"

  sbatch \
    --partition="${PARTITION}" \
    --gres="gpu:${NUM_PROCESSES}" \
    --export=ALL,NUM_PROCESSES="${NUM_PROCESSES}" \
    distilbert_train_mgpu_batch16_fp32_grad_acc.sbatch
else
  PARTITION="${PARTITION_DEFAULT}"

  echo "Submitting with NUM_GPUs=${NUM_PROCESSES} to partition=${PARTITION}"

  sbatch \
    --partition="${PARTITION}" \
    --gres="gpu:${NUM_PROCESSES}" \
    --export=ALL,NUM_PROCESSES="${NUM_PROCESSES}" \
    distilbert_train_gpu_batch16_fp32_grad_acc.sbatch
fi

