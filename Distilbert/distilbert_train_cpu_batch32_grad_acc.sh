#!/bin/bash
set -euo pipefail


# Default partition (single GPU)
PARTITION="atlasv2_mia_cpu01"


echo "Submitting to partition=${PARTITION}"

sbatch \
  --partition="${PARTITION}" \
  distilbert_train_cpu_batch32_grad_acc.sbatch

