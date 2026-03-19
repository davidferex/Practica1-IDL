#!/bin/bash
set -euo pipefail

# Default partition
PARTITION="atlasv2_mia_cpu01"

# Check argument
MODE=${1:-noipex}

if [[ "$MODE" == "ipex" ]]; then
    SBATCH_FILE="distilbert_train_cpu_batch16_grad_acc_ipex.sbatch"
    echo "Running WITH IPEX"
elif [[ "$MODE" == "noipex" ]]; then
    SBATCH_FILE="distilbert_train_cpu_batch16_grad_acc.sbatch"
    echo "Running WITHOUT IPEX"
else
    echo "Usage: $0 [ipex|noipex]"
    exit 1
fi

echo "Submitting to partition=${PARTITION}"

sbatch \
  --partition="${PARTITION}" \
  "${SBATCH_FILE}"
