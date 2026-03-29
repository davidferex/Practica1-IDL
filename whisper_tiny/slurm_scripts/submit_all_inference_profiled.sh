#!/bin/bash
# Lanza los 13 experimentos de inferencia en el clúster Atlas.
# Uso: bash submit_all_inference_profiled.sh  (desde el directorio slurm_scripts/)

set -euo pipefail

# Cadena A — CPU
A1=$(sbatch inf_cpu.sbatch | awk '{print $NF}')
A2=$(sbatch --dependency=afterok:${A1} inf_cpu_batch16.sbatch | awk '{print $NF}')
A3=$(sbatch --dependency=afterok:${A2} inf_cpu_batch32.sbatch | awk '{print $NF}')

# Cadena B — 1 GPU
B1=$(sbatch inf_gpu.sbatch | awk '{print $NF}')
B2=$(sbatch --dependency=afterok:${B1} inf_gpu_batch4.sbatch | awk '{print $NF}')
B3=$(sbatch --dependency=afterok:${B2} inf_gpu_batch8.sbatch | awk '{print $NF}')
B4=$(sbatch --dependency=afterok:${B3} inf_gpu_batch16.sbatch | awk '{print $NF}')
B5=$(sbatch --dependency=afterok:${B4} inf_gpu_batch32.sbatch | awk '{print $NF}')
B6=$(sbatch --dependency=afterok:${B5} inf_gpu_fp16.sbatch | awk '{print $NF}')
B7=$(sbatch --dependency=afterok:${B6} inf_gpu_compile.sbatch | awk '{print $NF}')
B8=$(sbatch --dependency=afterok:${B7} inf_bigmodel.sbatch | awk '{print $NF}')

# Cadena C — 2 GPUs (distribuido)
C1=$(sbatch inf_distributed.sbatch | awk '{print $NF}')
C2=$(sbatch --dependency=afterok:${C1} inf_distributed_fp16.sbatch | awk '{print $NF}')

echo "Jobs lanzados:"
echo "  atlasv2_mia_cpu01:      inf_cpu=${A1}  inf_cpu_batch16=${A2}  inf_cpu_batch32=${A3}"
echo "  atlasv2_mia_gpu01_1t4:  inf_gpu=${B1}  inf_gpu_batch4=${B2}  inf_gpu_batch8=${B3}  inf_gpu_batch16=${B4}  inf_gpu_batch32=${B5}  inf_gpu_fp16=${B6}  inf_gpu_compile=${B7}  inf_bigmodel=${B8}"
echo "  atlasv2_mia_gpu02_4t4:  inf_distributed=${C1}  inf_distributed_fp16=${C2}"
