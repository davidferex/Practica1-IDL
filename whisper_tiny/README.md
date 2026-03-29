# Whisper Tiny

## Estructura del directorio

```
whisper_tiny/
├── accelerate_scripts/      # Scripts Python de cada experimento
├── slurm_scripts/           # Scripts de lanzamiento en SLURM
├── results/                 # CSVs de resultados (memoria, kernels, trazas)
│   └── plots/               # Figuras generadas
├── logs/                    # Ficheros .out y .err de los jobs SLURM
├── generate_plots.py        # Genera todas las figuras desde los resultados
└── README.md
```

---

## Experimentos

Se ejecutan 13 experimentos organizados en 3 cadenas SLURM.

### Cadena A — CPU (cola `atlasv2_mia_cpu01`)

| Código            | Script Python                     | Descripción        |
| ----------------- | --------------------------------- | ------------------ |
| `inf_cpu`         | `whisper-tiny-inf_cpu.py`         | CPU FP32, batch=1  |
| `inf_cpu_batch16` | `whisper-tiny-inf_cpu_batch16.py` | CPU FP32, batch=16 |
| `inf_cpu_batch32` | `whisper-tiny-inf_cpu_batch32.py` | CPU FP32, batch=32 |

### Cadena B — 1 GPU T4 (cola `atlasv2_mia_gpu01_1t4`)

| Código            | Script Python                     | Descripción                     |
| ----------------- | --------------------------------- | ------------------------------- |
| `inf_gpu`         | `whisper-tiny-inf_gpu.py`         | GPU FP32, batch=1               |
| `inf_gpu_batch4`  | `whisper-tiny-inf_gpu_batch4.py`  | GPU FP32, batch=4               |
| `inf_gpu_batch8`  | `whisper-tiny-inf_gpu_batch8.py`  | GPU FP32, batch=8               |
| `inf_gpu_batch16` | `whisper-tiny-inf_gpu_batch16.py` | GPU FP32, batch=16              |
| `inf_gpu_batch32` | `whisper-tiny-inf_gpu_batch32.py` | GPU FP32, batch=32              |
| `inf_gpu_fp16`    | `whisper-tiny-inf_gpu_fp16.py`    | GPU FP16 AMP, batch=16          |
| `inf_gpu_compile` | `whisper-tiny-inf_gpu_compile.py` | GPU FP16 + torch.compile, b=16  |
| `inf_bigmodel`    | `whisper-tiny-inf_bigmodel.py`    | Big Model device_map=auto, b=16 |

### Cadena C — 2 GPU T4 (cola `atlasv2_mia_gpu02_4t4`)

| Código                 | Script Python                          | Descripción                     |
| ---------------------- | -------------------------------------- | ------------------------------- |
| `inf_distributed`      | `whisper-tiny-inf_distributed.py`      | Distributed FP32, batch=16/proc |
| `inf_distributed_fp16` | `whisper-tiny-inf_distributed_fp16.py` | Distributed FP16, batch=16/proc |

---

## Lanzamiento

### Todos los experimentos de una vez

Desde el directorio `slurm_scripts/`:

```bash
bash submit_all_inference_profiled.sh
```

Lanza los 13 jobs organizados en 3 cadenas independientes.

### Un experimento individual

Con los scripts slurm de cada uno de los experimentos

---

## Resultados y gráficas

Los resultados se escriben automáticamente en `results/` al terminar cada job.

Para generar las gráficas localmente:

```bash
python generate_plots.py
```

Las figuras se guardan en `results/plots/`.
