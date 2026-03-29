# whisper-tiny-inf_bigmodel.py
# Inferencia con Big Model Inference de Accelerate (device_map="auto"), batch=16.
# device_map coloca cada capa en el mejor dispositivo disponible automáticamente.
# El modelo NO se pasa por accelerator.prepare(); Accelerate solo se usa para profiling.

import csv
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator, ProfileKwargs
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# -------------------------------
# Directorio de resultados
# -------------------------------
_RESULTS_DIR = os.path.join(os.path.expanduser("~"), "whisper_tiny", "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

config_name = "inf_bigmodel"
DEVICE_MAP = os.environ.get("DEVICE_MAP", "auto")


# -------------------------------
# Helpers memoria
# -------------------------------
def get_rss_bytes():
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except Exception:
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            return None


def format_bytes(n):
    if n is None:
        return "N/A"
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.2f} KiB"
    return f"{n} B"


# -------------------------------
# Dataset sintético
# -------------------------------
class RandomAudioDataset(Dataset):
    """Espectrogramas log-mel aleatorios reproducibles (seed=42).
    Los datos se generan en __init__ para que todas las ejecuciones sean iguales."""

    def __init__(self, length: int, seed: int = 42):
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.data = torch.randn(length, 80, 3000, dtype=torch.float32, generator=rng)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# -------------------------------
# Inferencia
# -------------------------------
def main():
    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        print(output)
        trace_path = os.path.join(
            _RESULTS_DIR, f"trace_{p.step_num}_{config_name}.json"
        )
        p.export_chrome_trace(trace_path)
        print(f"Trace guardada en: {trace_path}", flush=True)

    profile_kwargs = ProfileKwargs(
        activities=["cpu", "cuda"],
        record_shapes=True,
        schedule_option={"wait": 0, "warmup": 0, "active": 2, "repeat": 1},
        on_trace_ready=trace_handler,
    )
    # Accelerate solo se usa para profiling; device_map gestiona la colocación del modelo
    accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

    # Carga con device_map: coloca automáticamente cada capa en el mejor dispositivo
    # "auto" → todo en GPU:0 (o CPU si no hay GPU)
    # No se llama accelerator.prepare(model) — device_map lo gestiona
    print(f"Loading model with device_map='{DEVICE_MAP}'...", flush=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny",
        device_map=DEVICE_MAP,
    )
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model.eval()

    # Muestra cómo se distribuyeron las capas
    print("\n--- Device Map (how layers were placed) ---", flush=True)
    if hasattr(model, "hf_device_map"):
        for layer, dev in model.hf_device_map.items():
            print(f"  {layer:<55} → {dev}", flush=True)
    print("-------------------------------------------\n", flush=True)

    # Con device_map los tensores viven en el dispositivo de cada capa; usamos cuda:0 para tracking
    primary_device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    batch_size = 16
    num_batches = 16  # 256 total samples

    dataset = RandomAudioDataset(batch_size * num_batches)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    # Warmup
    print("Warming up...", flush=True)
    # El input va al dispositivo de la primera capa (encoder conv1)
    first_device = next(iter(model.parameters())).device
    warmup_features = torch.randn(1, 80, 3000, dtype=torch.float32, device=first_device)
    with torch.no_grad():
        _ = model.generate(warmup_features, max_new_tokens=25)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Memoria inicial
    cpu_rss_start = get_rss_bytes()
    cpu_rss_peak = cpu_rss_start or 0
    gpu_mem_start = torch.cuda.memory_allocated(primary_device)
    torch.cuda.reset_peak_memory_stats(primary_device)
    num_gpus = torch.cuda.device_count()
    memory_log = []

    # Bucle de inferencia con profiling
    print(f"=== Inference: {config_name} (device_map='{DEVICE_MAP}') ===", flush=True)
    total_samples = 0
    t0 = time.time()

    with accelerator.profile() as prof:
        with torch.no_grad():
            for batch_idx, features in enumerate(dataloader):
                # El modelo enruta internamente los tensores entre dispositivos
                features = features.to(first_device)
                generated_ids = model.generate(features, max_new_tokens=25)
                decoded = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                total_samples += features.size(0)

                rss = get_rss_bytes()
                if rss and rss > cpu_rss_peak:
                    cpu_rss_peak = rss

                gpu_mem = torch.cuda.memory_allocated(primary_device)

                per_gpu = {
                    f"memory_gpu{i}_mb": torch.cuda.memory_allocated(i) / 1024**2
                    for i in range(num_gpus)
                }
                row = {
                    "batch_idx": batch_idx,
                    "memory_cpu_mb": rss / 1024**2 if rss else None,
                    **per_gpu,
                    "batch_size": features.size(0),
                    "device_map": DEVICE_MAP,
                }
                memory_log.append(row)
                print(
                    f"Batch {batch_idx}: {features.size(0)} samples, "
                    f"GPU0={per_gpu.get('memory_gpu0_mb', 0):.1f} MB",
                    flush=True,
                )
                prof.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    gpu_mem_peak = torch.cuda.max_memory_allocated(primary_device)
    elapsed = t1 - t0
    cpu_rss_end = get_rss_bytes()
    gpu_mem_end = torch.cuda.memory_allocated(primary_device)

    print(
        f"\nDone. config={config_name}  device_map={DEVICE_MAP}  "
        f"time={elapsed:.3f}s  samples={total_samples}",
        flush=True,
    )
    print(f"Throughput: {total_samples / elapsed:.2f} samples/s", flush=True)
    print(
        f"CPU RSS: start={format_bytes(cpu_rss_start)} "
        f"end={format_bytes(cpu_rss_end)} "
        f"peak~={format_bytes(cpu_rss_peak)}",
        flush=True,
    )
    for i in range(num_gpus):
        alloc = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        print(
            f"GPU {i} Mem: allocated={format_bytes(alloc)}  "
            f"reserved={format_bytes(reserved)}",
            flush=True,
        )

    # Guardar profiling
    events = prof.key_averages()
    if events:
        time_data = [
            {
                "kernel": e.key,
                "cpu_time_total_us": e.cpu_time_total,
                "self_cpu_time_total_us": e.self_cpu_time_total,
                "cuda_time_total_us": e.device_time_total,
                "self_cuda_time_total_us": e.self_device_time_total,
                "calls": e.count,
            }
            for e in events
        ]
        path = os.path.join(_RESULTS_DIR, f"summary_time_{config_name}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=time_data[0].keys())
            writer.writeheader()
            writer.writerows(time_data)
        print(f"Resumen del profiling guardado en: {path}", flush=True)

    # Guardar memoria
    if memory_log:
        path = os.path.join(_RESULTS_DIR, f"memory_{config_name}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=memory_log[0].keys())
            writer.writeheader()
            writer.writerows(memory_log)
        print(f"Evolucion memoria guardada en: {path}", flush=True)

    print("\n--- CUDA Profiling summary ---", flush=True)
    print(
        prof.key_averages().table(sort_by="cuda_time_total", row_limit=15), flush=True
    )

    accelerator.end_training()


if __name__ == "__main__":
    main()
