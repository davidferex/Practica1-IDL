# whisper-tiny-inf_cpu_batch16.py
# Inferencia en CPU, FP32, batch=16.
# Estudia el efecto del tamaño de batch en throughput y RAM respecto al batch=1.

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

config_name = "inf_cpu_batch16"


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
        output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
        print(output)
        trace_path = os.path.join(_RESULTS_DIR, f"trace_{p.step_num}_{config_name}.json")
        p.export_chrome_trace(trace_path)
        print(f"Trace guardada en: {trace_path}", flush=True)

    profile_kwargs = ProfileKwargs(
        activities=["cpu"],
        record_shapes=True,
        schedule_option={"wait": 0, "warmup": 0, "active": 1, "repeat": 1},
        on_trace_ready=trace_handler,
    )
    accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])

    # Modelo y procesador
    model_name = "openai/whisper-tiny"
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)

    batch_size = 16
    num_batches = 16  # 256 total samples

    dataset = RandomAudioDataset(batch_size * num_batches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = accelerator.prepare(model)
    model.eval()

    # Memoria inicial
    cpu_rss_start = get_rss_bytes()
    cpu_rss_peak = cpu_rss_start or 0
    memory_log = []

    print(f"Config: {config_name}  batch_size={batch_size}  device=cpu", flush=True)
    print("Warming up CPU...", flush=True)
    warmup_features = torch.randn(1, 80, 3000)
    with torch.no_grad():
        _ = model.generate(warmup_features, max_new_tokens=25)

    # Bucle de inferencia con profiling
    print(f"=== Inference: {config_name}  batch_size={batch_size} ===", flush=True)
    total_samples = 0
    t0 = time.time()

    with accelerator.profile() as prof:
        with torch.no_grad():
            for batch_idx, features in enumerate(dataloader):
                features = features.to(accelerator.device)
                generated_ids = model.generate(features, max_new_tokens=25)
                decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)

                total_samples += features.size(0)
                rss = get_rss_bytes()
                if rss and rss > cpu_rss_peak:
                    cpu_rss_peak = rss

                memory_log.append({
                    "batch_idx": batch_idx,
                    "memory_cpu_mb": rss / 1024 ** 2 if rss else None,
                    "batch_size": features.size(0),
                    "sample_output": repr(decoded[0]) if decoded else "",
                })
                print(f"Batch {batch_idx}: size={features.size(0)}  "
                      f"CPU={rss / 1024**2:.1f} MB", flush=True)
                prof.step()

    t1 = time.time()
    elapsed = t1 - t0
    cpu_rss_end = get_rss_bytes()

    print(f"\nDone. config={config_name}  time={elapsed:.3f}s  "
          f"samples={total_samples}", flush=True)
    print(f"Throughput: {total_samples / elapsed:.2f} samples/s", flush=True)
    print(f"CPU RSS: start={format_bytes(cpu_rss_start)} "
          f"end={format_bytes(cpu_rss_end)} "
          f"peak~={format_bytes(cpu_rss_peak)}", flush=True)

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
                "batch_size": batch_size,
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

    print("\n--- CPU Profiling summary ---", flush=True)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15),
          flush=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
