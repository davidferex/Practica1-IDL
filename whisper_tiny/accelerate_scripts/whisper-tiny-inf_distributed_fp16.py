# whisper-tiny-inf_distributed_fp16.py
# Inferencia distribuida FP16 en 2 GPUs con Accelerate (DDP), batch=16 por proceso.
# Igual que inf_distributed pero con precisión mixta FP16.
# Permite comparar el efecto de FP16 dentro del contexto distribuido.

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

config_name = "inf_distributed_fp16"


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
        rank = accelerator.process_index
        trace_path = os.path.join(
            _RESULTS_DIR, f"trace_{p.step_num}_{config_name}_rank{rank}.json"
        )
        p.export_chrome_trace(trace_path)
        print(f"Trace guardada en: {trace_path}", flush=True)

    profile_kwargs = ProfileKwargs(
        activities=["cpu", "cuda"],
        record_shapes=True,
        schedule_option={"wait": 0, "warmup": 0, "active": 2, "repeat": 1},
        on_trace_ready=trace_handler,
    )
    accelerator = Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[profile_kwargs],
    )
    device = accelerator.device

    print(f"[Process {accelerator.process_index}] device={device}  "
          f"num_processes={accelerator.num_processes}  "
          f"mixed_precision={accelerator.state.mixed_precision}", flush=True)

    # Modelo y procesador
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

    batch_size = 16   # batch por proceso
    num_batches = 256 // (batch_size * accelerator.num_processes)

    total_samples = batch_size * num_batches * accelerator.num_processes
    dataset = RandomAudioDataset(total_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    # generate() no es un forward() de DDP — hay que usar el modelo sin envolver
    raw_model = accelerator.unwrap_model(model)

    # Warmup
    if accelerator.is_main_process:
        print("Warming up...", flush=True)
    warmup_features = torch.randn(1, 80, 3000, dtype=torch.float32, device=device)
    with torch.no_grad():
        _ = raw_model.generate(warmup_features, max_new_tokens=25)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    accelerator.wait_for_everyone()

    # Memoria inicial
    cpu_rss_start = get_rss_bytes()
    cpu_rss_peak = cpu_rss_start or 0
    gpu_mem_start = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    memory_log = []

    # Bucle de inferencia con profiling
    if accelerator.is_main_process:
        print(f"=== Inference: {config_name}  "
              f"(num_processes={accelerator.num_processes}) ===", flush=True)
    samples_this_process = 0
    t0 = time.time()

    with accelerator.profile() as prof:
        with torch.no_grad():
            for batch_idx, features in enumerate(dataloader):
                generated_ids = raw_model.generate(features, max_new_tokens=25)

                gathered_ids = accelerator.gather_for_metrics(generated_ids)

                if accelerator.is_main_process:
                    decoded = processor.batch_decode(
                        gathered_ids, skip_special_tokens=True
                    )

                samples_this_process += features.size(0)

                rss = get_rss_bytes()
                if rss and rss > cpu_rss_peak:
                    cpu_rss_peak = rss

                if accelerator.is_main_process:
                    memory_log.append({
                        "batch_idx": batch_idx,
                        "memory_cpu_mb": rss / 1024 ** 2 if rss else None,
                        "memory_gpu0_mb": torch.cuda.memory_allocated(0) / 1024 ** 2,
                        "memory_gpu1_mb": (
                            torch.cuda.memory_allocated(1) / 1024 ** 2
                            if torch.cuda.device_count() > 1 else 0
                        ),
                        "local_batch_size": features.size(0),
                        "global_batch_size": gathered_ids.size(0),
                        "num_processes": accelerator.num_processes,
                    })
                    print(
                        f"Batch {batch_idx}: local={features.size(0)}  "
                        f"global={gathered_ids.size(0)}  "
                        f"GPU0={torch.cuda.memory_allocated(0)/1024**2:.1f} MB",
                        flush=True,
                    )
                prof.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    accelerator.wait_for_everyone()
    t1 = time.time()
    gpu_mem_peak = torch.cuda.max_memory_allocated(device)
    elapsed = t1 - t0

    global_samples = samples_this_process * accelerator.num_processes
    cpu_rss_end = get_rss_bytes()

    if accelerator.is_main_process:
        print(f"\nDone. config={config_name}  "
              f"num_processes={accelerator.num_processes}  "
              f"time={elapsed:.3f}s  global_samples={global_samples}", flush=True)
        print(f"Throughput: {global_samples / elapsed:.2f} samples/s "
              f"(wall-clock, {accelerator.num_processes} GPUs)", flush=True)
        print(f"CPU RSS [main]: start={format_bytes(cpu_rss_start)} "
              f"end={format_bytes(cpu_rss_end)} "
              f"peak~={format_bytes(cpu_rss_peak)}", flush=True)
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            print(f"GPU {i} Mem: allocated={format_bytes(alloc)}  "
                  f"reserved={format_bytes(reserved)}", flush=True)

    # Guardar resultados (solo proceso principal)
    if accelerator.is_main_process:
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

        if memory_log:
            path = os.path.join(_RESULTS_DIR, f"memory_{config_name}.csv")
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=memory_log[0].keys())
                writer.writeheader()
                writer.writerows(memory_log)
            print(f"Evolucion memoria guardada en: {path}", flush=True)

        print("\n--- CUDA Profiling summary ---", flush=True)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15),
              flush=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
