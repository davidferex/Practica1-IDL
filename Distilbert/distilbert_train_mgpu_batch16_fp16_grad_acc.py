#!/usr/bin/env python

import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from accelerate import Accelerator, ProfileKwargs
import platform

# -------------------------------
# Helpers memoria CPU
# -------------------------------
def get_rss_bytes():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        pass
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if platform.system().lower() == "linux":
            return int(usage.ru_maxrss * 1024)
        return int(usage.ru_maxrss)
    except Exception:
        return None


def format_bytes(n):
    if n is None:
        return "n/a"
    units = ["B","KiB","MiB","GiB","TiB"]
    x=float(n)
    for u in units:
        if x < 1024 or u==units[-1]:
            return f"{x:.2f} {u}"
        x/=1024


# -------------------------------
# Configuración
# -------------------------------
config_name = "mgpu_batch16_fp16_gradacc"
results_dir = os.path.join("results","distilbert")
os.makedirs(results_dir,exist_ok=True)

num_examples = 256
batch_size = 16
seq_length = 64
num_epochs = 1
learning_rate = 5e-5
num_labels = 2
gradient_accumulation_steps = 4


# -------------------------------
# Accelerate + profiler
# -------------------------------
def trace_handler(p):
    if accelerator.is_main_process:
        print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

        trace_path = os.path.join(
            results_dir,
            f"trace_{p.step_num}_{config_name}.json"
        )
        p.export_chrome_trace(trace_path)
        print(f"Trace guardada en: {trace_path}")


profile_kwargs = ProfileKwargs(
    activities=["cuda"],
    record_shapes=True,
    schedule_option={"wait": 0, "warmup": 0, "active": 2, "repeat": 1},
    on_trace_ready=trace_handler
)

accelerator = Accelerator(
    kwargs_handlers=[profile_kwargs],
    mixed_precision="fp16",
    gradient_accumulation_steps=gradient_accumulation_steps
)

device = accelerator.device
rank = accelerator.process_index
world = accelerator.num_processes


# -------------------------------
# Modelo
# -------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)

# -------------------------------
# Dataset
# -------------------------------
input_ids = torch.randint(0, tokenizer.vocab_size, (num_examples, seq_length))
attention_mask = torch.ones_like(input_ids)
labels = torch.randint(0, num_labels, (num_examples,))

dataset = TensorDataset(input_ids, attention_mask, labels)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

# -------------------------------
# Optimizer
# -------------------------------
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# -------------------------------
# Prepare
# -------------------------------
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
model.train()

# -------------------------------
# Memoria inicial
# -------------------------------
cpu_rss_start = get_rss_bytes()
cpu_rss_peak = cpu_rss_start

gpu_mem_start = torch.cuda.memory_allocated(device)
gpu_mem_peak = gpu_mem_start

memory_log = []
total_samples_local = 0

accelerator.wait_for_everyone()
torch.cuda.synchronize()

t0 = time.time()

# -------------------------------
# Training
# -------------------------------
with accelerator.profile() as prof:

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch_idx,(batch_input_ids,batch_attention_mask,batch_labels) in enumerate(dataloader):

            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            logits = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask
            ).logits

            loss = criterion(logits, batch_labels)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad()

            # memoria
            rss = get_rss_bytes()
            if rss and rss > cpu_rss_peak:
                cpu_rss_peak = rss

            gpu_mem = torch.cuda.memory_allocated(device)
            if gpu_mem > gpu_mem_peak:
                gpu_mem_peak = gpu_mem

            memory_log.append({
                "epoch":epoch,
                "batch_idx":batch_idx,
                "memory_cpu_mb": rss/1024**2 if rss else None,
                "memory_gpu_mb": gpu_mem/1024**2,
                "loss":loss.item()
            })

            epoch_loss += loss.item()
            total_samples_local += batch_input_ids.size(0)

            prof.step()

        # loss global
        avg_loss = torch.tensor(epoch_loss / len(dataloader), device=device)
        avg_loss = accelerator.reduce(avg_loss, reduction="mean").item()

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

# -------------------------------
# Throughput
# -------------------------------
accelerator.wait_for_everyone()
torch.cuda.synchronize()

t1 = time.time()

cpu_rss_end = get_rss_bytes()
gpu_mem_end = torch.cuda.memory_allocated(device)

# global samples
total_samples_global = accelerator.reduce(
    torch.tensor(total_samples_local, device=device),
    reduction="sum"
).item()

local_throughput = total_samples_local * gradient_accumulation_steps / (t1 - t0)
global_throughput = total_samples_global * gradient_accumulation_steps / (t1 - t0)

if accelerator.is_main_process:
    print(f"\nTiempo total: {t1-t0:.2f}s")
    print(f"Throughput LOCAL: {local_throughput:.2f} samples/s")
    print(f"Throughput GLOBAL: {global_throughput:.2f} samples/s")

# -------------------------------
# Guardar SOLO en main process
# -------------------------------
if accelerator.is_main_process:

    # profiling
    events = prof.key_averages()

    time_data = [{
        "kernel": e.key,
        "cpu_time_total_us": e.cpu_time_total,
        "self_cpu_time_total_us": e.self_cpu_time_total,
        "cuda_time_total_us": e.device_time_total,
        "self_cuda_time_total_us": e.self_device_time_total,
        "calls": e.count
    } for e in events]

    summary_time_path = os.path.join(
        results_dir,
        f"summary_time_{config_name}.csv"
    )

    with open(summary_time_path,"w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=time_data[0].keys())
        writer.writeheader()
        writer.writerows(time_data)

    print(f"Resumen profiling guardado en: {summary_time_path}")

    # memoria
    memory_path = os.path.join(
        results_dir,
        f"memory_{config_name}.csv"
    )

    with open(memory_path,"w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=memory_log[0].keys())
        writer.writeheader()
        writer.writerows(memory_log)

    print(f"Memoria guardada en: {memory_path}")

# -------------------------------
# Output ordenado por rank
# -------------------------------
accelerator.wait_for_everyone()

for r in range(world):
    if rank == r:
        print(f"\n[RANK {rank}/{world}]")
        print(f"CPU RSS: start={format_bytes(cpu_rss_start)} end={format_bytes(cpu_rss_end)} peak~={format_bytes(cpu_rss_peak)}")
        print(f"GPU Mem: start={format_bytes(gpu_mem_start)} end={format_bytes(gpu_mem_end)} peak~={format_bytes(gpu_mem_peak)}")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    accelerator.wait_for_everyone()

accelerator.end_training()