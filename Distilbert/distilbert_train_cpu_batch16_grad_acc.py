#!/usr/bin/env python
# distilbert_train_cpu_batch16_grad_acc.py
# Entrenamiento de DistilBERT en CPU usando gradient accumulation, batch de 16 (batch efectivo = 64),
# con profiling de Accelerate, tracking de memoria y throughput.

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
# Helpers de memoria CPU
# -------------------------------
def get_rss_bytes():
    """Current process RSS in bytes (best effort)."""
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
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} B"

# -------------------------------
# Configuración del experimento
# -------------------------------
config_name = "cpu_batch16_gradacc"
results_dir = os.path.join("results", "distilbert")
os.makedirs(results_dir, exist_ok=True)

num_examples = 256
batch_size = 16
seq_length = 64
num_epochs = 1
learning_rate = 5e-5
num_labels = 2
gradient_accumulation_steps = 4  # Batch efectivo = 16 * 4 = 64

# -------------------------------
# Modelo y tokenizer
# -------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=num_labels
)

# -------------------------------
# Dataset simulado
# -------------------------------
input_ids = torch.randint(0, tokenizer.vocab_size, (num_examples, seq_length))
attention_mask = torch.ones_like(input_ids)
labels = torch.randint(0, num_labels, (num_examples,))

dataset = TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------------------
# Optimizador y función de pérdida
# -------------------------------
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# -------------------------------
# Trace handler
# -------------------------------
def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
    print(output)
    trace_path = os.path.join(results_dir, f"trace_{p.step_num}_cpu_batch16_gradacc.json")
    p.export_chrome_trace(trace_path)
    print(f"Trace guardada en: {trace_path}")

# -------------------------------
# Inicialización de Accelerate con profiling y grad acc
# -------------------------------
profile_kwargs = ProfileKwargs(
    activities=["cpu"],
    record_shapes=True,
    on_trace_ready=trace_handler
)
accelerator = Accelerator(
    cpu=True,
    kwargs_handlers=[profile_kwargs],
    gradient_accumulation_steps=gradient_accumulation_steps
)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
device = accelerator.device
model.train()

# -------------------------------
# Tracking de memoria y throughput
# -------------------------------
cpu_rss_start = get_rss_bytes()
cpu_rss_peak = cpu_rss_start
memory_log = []
total_samples = 0
t0 = time.time()

# -------------------------------
# Bucle de entrenamiento con grad acc y profiling
# -------------------------------
with accelerator.profile() as prof:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (batch_input_ids, batch_attention_mask, batch_labels) in enumerate(dataloader):
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad() if accelerator.sync_gradients else None
            logits = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits
            loss = criterion(logits, batch_labels)
            loss.backward()

            if accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad()

            # Registro de memoria CPU por batch
            rss = get_rss_bytes()
            if rss is not None and (cpu_rss_peak is None or rss > cpu_rss_peak):
                cpu_rss_peak = rss
            memory_log.append({
                'epoch': epoch,
                'batch_idx': batch_idx,
                'memory_cpu_mb': rss / 1024**2 if rss else None,
                'loss': loss.item()
            })

            epoch_loss += loss.item()
            total_samples += batch_input_ids.size(0)

            prof.step()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

t1 = time.time()
cpu_rss_end = get_rss_bytes()
throughput = total_samples / (t1 - t0)

print(f"\nEntrenamiento completado en {t1-t0:.2f}s")
print(f"Throughput: {throughput:.2f} samples/s")
print(f"CPU RSS: start={format_bytes(cpu_rss_start)}  end={format_bytes(cpu_rss_end)}  peak~={format_bytes(cpu_rss_peak)}")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# -------------------------------
# Guardar resultados del profiling en CSV
# -------------------------------
events = prof.key_averages()
time_data_to_save = [{
    'kernel': e.key,
    'cpu_time_total_us': e.cpu_time_total,
    'self_cpu_time_total_us': e.self_cpu_time_total,
    'cpu_time_avg_us': e.cpu_time_total / e.count if e.count > 0 else 0,
    'calls_cpu': e.count
} for e in events]

summary_time_path = os.path.join(results_dir, f"summary_time_{config_name}.csv")
with open(summary_time_path, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=time_data_to_save[0].keys())
    writer.writeheader()
    writer.writerows(time_data_to_save)
print(f"Resumen del profiling guardado en: {summary_time_path}")

# -------------------------------
# Guardar evolución de memoria por batch
# -------------------------------
memory_path = os.path.join(results_dir, f"memory_cpu_{config_name}.csv")
with open(memory_path, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=memory_log[0].keys())
    writer.writeheader()
    writer.writerows(memory_log)
print(f"Evolución memoria CPU guardada en: {memory_path}")