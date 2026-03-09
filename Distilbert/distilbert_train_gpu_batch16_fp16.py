#!/usr/bin/env python
# distilbert_train_gpu_batch16_fp16.py
# Entrenamiento de DistilBERT en GPU usando FP16, batch de 16, con profiling de Accelerate.

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from accelerate import Accelerator, ProfileKwargs

# -------------------------------
# Configuración del experimento
# -------------------------------
config_name = "gpu_batch16_fp16"
results_dir = os.path.join("results", "distilbert")
os.makedirs(results_dir, exist_ok=True)

num_examples = 256
batch_size = 16
seq_length = 64
num_epochs = 1
learning_rate = 5e-5
num_labels = 2

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
# Inicialización de Accelerate con profiling y FP16
# -------------------------------
profile_kwargs = ProfileKwargs(activities=["cuda"], record_shapes=True)
accelerator = Accelerator(
    cpu=False,
    kwargs_handlers=[profile_kwargs],
    mixed_precision="fp16"
)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
device = accelerator.device
model.train()

# -------------------------------
# Bucle de entrenamiento con profiling
# -------------------------------
with accelerator.profile() as prof:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_input_ids, batch_attention_mask, batch_labels in dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits
            loss = criterion(logits, batch_labels)
            accelerator.backward(loss)
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss/len(dataloader):.4f}")

print("Entrenamiento completado.")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# -------------------------------
# Guardar resultados del profiling en CSV
# -------------------------------
events = prof.key_averages()
data_to_save = [{
    'kernel': e.key,
    'cpu_time_total_us': e.cpu_time_total,
    'self_cpu_time_total_us': e.self_cpu_time_total,
    'cpu_time_avg_us': e.cpu_time_total / e.count if e.count > 0 else 0,
    'calls_cpu': e.count,
    
    'cuda_time_total_us': e.device_time_total,
    'self_cuda_time_total_us': e.self_device_time_total,
    'cuda_time_avg_us': e.device_time_total / e.count if e.count > 0 else 0
} for e in events]

summary_path = os.path.join(results_dir, f"summary_{config_name}.csv")
try:
    with open(summary_path, mode='w', newline='') as f:
        fieldnames = data_to_save[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_to_save)
    print(f"Resumen del profiling guardado en: {summary_path}")
except Exception as e:
    print(f"Error al guardar CSV: {e}")

# -------------------------------
# Mostrar top 10 funciones por tiempo CUDA
# -------------------------------
print("Top 10 funciones por CUDA total:")
top_10 = sorted(data_to_save, key=lambda x: x['cuda_time_total_us'], reverse=True)[:10]
for entry in top_10:
    print(entry)