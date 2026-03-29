# generate_plots.py
# Lee los logs SLURM de logs/ para extraer las métricas reales
# (throughput, tiempo, pico de RAM/VRAM) y genera las gráficas.
# Misma lógica de extracción que collect_results.sh.

import csv
import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
logs_dir    = str(SCRIPT_DIR / "logs")
results_dir = str(SCRIPT_DIR / "results")
plots_dir   = str(SCRIPT_DIR / "results" / "plots")

os.makedirs(plots_dir, exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("  [skip] matplotlib / numpy no disponible")
    raise SystemExit(0)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Metadatos de cada experimento: config → (grupo, dispositivo, batch_size)
EXPERIMENT_META = {
    "inf_cpu":              ("1_cpu_baseline", "cpu",      1),
    "inf_cpu_batch16":      ("1_cpu_baseline", "cpu",     16),
    "inf_cpu_batch32":      ("1_cpu_baseline", "cpu",     32),
    "inf_gpu":              ("2_gpu_fp32",     "gpu",      1),
    "inf_gpu_batch4":       ("2_gpu_fp32",     "gpu",      4),
    "inf_gpu_batch8":       ("2_gpu_fp32",     "gpu",      8),
    "inf_gpu_batch16":      ("2_gpu_fp32",     "gpu",     16),
    "inf_gpu_batch32":      ("2_gpu_fp32",     "gpu",     32),
    "inf_gpu_fp16":         ("3_gpu_fp16",     "gpu",     16),
    "inf_gpu_compile":      ("4_gpu_advanced", "gpu",     16),
    "inf_bigmodel":         ("4_gpu_advanced", "gpu",     16),
    "inf_distributed":      ("5_distributed",  "multigpu",16),
    "inf_distributed_fp16": ("5_distributed",  "multigpu",16),
}

# ── Helpers de extracción (misma lógica que collect_results.sh) ───────────────

def _parse_to_mb(s: str) -> float | None:
    """Convierte '1.38 GiB', '182.74 MiB', '5 KiB', '128 B' a float MB."""
    m = re.match(r"([0-9.]+)\s*(GiB|MiB|KiB|B)$", s.strip())
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2)
    if unit == "GiB": return val * 1024
    if unit == "MiB": return val
    if unit == "KiB": return val / 1024
    return val / (1024 * 1024)


def _extract_peak_mb(line: str) -> float | None:
    """Extrae el valor 'peak~=X.XX GiB/MiB/KiB' de una línea de log."""
    m = re.search(r"peak~?=([0-9.]+ (?:GiB|MiB|KiB|B))", line)
    return _parse_to_mb(m.group(1)) if m else None


def _find_log(cfg: str) -> tuple[str, str] | None:
    """Devuelve (ruta, contenido) del log más reciente con 'Throughput:' para cfg."""
    if not os.path.isdir(logs_dir):
        return None
    best_path, best_jid, best_content = None, -1, ""
    for fname in os.listdir(logs_dir):
        if not (fname.startswith(f"{cfg}-") and fname.endswith(".out")):
            continue
        fpath = os.path.join(logs_dir, fname)
        try:
            content = open(fpath).read()
        except OSError:
            continue
        if "Throughput:" not in content:
            continue
        m = re.search(r"-(\d+)\.out$", fname)
        if m:
            jid = int(m.group(1))
            if jid > best_jid:
                best_jid, best_path, best_content = jid, fpath, content
    return (best_path, best_content) if best_path else None


def cargar_metricas(cfg: str) -> dict | None:
    """Extrae métricas del log SLURM igual que collect_results.sh."""
    result = _find_log(cfg)
    if result is None:
        return None
    _, content = result
    lines = content.splitlines()

    group, device, batch_size = EXPERIMENT_META.get(cfg, ("unknown", "unknown", 1))

    elapsed = throughput = total_samples = cpu_peak = gpu_peak = None
    num_processes = 1

    # Línea Done.: tiempo total y muestras
    for line in lines:
        if line.startswith("Done."):
            m = re.search(r"time=([0-9.]+)s", line)
            if m:
                elapsed = float(m.group(1))
            m = re.search(r"(?:global_)?samples=(\d+)", line)
            if m:
                total_samples = int(m.group(1))
            m = re.search(r"num_processes=(\d+)", line)
            if m:
                num_processes = int(m.group(1))
            break

    # Throughput real de pared (wall-clock)
    for line in lines:
        if line.startswith("Throughput:"):
            m = re.search(r"([0-9]+\.[0-9]+)", line)
            if m:
                throughput = float(m.group(1))
            break

    # Pico de RAM CPU
    for line in lines:
        if "CPU RSS" in line:
            cpu_peak = _extract_peak_mb(line)
            break

    # Pico de VRAM GPU
    # 1) Scripts de 1 GPU imprimen "GPU Mem: ... peak~=X"
    for line in lines:
        if line.startswith("GPU Mem:"):
            gpu_peak = _extract_peak_mb(line)
            break

    # 2) Distributed/bigmodel imprimen "GPU N Mem: allocated=X reserved=Y"
    #    Sumar los allocated y multiplicar por num_processes (rank-1 aparece como 0)
    if gpu_peak is None:
        total_alloc = 0.0
        for line in lines:
            m = re.match(r"GPU \d+ Mem: allocated=([0-9.]+ (?:GiB|MiB|KiB|B))", line)
            if m:
                v = _parse_to_mb(m.group(1))
                if v:
                    total_alloc += v
        if total_alloc > 0:
            gpu_peak = total_alloc * num_processes if num_processes > 1 else total_alloc

    return {
        "config":               cfg,
        "group":                group,
        "device":               device,
        "batch_size":           str(batch_size),
        "elapsed_s":            f"{elapsed:.3f}" if elapsed is not None else "",
        "throughput_samples_s": f"{throughput:.2f}" if throughput is not None else "",
        "total_samples":        str(total_samples) if total_samples is not None else "",
        "cpu_rss_peak_mb":      f"{cpu_peak:.1f}" if cpu_peak is not None else "",
        "gpu_vram_peak_mb":     f"{gpu_peak:.1f}" if gpu_peak is not None else "",
    }


# ── Carga de métricas ─────────────────────────────────────────────────────────
rows = []
for cfg in EXPERIMENT_META:
    row = cargar_metricas(cfg)
    if row:
        rows.append(row)

if not rows:
    print("  [aviso] No se encontraron logs en logs/. Ejecuta los experimentos primero.")
    raise SystemExit(0)


def flt(s) -> float | None:
    try:
        v = float(str(s).strip('"'))
        return v if v > 0 else None
    except Exception:
        return None


def load_csv(filename: str) -> list[dict]:
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


GROUP_COLOR = {
    "1_cpu_baseline": "#4E8EC4",
    "2_gpu_fp32":     "#E07B22",
    "3_gpu_fp16":     "#2AAA5B",
    "4_gpu_advanced": "#9254C8",
    "5_distributed":  "#C0392B",
}

GROUP_LABEL = {
    "1_cpu_baseline": "CPU Baseline",
    "2_gpu_fp32":     "GPU FP32",
    "3_gpu_fp16":     "GPU FP16",
    "4_gpu_advanced": "GPU Advanced Techniques",
    "5_distributed":  "Distributed (Multi-GPU)",
}

def glabel(g: str) -> str:
    return GROUP_LABEL.get(g, g.replace("_", " ").title())

DISPLAY_NAME = {
    "inf_cpu":              "CPU FP32  b=1",
    "inf_cpu_batch16":      "CPU FP32  b=16",
    "inf_cpu_batch32":      "CPU FP32  b=32",
    "inf_gpu":              "GPU FP32  b=1",
    "inf_gpu_batch4":       "GPU FP32  b=4",
    "inf_gpu_batch8":       "GPU FP32  b=8",
    "inf_gpu_batch16":      "GPU FP32  b=16",
    "inf_gpu_batch32":      "GPU FP32  b=32",
    "inf_gpu_fp16":         "GPU FP16  b=16",
    "inf_gpu_compile":      "GPU FP16 + compile",
    "inf_bigmodel":         "Big Model (auto-map)",
    "inf_distributed":      "Distributed FP32  2×GPU",
    "inf_distributed_fp16": "Distributed FP16  2×GPU",
}

def dname(cfg: str) -> str:
    return DISPLAY_NAME.get(cfg, cfg)

ok_rows = [r for r in rows if flt(r.get("throughput_samples_s"))]
cfg_map  = {r["config"]: r for r in ok_rows}
saved: list[str] = []

def save(fig, name: str) -> None:
    p = os.path.join(plots_dir, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    saved.append(name)
    print(f"  ✓  {name}")


# ── 01 · Throughput — todos los experimentos ──────────────────────────────────
if ok_rows:
    labels = [dname(r["config"]) for r in ok_rows]
    vals   = [flt(r["throughput_samples_s"]) for r in ok_rows]
    colors = [GROUP_COLOR.get(r["group"], "#888") for r in ok_rows]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.45)))
    bars = ax.barh(labels, vals, color=colors, height=0.65)
    ax.set_xlabel("Throughput (samples / s)")
    ax.set_title("Inference Throughput — All Experiments")
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=7)
    legend = [mpatches.Patch(color=c, label=glabel(g))
              for g, c in GROUP_COLOR.items()
              if any(r["group"] == g for r in ok_rows)]
    ax.legend(handles=legend, fontsize=7, loc="lower right")
    save(fig, "01_throughput_all.png")


# ── 02 · Speedup vs CPU baseline ─────────────────────────────────────────────
base = cfg_map.get("inf_cpu")
if base:
    base_tp = flt(base["throughput_samples_s"])
    cmp_rows = [r for r in ok_rows if r["config"] != "inf_cpu"]
    if cmp_rows and base_tp:
        labels   = [dname(r["config"]) for r in cmp_rows]
        speedups = [(flt(r["throughput_samples_s"]) or 0) / base_tp for r in cmp_rows]
        colors   = [GROUP_COLOR.get(r["group"], "#888") for r in cmp_rows]

        fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.45)))
        bars = ax.barh(labels, speedups, color=colors, height=0.65)
        ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Speedup vs CPU FP32 baseline (×)")
        ax.set_title("Speedup Relative to CPU FP32 Baseline")
        ax.bar_label(bars, fmt="%.1f×", padding=4, fontsize=7)
        save(fig, "02_speedup_vs_cpu.png")


# ── 03 · Estudio de batch size — throughput ───────────────────────────────────
BATCH_SERIES = {
    "CPU FP32": (["inf_cpu", "inf_cpu_batch16", "inf_cpu_batch32"], "#4E8EC4", "o"),
    "GPU FP32": (["inf_gpu", "inf_gpu_batch4", "inf_gpu_batch8",
                  "inf_gpu_batch16", "inf_gpu_batch32"],            "#E07B22", "s"),
}

has_batch = any(cfg in cfg_map for cfgs, *_ in BATCH_SERIES.values() for cfg in cfgs)
if has_batch:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, (cfgs, color, marker) in BATCH_SERIES.items():
        pts = [(flt(cfg_map[c]["batch_size"]), flt(cfg_map[c]["throughput_samples_s"]))
               for c in cfgs if c in cfg_map]
        if pts:
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=marker, label=label, color=color,
                    linewidth=1.8, markersize=7)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (samples / s)")
    ax.set_title("Throughput vs Batch Size")
    ax.set_xticks([1, 4, 8, 16, 32])
    ax.legend(fontsize=8)
    save(fig, "03_batch_throughput.png")


# ── 04 · Estudio de batch size — memoria ─────────────────────────────────────
if has_batch:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for label, (cfgs, color, marker) in BATCH_SERIES.items():
        pts = [(flt(cfg_map[c]["batch_size"]), flt(cfg_map[c].get("cpu_rss_peak_mb")))
               for c in cfgs if c in cfg_map and flt(cfg_map[c].get("cpu_rss_peak_mb"))]
        if pts:
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=marker, label=label, color=color,
                    linewidth=1.8, markersize=7)
    ax.set_xlabel("Batch size"); ax.set_ylabel("CPU RSS peak (MB)")
    ax.set_title("CPU Memory vs Batch Size"); ax.set_xticks([1, 4, 8, 16, 32])
    ax.legend(fontsize=8)

    ax = axes[1]
    for label, (cfgs, color, marker) in BATCH_SERIES.items():
        if label == "CPU FP32":
            continue
        pts = [(flt(cfg_map[c]["batch_size"]), flt(cfg_map[c].get("gpu_vram_peak_mb")))
               for c in cfgs if c in cfg_map and flt(cfg_map[c].get("gpu_vram_peak_mb"))]
        if pts:
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=marker, label=label, color=color,
                    linewidth=1.8, markersize=7)
    ax.set_xlabel("Batch size"); ax.set_ylabel("GPU VRAM peak (MB)")
    ax.set_title("GPU Memory vs Batch Size"); ax.set_xticks([1, 4, 8, 16, 32])
    ax.legend(fontsize=8)

    plt.tight_layout()
    save(fig, "04_batch_memory.png")


# ── 05 · Comparación de precisión FP32 / FP16 / FP16+compile (batch=16) ──────
PREC_CFGS = [
    ("FP32",         "inf_gpu_batch16", "#E07B22"),
    ("FP16",         "inf_gpu_fp16",    "#2AAA5B"),
    ("FP16+compile", "inf_gpu_compile", "#9254C8"),
]
prec_vals = [(lbl, flt(cfg_map.get(cfg, {}).get("throughput_samples_s")), col)
             for lbl, cfg, col in PREC_CFGS]
prec_vals = [(l, v, c) for l, v, c in prec_vals if v]

if prec_vals:
    labels = [d[0] for d in prec_vals]
    vals   = [d[1] for d in prec_vals]
    colors = [d[2] for d in prec_vals]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(x, vals, 0.5, color=colors)
    ax.set_ylabel("Throughput (samples / s)")
    ax.set_title("Precision Comparison — GPU batch=16 (FP32 / FP16 / FP16+compile)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    save(fig, "05_precision_gpu.png")


# ── 06 · Evolución de memoria durante inferencia — experimentos CPU ───────────
CPU_MEM_CFGS = ["inf_cpu", "inf_cpu_batch16", "inf_cpu_batch32"]
MEM_COLORS   = {"inf_cpu": "#4E8EC4", "inf_cpu_batch16": "#1B5F99", "inf_cpu_batch32": "#0A2F4A"}
MEM_LABELS   = {"inf_cpu": "batch=1",  "inf_cpu_batch16": "batch=16", "inf_cpu_batch32": "batch=32"}

cpu_mem_data = {c: load_csv(f"memory_{c}.csv") for c in CPU_MEM_CFGS}
cpu_mem_data = {c: v for c, v in cpu_mem_data.items() if v}

if cpu_mem_data:
    fig, ax = plt.subplots(figsize=(9, 4))
    for cfg, mem_rows in cpu_mem_data.items():
        try:
            xs = [int(r.get("batch_idx", i)) for i, r in enumerate(mem_rows)]
            ys = [float(r["memory_cpu_mb"]) for r in mem_rows]
            ax.plot(xs, ys, label=MEM_LABELS.get(cfg, cfg),
                    color=MEM_COLORS.get(cfg, "#888"), linewidth=1.8)
        except (ValueError, KeyError):
            pass
    ax.set_xlabel("Batch index"); ax.set_ylabel("CPU RSS (MB)")
    ax.set_title("CPU Memory During Inference")
    ax.legend(fontsize=8)
    save(fig, "06_memory_evolution_cpu.png")


# ── 07 · Top 10 kernels por self CPU time — experimentos CPU ──────────────────
for cfg in CPU_MEM_CFGS:
    timing = load_csv(f"summary_time_{cfg}.csv")
    if not timing:
        continue
    def _sct(r):
        try: return float(r.get("self_cpu_time_total_us") or 0)
        except: return 0.0
    top = sorted(timing, key=_sct, reverse=True)[:10]
    kernels  = [r.get("kernel", "?")[:45] for r in top]
    times_us = [_sct(r) for r in top]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(kernels[::-1], times_us[::-1], color="#4E8EC4", height=0.65)
    ax.set_xlabel("Self CPU time (µs)")
    ax.set_title(f"Top 10 Kernels by Self CPU Time — {dname(cfg)}")
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=7)
    save(fig, f"07_top_kernels_{cfg}.png")


# ── 08 · Top 10 kernels — GPU baseline ───────────────────────────────────────
for cfg in ["inf_gpu", "inf_gpu_fp16"]:
    timing = load_csv(f"summary_time_{cfg}.csv")
    if not timing:
        continue
    def _cuda(r):
        try: return float(r.get("cuda_time_total_us") or 0)
        except: return 0.0
    def _sct2(r):
        try: return float(r.get("self_cpu_time_total_us") or 0)
        except: return 0.0

    top = sorted(timing, key=_cuda, reverse=True)[:10]
    if any(_cuda(r) > 0 for r in top):
        times_us = [_cuda(r) for r in top]
        col, xlabel = "#E07B22", "CUDA time total (µs)"
    else:
        top      = sorted(timing, key=_sct2, reverse=True)[:10]
        times_us = [_sct2(r) for r in top]
        col, xlabel = "#4E8EC4", "Self CPU time (µs)"

    kernels = [r.get("kernel", "?")[:45] for r in top]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(kernels[::-1], times_us[::-1], color=col, height=0.65)
    ax.set_xlabel(xlabel)
    ax.set_title(f"Top 10 Kernels — {dname(cfg)}")
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=7)
    save(fig, f"08_top_kernels_{cfg}.png")


# ── 09 · Evolución VRAM GPU — estudio de batch size FP32 ─────────────────────
GPU_BS_SERIES = [
    ("inf_gpu",         1,  "#FFC080"),
    ("inf_gpu_batch4",  4,  "#F0A040"),
    ("inf_gpu_batch8",  8,  "#E07B22"),
    ("inf_gpu_batch16", 16, "#C05010"),
    ("inf_gpu_batch32", 32, "#803000"),
]
gpu_mem_evo = {}
for cfg, bs, col in GPU_BS_SERIES:
    mem_rows = load_csv(f"memory_{cfg}.csv")
    if mem_rows and any(r.get("memory_gpu_mb") for r in mem_rows):
        gpu_mem_evo[cfg] = (bs, col, mem_rows)

if gpu_mem_evo:
    fig, ax = plt.subplots(figsize=(9, 4))
    for cfg, (bs, col, mem_rows) in gpu_mem_evo.items():
        try:
            xs = [int(r.get("batch_idx", i)) for i, r in enumerate(mem_rows)]
            ys = [float(r["memory_gpu_mb"]) for r in mem_rows]
            ax.plot(xs, ys, label=f"batch={bs}", color=col, linewidth=1.8)
        except (ValueError, KeyError):
            pass
    ax.set_xlabel("Batch index"); ax.set_ylabel("GPU memory allocated (MB)")
    ax.set_title("GPU Memory During Inference — FP32 Batch-Size Study")
    ax.legend(fontsize=8)
    save(fig, "09_memory_evolution_gpu.png")


# ── 10 · Comparación de técnicas GPU a batch=16 (throughput + VRAM) ──────────
TECH_CFGS = [
    ("FP32",         "inf_gpu_batch16",      "2_gpu_fp32"),
    ("FP16",         "inf_gpu_fp16",         "3_gpu_fp16"),
    ("FP16+compile", "inf_gpu_compile",      "4_gpu_advanced"),
    ("BigModel",     "inf_bigmodel",         "4_gpu_advanced"),
    ("Distrib FP32", "inf_distributed",      "5_distributed"),
    ("Distrib FP16", "inf_distributed_fp16", "5_distributed"),
]
tech_rows = [
    (lbl, cfg_map[cfg], GROUP_COLOR.get(grp, "#888"))
    for lbl, cfg, grp in TECH_CFGS
    if cfg in cfg_map and flt(cfg_map[cfg].get("throughput_samples_s"))
]

if tech_rows:
    labels = [t[0] for t in tech_rows]
    tps    = [flt(t[1]["throughput_samples_s"]) for t in tech_rows]
    vrams  = [flt(t[1].get("gpu_vram_peak_mb")) for t in tech_rows]
    colors = [t[2] for t in tech_rows]
    x      = np.arange(len(labels))
    has_vram = any(v is not None for v in vrams)

    ncols = 2 if has_vram else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    bars1 = axes[0].bar(x, tps, color=colors, width=0.6)
    axes[0].set_ylabel("Throughput (samples / s)")
    axes[0].set_title("Throughput — GPU Techniques (batch=16)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].bar_label(bars1, fmt="%.1f", fontsize=8, padding=3)

    if has_vram:
        vram_vals = [v if v else 0 for v in vrams]
        bars2 = axes[1].bar(x, vram_vals, color=colors, width=0.6)
        axes[1].set_ylabel("GPU VRAM peak (MB)")
        axes[1].set_title("VRAM Peak — GPU Techniques (batch=16)")
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=20, ha="right")
        axes[1].bar_label(bars2, fmt="%.0f", fontsize=8, padding=3)

    plt.tight_layout()
    save(fig, "10_gpu_technique_comparison.png")


# ── 11 · Eficiencia de escalado distribuido ───────────────────────────────────
SCALE_PAIRS = [
    ("FP32", "inf_gpu_batch16", "inf_distributed"),
    ("FP16", "inf_gpu_fp16",    "inf_distributed_fp16"),
]
scale_data = []
for label, single_cfg, dist_cfg in SCALE_PAIRS:
    s = cfg_map.get(single_cfg)
    d = cfg_map.get(dist_cfg)
    if s and d:
        tp_s = flt(s.get("throughput_samples_s"))
        tp_d = flt(d.get("throughput_samples_s"))
        if tp_s and tp_d:
            scale_data.append((label, tp_s, tp_d, tp_d / tp_s * 100 / 2))

if scale_data:
    labels  = [r[0] for r in scale_data]
    tp_1gpu = [r[1] for r in scale_data]
    tp_2gpu = [r[2] for r in scale_data]
    effs    = [r[3] for r in scale_data]
    x = np.arange(len(labels)); w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(x - w/2, tp_1gpu, w, label="1×GPU (batch=16)",    color="#E07B22")
    ax1.bar(x + w/2, tp_2gpu, w, label="2×GPU DDP (16/proc)", color="#C0392B")
    ax1.set_ylabel("Throughput (samples / s)")
    ax1.set_title("Single vs Distributed Throughput")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.legend(fontsize=8)

    bars = ax2.bar(x, effs, color="#C0392B", width=0.4)
    ax2.axhline(100, color="grey", linestyle="--", linewidth=0.8, label="ideal (100 %)")
    ax2.set_ylabel("Scaling efficiency (%)")
    ax2.set_title("Scaling Efficiency — 2 GPUs (ideal = 100 %)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 130)
    ax2.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    save(fig, "11_distributed_scaling.png")


# ── 12 · Throughput — experimentos CPU ───────────────────────────────────────
CPU_TP_CFGS = ["inf_cpu", "inf_cpu_batch16", "inf_cpu_batch32"]
cpu_tp_rows = [(dname(c), flt(cfg_map[c]["throughput_samples_s"]))
               for c in CPU_TP_CFGS
               if c in cfg_map and flt(cfg_map[c].get("throughput_samples_s"))]
if cpu_tp_rows:
    labels, vals = zip(*cpu_tp_rows)
    colors = ["#4E8EC4", "#1B5F99", "#0A2F4A"][:len(labels)]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(labels)), vals, color=colors, width=0.55)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Throughput (samples / s)")
    ax.set_title("Throughput — CPU Experiments")
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    plt.tight_layout()
    save(fig, "12_throughput_cpu.png")


# ── 13 · Throughput — 1 GPU (básico vs avanzado) ─────────────────────────────
GPU_BASIC_CFGS = ["inf_gpu", "inf_gpu_batch4", "inf_gpu_batch8",
                  "inf_gpu_batch16", "inf_gpu_batch32", "inf_gpu_fp16"]
GPU_ADV_CFGS   = ["inf_gpu_compile", "inf_bigmodel"]
GPU_BASIC_COL  = "#E07B22"
GPU_ADV_COL    = "#9254C8"

gpu1_basic = [(dname(c), flt(cfg_map[c]["throughput_samples_s"]), GPU_BASIC_COL)
              for c in GPU_BASIC_CFGS
              if c in cfg_map and flt(cfg_map[c].get("throughput_samples_s"))]
gpu1_adv   = [(dname(c), flt(cfg_map[c]["throughput_samples_s"]), GPU_ADV_COL)
              for c in GPU_ADV_CFGS
              if c in cfg_map and flt(cfg_map[c].get("throughput_samples_s"))]
gpu1_all   = gpu1_basic + gpu1_adv

if gpu1_all:
    labels, vals, colors = zip(*gpu1_all)
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(labels)), vals, color=colors, width=0.6)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Throughput (samples / s)")
    ax.set_title("Throughput — 1-GPU Experiments")
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    ax.legend(handles=[
        mpatches.Patch(color=GPU_BASIC_COL, label="Basic (FP32/FP16 · batch size)"),
        mpatches.Patch(color=GPU_ADV_COL,   label="Advanced (compile · Big Model)"),
    ], fontsize=8)
    plt.tight_layout()
    save(fig, "13_throughput_1gpu.png")


# ── 14 · Throughput — Multi-GPU ──────────────────────────────────────────────
MGPU_CFGS = ["inf_distributed", "inf_distributed_fp16"]
mgpu_rows = [(dname(c), flt(cfg_map[c]["throughput_samples_s"]))
             for c in MGPU_CFGS
             if c in cfg_map and flt(cfg_map[c].get("throughput_samples_s"))]
if mgpu_rows:
    labels, vals = zip(*mgpu_rows)
    colors = ["#C0392B", "#922B21"][:len(labels)]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(labels)), vals, color=colors, width=0.45)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Throughput (samples / s)")
    ax.set_title("Throughput — Multi-GPU Experiments")
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    plt.tight_layout()
    save(fig, "14_throughput_mgpu.png")


# ── 15 · Memoria — experimentos CPU (CPU RSS) ─────────────────────────────────
cpu_mem_rows = [(dname(c), flt(cfg_map[c].get("cpu_rss_peak_mb")))
                for c in CPU_TP_CFGS
                if c in cfg_map and flt(cfg_map[c].get("cpu_rss_peak_mb"))]
if cpu_mem_rows:
    labels, vals = zip(*cpu_mem_rows)
    colors = ["#4E8EC4", "#1B5F99", "#0A2F4A"][:len(labels)]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(labels)), vals, color=colors, width=0.55)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("CPU RSS peak (MB)")
    ax.set_title("Memory Usage — CPU Experiments")
    ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=3)
    plt.tight_layout()
    save(fig, "15_memory_cpu.png")


# ── 16 · Memoria — 1 GPU (VRAM) ──────────────────────────────────────────────
gpu1_mem_basic = [(dname(c), flt(cfg_map[c].get("gpu_vram_peak_mb")), GPU_BASIC_COL)
                  for c in GPU_BASIC_CFGS
                  if c in cfg_map and flt(cfg_map[c].get("gpu_vram_peak_mb"))]
gpu1_mem_adv   = [(dname(c), flt(cfg_map[c].get("gpu_vram_peak_mb")), GPU_ADV_COL)
                  for c in GPU_ADV_CFGS
                  if c in cfg_map and flt(cfg_map[c].get("gpu_vram_peak_mb"))]
gpu1_mem_all   = gpu1_mem_basic + gpu1_mem_adv

if gpu1_mem_all:
    labels, vals, colors = zip(*gpu1_mem_all)
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(labels)), vals, color=colors, width=0.6)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("GPU VRAM peak (MB)")
    ax.set_title("Memory Usage — 1-GPU Experiments")
    ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=3)
    ax.legend(handles=[
        mpatches.Patch(color=GPU_BASIC_COL, label="Basic (FP32/FP16 · batch size)"),
        mpatches.Patch(color=GPU_ADV_COL,   label="Advanced (compile · Big Model)"),
    ], fontsize=8)
    plt.tight_layout()
    save(fig, "16_memory_1gpu.png")


# ── 17 · Memoria — Multi-GPU (VRAM) ──────────────────────────────────────────
mgpu_mem_rows = [(dname(c), flt(cfg_map[c].get("gpu_vram_peak_mb")))
                 for c in MGPU_CFGS
                 if c in cfg_map and flt(cfg_map[c].get("gpu_vram_peak_mb"))]
if mgpu_mem_rows:
    labels, vals = zip(*mgpu_mem_rows)
    colors = ["#C0392B", "#922B21"][:len(labels)]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(labels)), vals, color=colors, width=0.45)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("GPU VRAM peak (MB)")
    ax.set_title("Memory Usage — Multi-GPU Experiments")
    ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=3)
    plt.tight_layout()
    save(fig, "17_memory_mgpu.png")


# ── 18 · Memoria — todos los experimentos (CPU RSS + VRAM GPU) ────────────────
ALL_CFGS_ORDERED = [
    "inf_cpu", "inf_cpu_batch16", "inf_cpu_batch32",
    "inf_gpu", "inf_gpu_batch4", "inf_gpu_batch8", "inf_gpu_batch16", "inf_gpu_batch32",
    "inf_gpu_fp16", "inf_gpu_compile", "inf_bigmodel",
    "inf_distributed", "inf_distributed_fp16",
]
mem_all_rows = [cfg_map[c] for c in ALL_CFGS_ORDERED if c in cfg_map]

if mem_all_rows:
    labels_all   = [dname(r["config"]) for r in mem_all_rows]
    cpu_rss_all  = [flt(r.get("cpu_rss_peak_mb"))  or 0 for r in mem_all_rows]
    gpu_vram_all = [flt(r.get("gpu_vram_peak_mb")) or 0 for r in mem_all_rows]
    grp_colors   = [GROUP_COLOR.get(r["group"], "#888") for r in mem_all_rows]

    n = len(labels_all)
    y = np.arange(n)
    h = 0.38

    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.55)))
    bars_cpu  = ax.barh(y + h/2, cpu_rss_all,  height=h, color="#4E8EC4",
                        label="CPU RSS peak",  edgecolor="white", linewidth=0.5)
    bars_vram = ax.barh(y - h/2, gpu_vram_all, height=h, color=grp_colors,
                        label="GPU VRAM peak", edgecolor="white", linewidth=0.5, alpha=0.85)

    for bar, v in zip(bars_cpu, cpu_rss_all):
        if v > 0:
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                    f"{v:.0f}", va="center", fontsize=7, color="#1B5F99")
    for bar, v in zip(bars_vram, gpu_vram_all):
        if v > 0:
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                    f"{v:.0f}", va="center", fontsize=7, color="#555")

    ax.set_yticks(y)
    ax.set_yticklabels(labels_all, fontsize=8)
    ax.set_xlabel("Memory peak (MB)")
    ax.set_title("Memory Usage — All Experiments (CPU RSS vs GPU VRAM)", fontsize=10)
    group_legend = [mpatches.Patch(color=c, label=glabel(g))
                    for g, c in GROUP_COLOR.items()
                    if any(r["group"] == g for r in mem_all_rows)]
    cpu_handle = mpatches.Patch(color="#4E8EC4", label="CPU RSS peak (all)")
    ax.legend(handles=[cpu_handle] + group_legend, fontsize=7,
              loc="lower right", title="GPU VRAM by group")
    plt.tight_layout()
    save(fig, "18_memory_all.png")


# ── 19 · Rendimiento — todos los experimentos, barras verticales, español ─────
if ok_rows:
    labels19 = [dname(r["config"]) for r in ok_rows]
    vals19   = [flt(r["throughput_samples_s"]) for r in ok_rows]
    colors19 = [GROUP_COLOR.get(r["group"], "#888") for r in ok_rows]

    fig, ax = plt.subplots(figsize=(13, 5))
    x19 = np.arange(len(labels19))
    bars19 = ax.bar(x19, vals19, color=colors19, width=0.6, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x19)
    ax.set_xticklabels(labels19, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Rendimiento (muestras / s)")
    ax.set_xlabel("Experimento")
    ax.set_title("Rendimiento de Inferencia — Todos los Experimentos")
    ax.bar_label(bars19, fmt="%.2f", fontsize=7, padding=3)
    legend19 = [mpatches.Patch(color=c, label=glabel(g))
                for g, c in GROUP_COLOR.items()
                if any(r["group"] == g for r in ok_rows)]
    ax.legend(handles=legend19, fontsize=7)
    plt.tight_layout()
    save(fig, "19_throughput_all_vertical_es.png")


# ── 20 · Memoria — todos los experimentos, barras verticales, español ─────────
if mem_all_rows:
    n20 = len(labels_all)
    x20 = np.arange(n20)
    w20 = 0.38

    fig, ax = plt.subplots(figsize=(14, 5))
    bars_cpu20  = ax.bar(x20 - w20/2, cpu_rss_all,  width=w20, color="#4E8EC4",
                         label="Pico RAM CPU", edgecolor="white", linewidth=0.5)
    bars_vram20 = ax.bar(x20 + w20/2, gpu_vram_all, width=w20, color=grp_colors,
                         label="Pico VRAM GPU", edgecolor="white", linewidth=0.5, alpha=0.85)

    for bar, v in zip(bars_cpu20, cpu_rss_all):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, color="#1B5F99")
    for bar, v in zip(bars_vram20, gpu_vram_all):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, color="#555")

    ax.set_xticks(x20)
    ax.set_xticklabels(labels_all, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Memoria pico (MB)")
    ax.set_xlabel("Experimento")
    ax.set_title("Uso de Memoria — Todos los Experimentos (RAM CPU vs VRAM GPU)")
    group_legend20 = [mpatches.Patch(color=c, label=glabel(g))
                      for g, c in GROUP_COLOR.items()
                      if any(r["group"] == g for r in mem_all_rows)]
    cpu_handle20 = mpatches.Patch(color="#4E8EC4", label="Pico RAM CPU (todos)")
    ax.legend(handles=[cpu_handle20] + group_legend20, fontsize=7,
              title="VRAM GPU por grupo", title_fontsize=7)
    plt.tight_layout()
    save(fig, "20_memory_all_vertical_es.png")


print(f"\n  {len(saved)} plot(s) written to {plots_dir}/")
