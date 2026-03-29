"""
generate_plots.py
=================
Generates inference result plots reading all data from:
  - report_data/00_SUMMARY.csv   (throughput, VRAM, elapsed, CPU RSS)
  - results/memory_<cfg>.csv     (per-batch memory evolution)

Plots produced:
  throughput_all.png
  memory_all.png
  tp_cpu.png / mem_cpu.png
  tp_gpu_batches.png / mem_gpu_batches.png
  tp_gpu_advanced.png / mem_gpu_advanced.png
  tp_distributed.png / mem_distributed.png

Usage:
    python generate_plots.py

Output: whisper_tiny/results/plots/
"""

import csv
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
summary_csv = str(SCRIPT_DIR / "report_data" / "00_SUMMARY.csv")
results_dir = str(SCRIPT_DIR / "results")
plots_dir = str(SCRIPT_DIR / "results" / "plots")

os.makedirs(plots_dir, exist_ok=True)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("  [skip] matplotlib / numpy not available")
    raise SystemExit(0)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ── Load summary CSV ──────────────────────────────────────────────────────────
rows = []
with open(summary_csv) as f:
    for row in csv.DictReader(f):
        rows.append({k: v.strip('"') for k, v in row.items()})


def flt(s) -> float:
    try:
        return float(str(s).strip('"'))
    except:
        return 0.0


GROUP_COLOR = {
    "1_cpu_baseline": "#4E8EC4",
    "2_gpu_fp32": "#E07B22",
    "3_gpu_fp16": "#2AAA5B",
    "4_gpu_advanced": "#9254C8",
    "5_distributed": "#C0392B",
}

GROUP_LABEL = {
    "1_cpu_baseline": "CPU Baseline",
    "2_gpu_fp32": "GPU FP32",
    "3_gpu_fp16": "GPU FP16",
    "4_gpu_advanced": "GPU Advanced Techniques",
    "5_distributed": "Distributed (Multi-GPU)",
}


def glabel(g: str) -> str:
    return GROUP_LABEL.get(g, g.replace("_", " ").title())


DISPLAY_NAME = {
    "inf_cpu": "CPU FP32  b=1",
    "inf_cpu_batch16": "CPU FP32  b=16",
    "inf_cpu_batch32": "CPU FP32  b=32",
    "inf_gpu": "GPU FP32  b=1",
    "inf_gpu_batch4": "GPU FP32  b=4",
    "inf_gpu_batch8": "GPU FP32  b=8",
    "inf_gpu_batch16": "GPU FP32  b=16",
    "inf_gpu_batch32": "GPU FP32  b=32",
    "inf_gpu_fp16": "GPU FP16  b=16",
    "inf_gpu_compile": "GPU FP16 + compile",
    "inf_bigmodel": "Big Model (auto-map)",
    "inf_distributed": "Distributed FP32  2×GPU",
    "inf_distributed_fp16": "Distributed FP16  2×GPU",
}


def dname(cfg: str) -> str:
    return DISPLAY_NAME.get(cfg, cfg)


ok_rows = [r for r in rows if flt(r.get("throughput_samples_s", "0")) > 0]
cfg_map = {r["config"]: r for r in ok_rows}
saved: list[str] = []


def save(fig: plt.Figure, name: str) -> None:
    p = os.path.join(plots_dir, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    saved.append(name)
    print(f"  ✓  {name}")


# ── Group definitions ─────────────────────────────────────────────────────────
GROUPS = {
    "cpu": {
        "cfgs": ["inf_cpu", "inf_cpu_batch16", "inf_cpu_batch32"],
        "title": "CPU Experiments",
        "color": "#4E8EC4",
    },
    "gpu_batches": {
        "cfgs": [
            "inf_gpu",
            "inf_gpu_batch4",
            "inf_gpu_batch8",
            "inf_gpu_batch16",
            "inf_gpu_batch32",
        ],
        "title": "GPU FP32 — Batch Size Study",
        "color": "#E07B22",
    },
    "gpu_advanced": {
        "cfgs": ["inf_gpu_fp16", "inf_gpu_compile", "inf_bigmodel"],
        "title": "GPU Advanced Optimizations",
        "color": "#9254C8",
    },
    "distributed": {
        "cfgs": ["inf_distributed", "inf_distributed_fp16"],
        "title": "Distributed / Multi-GPU",
        "color": "#C0392B",
    },
}


def plot_group_throughput(group_key: str) -> None:
    g = GROUPS[group_key]
    cfgs = [c for c in g["cfgs"] if c in cfg_map]
    labels = [dname(c) for c in cfgs]
    vals = [flt(cfg_map[c]["throughput_samples_s"]) for c in cfgs]
    if not cfgs:
        return
    fig, ax = plt.subplots(figsize=(max(5, len(cfgs) * 1.2), 4))
    bars = ax.bar(
        range(len(labels)), vals, color=g["color"], width=0.6, edgecolor="white"
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Throughput (samples / s)")
    ax.set_title(f"Throughput — {g['title']}")
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    plt.tight_layout()
    save(fig, f"tp_{group_key}.png")


def plot_group_memory(group_key: str) -> None:
    g = GROUPS[group_key]
    cfgs = [c for c in g["cfgs"] if c in cfg_map]
    if not cfgs:
        return
    labels = [dname(c) for c in cfgs]
    cpu_rss = [flt(cfg_map[c].get("cpu_rss_peak_mb", "0")) for c in cfgs]
    gpu_vram = [flt(cfg_map[c].get("gpu_vram_peak_mb", "0")) for c in cfgs]
    has_gpu = any(v > 0 for v in gpu_vram)

    x = np.arange(len(labels))
    w = 0.38 if has_gpu else 0.55

    fig, ax = plt.subplots(figsize=(max(5, len(cfgs) * 1.2), 4))

    if has_gpu:
        bars_cpu = ax.bar(
            x - w / 2,
            cpu_rss,
            width=w,
            color="#4E8EC4",
            label="CPU RSS peak",
            edgecolor="white",
            linewidth=0.5,
        )
        bars_vram = ax.bar(
            x + w / 2,
            gpu_vram,
            width=w,
            color=g["color"],
            label="GPU VRAM peak",
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )
        for bar, v in zip(bars_cpu, cpu_rss):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#1B5F99",
                )
        for bar, v in zip(bars_vram, gpu_vram):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#555",
                )
        ax.legend(fontsize=8)
    else:
        bars = ax.bar(x, cpu_rss, width=w, color="#4E8EC4", edgecolor="white")
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=3)

    ax.set_xticks(x.tolist())
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Memory peak (MB)")
    ax.set_title(f"Memory — {g['title']}")
    plt.tight_layout()
    save(fig, f"mem_{group_key}.png")


# ── Per-group plots ───────────────────────────────────────────────────────────
for gk in GROUPS:
    plot_group_throughput(gk)
    plot_group_memory(gk)

# ── 19 · Throughput — all experiments, vertical bars, Spanish titles ──────────
ALL_CFGS_ORDERED = [
    "inf_cpu",
    "inf_cpu_batch16",
    "inf_cpu_batch32",
    "inf_gpu",
    "inf_gpu_batch4",
    "inf_gpu_batch8",
    "inf_gpu_batch16",
    "inf_gpu_batch32",
    "inf_gpu_fp16",
    "inf_gpu_compile",
    "inf_bigmodel",
    "inf_distributed",
    "inf_distributed_fp16",
]
all_rows = [cfg_map[c] for c in ALL_CFGS_ORDERED if c in cfg_map]

if all_rows:
    labels_all = [dname(r["config"]) for r in all_rows]
    vals_all = [flt(r["throughput_samples_s"]) for r in all_rows]
    colors_all = [GROUP_COLOR.get(r["group"], "#888") for r in all_rows]
    cpu_rss_all = [flt(r.get("cpu_rss_peak_mb", "0")) for r in all_rows]
    gpu_vram_all = [flt(r.get("gpu_vram_peak_mb", "0")) for r in all_rows]
    grp_colors = [GROUP_COLOR.get(r["group"], "#888") for r in all_rows]

    fig, ax = plt.subplots(figsize=(13, 5))
    x19 = np.arange(len(labels_all))
    bars19 = ax.bar(
        x19, vals_all, color=colors_all, width=0.6, edgecolor="white", linewidth=0.5
    )
    ax.set_xticks(x19.tolist())
    ax.set_xticklabels(labels_all, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Rendimiento (muestras / s)")
    ax.set_xlabel("Experimento")
    ax.set_title("Rendimiento de Inferencia — Todos los Experimentos")
    ax.bar_label(bars19, fmt="%.2f", fontsize=7, padding=3)
    legend19 = [
        mpatches.Patch(color=c, label=glabel(g))
        for g, c in GROUP_COLOR.items()
        if any(r["group"] == g for r in all_rows)
    ]
    ax.legend(handles=legend19, fontsize=7)
    plt.tight_layout()
    save(fig, "throughput_all.png")

    # ── 20 · Memory — all experiments, vertical bars, Spanish titles ──────────
    n20 = len(labels_all)
    x20 = np.arange(n20)
    w20 = 0.38

    fig, ax = plt.subplots(figsize=(14, 5))
    bars_cpu20 = ax.bar(
        x20 - w20 / 2,
        cpu_rss_all,
        width=w20,
        color="#4E8EC4",
        label="Pico RAM CPU",
        edgecolor="white",
        linewidth=0.5,
    )
    bars_vram20 = ax.bar(
        x20 + w20 / 2,
        gpu_vram_all,
        width=w20,
        color=grp_colors,
        label="Pico VRAM GPU",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )

    for bar, v in zip(bars_cpu20, cpu_rss_all):
        if v > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#1B5F99",
            )
    for bar, v in zip(bars_vram20, gpu_vram_all):
        if v > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#555",
            )

    ax.set_xticks(x20.tolist())
    ax.set_xticklabels(labels_all, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Memoria pico (MB)")
    ax.set_xlabel("Experimento")
    ax.set_title("Uso de Memoria — Todos los Experimentos (RAM CPU vs VRAM GPU)")

    group_legend20 = [
        mpatches.Patch(color=c, label=glabel(g))
        for g, c in GROUP_COLOR.items()
        if any(r["group"] == g for r in all_rows)
    ]
    cpu_handle20 = mpatches.Patch(color="#4E8EC4", label="Pico RAM CPU (todos)")
    ax.legend(
        handles=[cpu_handle20] + group_legend20,
        fontsize=7,
        title="VRAM GPU por grupo",
        title_fontsize=7,
    )
    plt.tight_layout()
    save(fig, "memory_all.png")

print(f"\n  {len(saved)} plot(s) written to {plots_dir}/")
