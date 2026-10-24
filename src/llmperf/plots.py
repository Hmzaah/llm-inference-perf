"""Make the three figures in docs/figures/ (python -m llmperf plots)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window needed, just write PNG files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from llmperf import calc  # noqa: E402
from llmperf.specs import load_gpus, load_models  # noqa: E402

# Fixed colours for up to three series (blue, orange, aqua), plus ink colours.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def roofline_plot(out: Path) -> Path:
    """Roofline of an A100 with LLaMA 3.1 8B fp16 prefill and decode points."""
    m, gpu = load_models()["llama-3.1-8b"], load_gpus()["a100-80gb"]
    ce, me = calc.DEFAULT_COMPUTE_EFF, calc.DEFAULT_MEMORY_EFF
    peak, bw = gpu.peak_flops("fp16"), gpu.bandwidth

    ai = np.logspace(-0.5, 4.5, 400)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.loglog(ai, np.minimum(peak, ai * bw) / 1e12, color=INK, linewidth=2, label="hardware roof (datasheet)")
    ax.loglog(ai, np.minimum(peak * ce, ai * bw * me) / 1e12, color=INK_2, linewidth=2, linestyle="--",
              label=f"effective roof (compute x{ce}, memory x{me})")
    ridge = calc.ridge_point(gpu, "fp16")
    ax.axvline(ridge, color=GRID, linewidth=1.5)
    ax.annotate(f"ridge point\n{ridge:.0f} FLOPs/byte", (ridge, 3), color=INK_2, fontsize=9,
                xytext=(6, 0), textcoords="offset points")

    points = [
        ("prefill, 2048-token prompt", calc.prefill(m, gpu, 2048), ORANGE),
        ("decode, batch 1", calc.decode_step(m, gpu, 512, 1), BLUE),
        ("decode, batch 64", calc.decode_step(m, gpu, 512, 64), BLUE),
        ("decode, batch 256", calc.decode_step(m, gpu, 512, 256), BLUE),
    ]
    for label, r, color in points:
        achieved = r.flops / r.time / 1e12
        ax.loglog(r.arithmetic_intensity, achieved, "o", markersize=9, color=color,
                  markeredgecolor="white", markeredgewidth=2)
        ax.annotate(f"{label}\n{r.bottleneck}", (r.arithmetic_intensity, achieved), fontsize=8.5, color=INK,
                    xytext=(8, -18), textcoords="offset points")
    _style(ax, "Roofline: LLaMA 3.1 8B (fp16) on A100 80GB, decode context 512",
           "arithmetic intensity (FLOPs per byte moved)", "attainable TFLOP/s")
    ax.set_ylim(0.5, 600)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    path = out / "roofline.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def throughput_vs_batch_plot(out: Path) -> Path:
    """Decode tokens/s vs batch size for a short and a long context."""
    m, gpu = load_models()["llama-3.1-8b"], load_gpus()["a100-80gb"]
    batches = [2**i for i in range(0, 11)]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for ctx, color in [(256, BLUE), (2048, ORANGE)]:
        rows = calc.batch_sweep(m, gpu, ctx, batches)
        limit = calc.max_batch(m, gpu, ctx)
        tps = [r["tokens_per_s"] for r in rows]
        ax.semilogx(batches, tps, "-o", color=color, linewidth=2, markersize=6, base=2,
                    label=f"context {ctx} tokens (max batch {limit})")
        # mark the batches that do not fit in memory with hollow markers
        over = [(b, t) for b, t, r in zip(batches, tps, rows, strict=True) if not r["fits"]]
        if over:
            ax.plot(*zip(*over, strict=True), "o", markersize=9, markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=2)
    # compute ceiling: tokens/s if every step ran at the effective compute peak (context 256)
    ceiling = gpu.peak_flops("fp16") * calc.DEFAULT_COMPUTE_EFF / calc.flops_per_token(m, 256)
    ax.axhline(ceiling, color=INK_2, linestyle="--", linewidth=1.5)
    ax.annotate(f"compute roof at context 256: {ceiling:,.0f} tokens/s", (batches[-1], ceiling), color=INK_2,
                fontsize=9, xytext=(0, -14), textcoords="offset points", ha="right")
    _style(ax, "Decode throughput vs batch: LLaMA 3.1 8B (fp16) on A100 80GB",
           "batch size", "tokens/s (all sequences)")
    ax.set_ylim(0, ceiling * 1.4)
    ax.legend(frameon=False, loc="upper left", fontsize=9, title="hollow = KV cache does not fit", title_fontsize=8.5)
    fig.tight_layout()
    path = out / "throughput_vs_batch.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def memory_vs_seq_plot(out: Path) -> Path:
    """Weights + KV cache vs context length for fp16 / int8 / int4 weights, batch 8."""
    m = load_models()["llama-3.1-8b"]
    gpus = load_gpus()
    seq = np.arange(0, 32769, 512)
    batch = 8
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for precision, color in [("fp16", BLUE), ("int8", ORANGE), ("int4", AQUA)]:
        total = [(calc.weight_bytes(m, precision) + calc.kv_cache_bytes(m, int(s), batch)) / calc.GB for s in seq]
        ax.plot(seq, total, color=color, linewidth=2, label=f"{precision} weights")
    for key in ("rtx-4090", "a100-80gb"):
        g = gpus[key]
        ax.axhline(g.memory_gb, color=INK_2, linestyle="--", linewidth=1.2)
        ax.annotate(f"{g.name}: {g.memory_gb:g} GB", (0, g.memory_gb), color=INK_2, fontsize=9,
                    xytext=(4, 4), textcoords="offset points")
    _style(ax, f"Memory vs context length: LLaMA 3.1 8B, batch {batch}, fp16 KV cache",
           "context length (tokens)", "GB (weights + KV cache)")
    ax.set_xlim(0, seq[-1])
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    path = out / "memory_vs_seq_len.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def make_all(out_dir: str = "docs/figures") -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [roofline_plot(out), throughput_vs_batch_plot(out), memory_vs_seq_plot(out)]
