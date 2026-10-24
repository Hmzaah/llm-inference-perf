"""Command line interface: python -m llmperf <command> ...

Commands:
  list       show the built-in models and GPUs
  estimate   memory, prefill time, decode time and bottleneck for one setup
  sweep      decode throughput for a range of batch sizes
  quant      compare fp16 / int8 / int4 for one setup
  plots      write the figures in docs/figures/
"""

from __future__ import annotations

import argparse

from llmperf import calc
from llmperf.specs import load_gpus, load_models

PRECISIONS = list(calc.BYTES_PER_PARAM)


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows, strict=False)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(x).ljust(w) for x, w in zip(row, widths, strict=True)))


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="llama-3.1-8b", help="model key, see `list`")
    p.add_argument("--gpu", default="a100-80gb", help="GPU key, see `list`, or 'custom'")
    p.add_argument("--seq-len", type=int, default=2048, help="context length in tokens (prompt length)")
    p.add_argument("--precision", default="fp16", choices=PRECISIONS, help="weight format")
    p.add_argument("--kv-precision", default="fp16", choices=PRECISIONS, help="KV cache format")
    p.add_argument("--compute-dtype", default=None, choices=["fp16", "int8", "fp8"],
                   help="math used by the matmuls (default depends on --precision, see README)")
    p.add_argument("--compute-eff", type=float, default=calc.DEFAULT_COMPUTE_EFF,
                   help="fraction of peak FLOP/s reached (assumption)")
    p.add_argument("--memory-eff", type=float, default=calc.DEFAULT_MEMORY_EFF,
                   help="fraction of peak memory bandwidth reached (assumption)")
    p.add_argument("--usable-memory", type=float, default=calc.DEFAULT_USABLE_MEMORY_FRACTION,
                   help="fraction of GPU memory available for weights + KV cache")
    # user-defined GPU (used with --gpu custom)
    p.add_argument("--memory-gb", type=float, help="custom GPU: memory capacity in GB")
    p.add_argument("--bandwidth-gb-s", type=float, help="custom GPU: memory bandwidth in GB/s")
    p.add_argument("--fp16-tflops", type=float, help="custom GPU: dense FP16/BF16 TFLOPS")
    p.add_argument("--int8-tops", type=float, help="custom GPU: dense INT8 TOPS")
    p.add_argument("--fp8-tflops", type=float, help="custom GPU: dense FP8 TFLOPS")


def pick_model(key: str) -> calc.ModelSpec:
    models = load_models()
    if key not in models:
        raise SystemExit(f"unknown model {key!r}; choose from {', '.join(models)}")
    return models[key]


def pick_gpu(args: argparse.Namespace) -> calc.GPUSpec:
    if args.gpu == "custom":
        if not (args.memory_gb and args.bandwidth_gb_s and args.fp16_tflops):
            raise SystemExit("--gpu custom needs --memory-gb, --bandwidth-gb-s and --fp16-tflops")
        return calc.GPUSpec("custom GPU", args.memory_gb, args.bandwidth_gb_s, args.fp16_tflops,
                            args.int8_tops, args.fp8_tflops)
    gpus = load_gpus()
    if args.gpu not in gpus:
        raise SystemExit(f"unknown GPU {args.gpu!r}; choose from {', '.join(gpus)} or 'custom'")
    return gpus[args.gpu]


def phase_kwargs(args: argparse.Namespace, precision: str | None = None) -> dict:
    return {
        "precision": precision or args.precision,
        "kv_precision": args.kv_precision,
        "compute_dtype": args.compute_dtype,
        "compute_eff": args.compute_eff,
        "memory_eff": args.memory_eff,
    }


def cmd_list(args: argparse.Namespace) -> None:
    rows = [[k, m.name, f"{calc.count_params(m) / 1e9:.2f}B", m.n_layers, m.hidden_size,
             f"{m.n_heads}/{m.n_kv_heads}", m.head_dim] for k, m in load_models().items()]
    print_table(["model key", "name", "params", "layers", "hidden", "heads q/kv", "head_dim"], rows)
    print()
    rows = [[k, g.name, g.memory_gb, g.bandwidth_gb_s, g.fp16_tflops, g.int8_tops or "-", g.fp8_tflops or "-"]
            for k, g in load_gpus().items()]
    print_table(["gpu key", "name", "mem GB", "BW GB/s", "FP16 TF", "INT8 TOPS", "FP8 TF"], rows)
    print("\nAll compute numbers are DENSE (no sparsity). Sources: src/llmperf/data/*.json")


def cmd_estimate(args: argparse.Namespace) -> None:
    m, gpu = pick_model(args.model), pick_gpu(args)
    kw = phase_kwargs(args)
    w = calc.weight_bytes(m, args.precision)
    kv = calc.kv_cache_bytes(m, args.seq_len, args.batch, args.kv_precision)
    fits = calc.fits_in_memory(m, gpu, args.seq_len, args.batch, args.precision, args.kv_precision,
                               args.usable_memory)
    pre = calc.prefill(m, gpu, args.seq_len, args.batch, **kw)
    dec = calc.decode_step(m, gpu, args.seq_len, args.batch, **kw)
    dtype = args.compute_dtype or calc.DEFAULT_COMPUTE_DTYPE[args.precision]
    ridge = calc.ridge_point(gpu, dtype)

    print(f"{m.name} on {gpu.name} | batch {args.batch} | context {args.seq_len} | "
          f"weights {args.precision} | KV {args.kv_precision} | math {dtype}")
    print(f"efficiency: compute {args.compute_eff}, memory {args.memory_eff} (assumptions)\n")
    rows = [
        ["weights", f"{w / calc.GB:.2f} GB"],
        ["KV cache", f"{kv / calc.GB:.2f} GB"],
        ["fits in memory", f"{'yes' if fits else 'NO'} ({gpu.memory_gb:g} GB x {args.usable_memory} usable)"],
        ["max batch at this context", str(calc.max_batch(m, gpu, args.seq_len, args.precision,
                                                        args.kv_precision, args.usable_memory))],
        ["prefill time (= time to first token)", f"{pre.time * 1e3:.1f} ms ({pre.bottleneck})"],
        ["decode time per step", f"{dec.time * 1e3:.2f} ms ({dec.bottleneck})"],
        ["decode tokens/s (whole batch)", f"{args.batch / dec.time:.1f}"],
        ["decode tokens/s per sequence", f"{1 / dec.time:.1f}"],
        ["prefill intensity", f"{pre.arithmetic_intensity:.1f} FLOPs/byte"],
        ["decode intensity", f"{dec.arithmetic_intensity:.1f} FLOPs/byte"],
        ["GPU ridge point", f"{ridge:.1f} FLOPs/byte"],
    ]
    print_table(["metric", "value"], rows)


def cmd_sweep(args: argparse.Namespace) -> None:
    m, gpu = pick_model(args.model), pick_gpu(args)
    batches = [int(b) for b in args.batches.split(",")]
    rows = calc.batch_sweep(m, gpu, args.seq_len, batches, usable_fraction=args.usable_memory, **phase_kwargs(args))
    limit = calc.max_batch(m, gpu, args.seq_len, args.precision, args.kv_precision, args.usable_memory)
    print(f"{m.name} on {gpu.name} | context {args.seq_len} | weights {args.precision} | max batch {limit}\n")
    print_table(["batch", "step ms", "tokens/s", "bottleneck", "fits in memory"],
                [[r["batch"], f"{r['step_ms']:.2f}", f"{r['tokens_per_s']:.0f}", r["bottleneck"],
                  "yes" if r["fits"] else "NO"] for r in rows])


def cmd_quant(args: argparse.Namespace) -> None:
    m, gpu = pick_model(args.model), pick_gpu(args)
    print(f"{m.name} on {gpu.name} | batch {args.batch} | context {args.seq_len} | KV {args.kv_precision}\n")
    rows = []
    for p in ["fp16", "int8", "int4"]:
        try:
            dec = calc.decode_step(m, gpu, args.seq_len, args.batch, **phase_kwargs(args, p))
            pre = calc.prefill(m, gpu, args.seq_len, args.batch, **phase_kwargs(args, p))
        except ValueError as err:  # e.g. no INT8 peak published for this GPU
            rows.append([p, f"{calc.weight_bytes(m, p) / calc.GB:.2f}", "-", "-", "-", "-", f"n/a: {err}"])
            continue
        rows.append([
            p,
            f"{calc.weight_bytes(m, p) / calc.GB:.2f}",
            calc.max_batch(m, gpu, args.seq_len, p, args.kv_precision, args.usable_memory),
            f"{pre.time * 1e3:.1f}",
            f"{dec.time * 1e3:.2f}",
            f"{args.batch / dec.time:.1f}",
            dec.bottleneck,
        ])
    print_table(["weights", "weights GB", "max batch", "prefill ms", "decode ms/step", "tokens/s", "decode bound"],
                rows)


def cmd_plots(args: argparse.Namespace) -> None:
    from llmperf import plots  # imported here so the other commands don't need matplotlib

    for path in plots.make_all(args.out):
        print("wrote", path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llmperf", description="Roofline estimates for LLM inference")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show built-in models and GPUs").set_defaults(func=cmd_list)

    p = sub.add_parser("estimate", help="estimate one configuration")
    add_common_args(p)
    p.add_argument("--batch", type=int, default=1)
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("sweep", help="decode throughput vs batch size")
    add_common_args(p)
    p.add_argument("--batches", default="1,2,4,8,16,32,64,128,256,512")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("quant", help="compare fp16 / int8 / int4")
    add_common_args(p)
    p.add_argument("--batch", type=int, default=1)
    p.set_defaults(func=cmd_quant)

    p = sub.add_parser("plots", help="write figures")
    p.add_argument("--out", default="docs/figures")
    p.set_defaults(func=cmd_plots)

    args = parser.parse_args(argv)
    args.func(args)
