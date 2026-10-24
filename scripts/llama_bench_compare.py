"""Compare your own llama-bench run with this tool's estimate.

1. Run llama.cpp's benchmark on your GPU and save JSON:
     llama-bench -m llama-2-7b.Q4_0.gguf -ngl 99 -o json > my_llama_bench.json
2. Run this script with the matching model key and your GPU:
     python scripts/llama_bench_compare.py my_llama_bench.json --model llama-2-7b --gpu rtx-4090
   or describe your GPU yourself:
     python scripts/llama_bench_compare.py my_llama_bench.json --model llama-2-7b --gpu custom \
         --memory-gb 8 --bandwidth-gb-s 272 --fp16-tflops 22 --int8-tops 88

The bytes per weight are taken from the file itself (model_size / model_n_params that
llama-bench reports), so any GGUF quantization works (Q4_0, Q4_K_M, Q8_0, ...).
"""

from __future__ import annotations

import argparse
import json

from llmperf import calc
from llmperf.specs import load_gpus, load_models


def load_runs(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:  # PowerShell '>' can write a BOM
        text = f.read().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]  # jsonl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bench_json", help="output of: llama-bench ... -o json")
    p.add_argument("--model", required=True, help="model key, e.g. llama-2-7b, llama-3.1-8b, qwen2.5-7b")
    p.add_argument("--gpu", required=True, help="GPU key or 'custom'")
    p.add_argument("--memory-gb", type=float)
    p.add_argument("--bandwidth-gb-s", type=float)
    p.add_argument("--fp16-tflops", type=float)
    p.add_argument("--int8-tops", type=float)
    p.add_argument("--compute-dtype", default="int8", choices=["int8", "fp16"],
                   help="math assumed for prompt processing (llama.cpp quantized kernels use int8)")
    p.add_argument("--compute-eff", type=float, default=calc.DEFAULT_COMPUTE_EFF)
    p.add_argument("--memory-eff", type=float, default=calc.DEFAULT_MEMORY_EFF)
    args = p.parse_args()

    model = load_models()[args.model]
    if args.gpu == "custom":
        gpu = calc.GPUSpec("custom GPU", args.memory_gb, args.bandwidth_gb_s, args.fp16_tflops, args.int8_tops)
    else:
        gpu = load_gpus()[args.gpu]

    runs = load_runs(args.bench_json)
    first = runs[0]
    n_params = first["model_n_params"]
    if abs(n_params - calc.count_params(model)) / n_params > 0.02:
        print(f"WARNING: file has {n_params:,} params but {model.name} has {calc.count_params(model):,}. "
              "Is --model the right one?")

    # Bytes per weight measured from the file (includes block scales, mixed tensor types).
    calc.BYTES_PER_PARAM["gguf"] = first["model_size"] / n_params
    calc.DEFAULT_COMPUTE_DTYPE["gguf"] = args.compute_dtype

    print(f"file: {first.get('model_filename', '?')} ({first.get('model_type', '?')}), "
          f"{calc.BYTES_PER_PARAM['gguf'] * 8:.2f} bits per weight")
    print(f"GPU reported by llama-bench: {first.get('gpu_info', '?')} | GPU used for estimate: {gpu.name}")
    print(f"efficiency: compute {args.compute_eff}, memory {args.memory_eff}\n")
    print(f"{'test':<12}{'measured tok/s':>16}{'estimate tok/s':>16}{'error':>9}")

    for r in runs:
        depth = r.get("n_depth", 0)
        kw = dict(precision="gguf", kv_precision="fp16", compute_eff=args.compute_eff, memory_eff=args.memory_eff)
        if r["n_prompt"] > 0 and r["n_gen"] == 0:
            name = f"pp{r['n_prompt']}"
            est = r["n_prompt"] / calc.prefill(model, gpu, r["n_prompt"], 1, **kw).time
        elif r["n_gen"] > 0 and r["n_prompt"] == 0:
            name = f"tg{r['n_gen']}"
            total = sum(calc.decode_step(model, gpu, depth + i, 1, **kw).time for i in range(r["n_gen"]))
            est = r["n_gen"] / total
        else:
            continue  # mixed pp+tg tests are skipped to keep this simple
        if depth:
            name += f"@d{depth}"
        measured = r["avg_ts"]
        print(f"{name:<12}{measured:>16.1f}{est:>16.1f}{(est - measured) / measured * 100:>+8.0f}%")


if __name__ == "__main__":
    main()
