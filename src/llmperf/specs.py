"""Load the model and GPU tables from the JSON files in llmperf/data/."""

from __future__ import annotations

import json
from pathlib import Path

from llmperf.calc import GPUSpec, ModelSpec

DATA_DIR = Path(__file__).parent / "data"

MODEL_FIELDS = ["n_layers", "hidden_size", "n_heads", "n_kv_heads", "head_dim",
                "intermediate_size", "vocab_size", "tie_embeddings", "qkv_bias"]
GPU_FIELDS = ["memory_gb", "bandwidth_gb_s", "fp16_tflops", "int8_tops", "fp8_tflops"]


def _read(filename: str) -> dict:
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


def load_models() -> dict[str, ModelSpec]:
    raw = _read("models.json")
    return {key: ModelSpec(name=v["name"], **{f: v[f] for f in MODEL_FIELDS}) for key, v in raw.items()}


def load_gpus() -> dict[str, GPUSpec]:
    raw = _read("gpus.json")
    return {key: GPUSpec(name=v["name"], **{f: v.get(f) for f in GPU_FIELDS}) for key, v in raw.items()}


def source_of(kind: str, key: str) -> str:
    """Source URL stored next to a model ('models') or GPU ('gpus') entry."""
    return _read(f"{kind}.json")[key]["source"]
