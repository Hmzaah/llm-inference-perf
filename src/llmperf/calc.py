"""All the performance math lives in this one file.

Units used everywhere in this module:
  - memory and traffic in bytes
  - compute in FLOPs (floating point operations; one multiply-add = 2 FLOPs)
  - time in seconds
  - GPU bandwidth in bytes/s and GPU peak compute in FLOP/s

The model is a roofline model. For any piece of work we compute two times:
  compute_time = FLOPs / (peak FLOP/s * compute_efficiency)
  memory_time  = bytes moved / (bandwidth * memory_efficiency)
The real hardware can overlap the two, so the estimate is the larger of them.
Whichever one is larger is the bottleneck ("compute-bound" or "memory-bound").

The two efficiency factors are assumptions, not measurements. They say what
fraction of the datasheet peak a real inference engine reaches. They are
plain function arguments so they are always visible and easy to change.
"""

from __future__ import annotations

from dataclasses import dataclass

# Default efficiency factors (assumptions, explained in the README).
DEFAULT_COMPUTE_EFF = 0.6  # fraction of peak tensor-core FLOP/s reached by big GEMMs
DEFAULT_MEMORY_EFF = 0.8  # fraction of peak DRAM bandwidth reached by streaming reads
DEFAULT_USABLE_MEMORY_FRACTION = 0.9  # rest is left for activations, runtime, fragmentation

# Bytes needed to store one weight.
# fp16/bf16: 16 bits. int8: 8 bits. int4: 4 bits.
# q8_0 and q4_0 are llama.cpp (GGUF) block formats: every block of 32 weights
# also stores one fp16 scale (2 bytes), so q4_0 = (32*0.5 + 2) / 32 = 0.5625
# and q8_0 = (32*1 + 2) / 32 = 1.0625 bytes per weight.
BYTES_PER_PARAM = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "q8_0": 34 / 32,
    "q4_0": 18 / 32,
}

# Which tensor-core peak the matrix multiplies use for each weight format.
# fp16/bf16: plain 16-bit math.
# int8: assumes W8A8 (weights and activations both 8-bit), so the INT8 peak.
# fp8: FP8 peak.
# int4: assumes W4A16 (weights unpacked to 16-bit before the multiply, as in
#       GPTQ/AWQ kernels), so the 16-bit peak. INT4 saves memory, not compute.
# q8_0/q4_0: llama.cpp's quantized matmul kernels quantize activations to
#       8-bit and use INT8 dot products, so the INT8 peak.
DEFAULT_COMPUTE_DTYPE = {
    "fp16": "fp16",
    "bf16": "fp16",
    "fp8": "fp8",
    "int8": "int8",
    "int4": "fp16",
    "q8_0": "int8",
    "q4_0": "int8",
}

GB = 1e9  # this project uses decimal GB (1e9 bytes), like GPU datasheets do


@dataclass
class ModelSpec:
    """Shape of a dense decoder-only transformer (numbers from config.json)."""

    name: str
    n_layers: int
    hidden_size: int
    n_heads: int  # query heads
    n_kv_heads: int  # key/value heads (fewer than n_heads means GQA)
    head_dim: int
    intermediate_size: int  # width of the SwiGLU MLP
    vocab_size: int
    tie_embeddings: bool = False  # True if the output layer reuses the input embedding
    qkv_bias: bool = False  # Qwen2 adds a bias to the q, k and v projections
    params: int | None = None  # parameter count; if None it is computed


@dataclass
class GPUSpec:
    """Datasheet numbers for one accelerator. All compute numbers are DENSE."""

    name: str
    memory_gb: float
    bandwidth_gb_s: float
    fp16_tflops: float
    int8_tops: float | None = None
    fp8_tflops: float | None = None

    def peak_flops(self, compute_dtype: str) -> float:
        """Peak dense FLOP/s (or OP/s for int8) for the given math type."""
        table = {"fp16": self.fp16_tflops, "int8": self.int8_tops, "fp8": self.fp8_tflops}
        if compute_dtype not in table:
            raise ValueError(f"unknown compute dtype {compute_dtype!r}")
        value = table[compute_dtype]
        if value is None:
            raise ValueError(f"{self.name} has no published dense {compute_dtype} peak")
        return value * 1e12

    @property
    def bandwidth(self) -> float:
        return self.bandwidth_gb_s * GB


# ---------------------------------------------------------------------------
# Parameter count and memory
# ---------------------------------------------------------------------------


def count_params(m: ModelSpec) -> int:
    """Count weights of a LLaMA/Qwen-style model from its shape.

    Per layer:
      attention: q (h x h), k and v (h x kv_dim each), o (h x h)
                 where kv_dim = n_kv_heads * head_dim
      MLP (SwiGLU): gate, up (h x i) and down (i x h) = 3 * h * i
      two RMSNorm weight vectors of size h
    Plus: input embedding (vocab x h), final norm (h), output layer (h x vocab)
    unless it is tied to the input embedding.
    """
    h = m.hidden_size
    q_dim = m.n_heads * m.head_dim
    kv_dim = m.n_kv_heads * m.head_dim

    attn = h * q_dim + 2 * h * kv_dim + q_dim * h
    if m.qkv_bias:
        attn += q_dim + 2 * kv_dim
    mlp = 3 * h * m.intermediate_size
    norms = 2 * h
    per_layer = attn + mlp + norms

    embedding = m.vocab_size * h
    lm_head = 0 if m.tie_embeddings else m.vocab_size * h
    return m.n_layers * per_layer + embedding + h + lm_head


def total_params(m: ModelSpec) -> int:
    return m.params if m.params is not None else count_params(m)


def weight_bytes(m: ModelSpec, precision: str) -> float:
    """Memory to hold all weights = params x bytes per param."""
    return total_params(m) * BYTES_PER_PARAM[precision]


def kv_bytes_per_token(m: ModelSpec, kv_precision: str = "fp16") -> float:
    """KV cache for ONE token of ONE sequence.

    Every layer stores one K vector and one V vector per KV head:
      2 (K and V) x layers x kv_heads x head_dim x bytes
    """
    return 2 * m.n_layers * m.n_kv_heads * m.head_dim * BYTES_PER_PARAM[kv_precision]


def kv_cache_bytes(m: ModelSpec, seq_len: int, batch: int, kv_precision: str = "fp16") -> float:
    """KV cache = 2 x layers x kv_heads x head_dim x seq_len x batch x bytes."""
    return kv_bytes_per_token(m, kv_precision) * seq_len * batch


def matmul_params(m: ModelSpec) -> int:
    """Weights that take part in matrix multiplies for every token.

    The input embedding is a table lookup (one row per token), not a matmul,
    so it is excluded. The output layer (lm_head) IS a matmul and is included.
    """
    return total_params(m) - m.vocab_size * m.hidden_size


def weight_bytes_read_per_step(m: ModelSpec, precision: str) -> float:
    """Weight bytes streamed from memory for one forward pass.

    Every matmul weight is read once per forward pass no matter how many
    tokens are in the batch. Embedding rows read for the current tokens are
    tiny and ignored. If embeddings are tied, the lm_head reads the full table,
    which matmul_params already counts.
    """
    return matmul_params(m) * BYTES_PER_PARAM[precision]


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------


def attention_flops_per_token(m: ModelSpec, context_len: int) -> float:
    """FLOPs to attend from one new token to `context_len` cached tokens.

    Per layer and per query head:
      scores  = q . k for every cached token -> context_len * head_dim MACs
      output  = sum of weights * v          -> context_len * head_dim MACs
    2 MACs-worth of work, 2 FLOPs per MAC -> 4 * context_len * head_dim.
    Total: 4 x layers x n_heads x head_dim x context_len.
    GQA shares K/V between heads, which saves memory, but every query head
    still does its own dot products, so this uses n_heads, not n_kv_heads.
    """
    return 4 * m.n_layers * m.n_heads * m.head_dim * context_len


def flops_per_token(m: ModelSpec, context_len: int) -> float:
    """FLOPs for one token = 2 x matmul params + attention term.

    2 x params: each weight is used in one multiply-add (2 FLOPs) per token.
    """
    return 2 * matmul_params(m) + attention_flops_per_token(m, context_len)


def prefill_flops(m: ModelSpec, prompt_len: int, batch: int = 1) -> float:
    """FLOPs to process a whole prompt.

    Token number t (1-based) attends to t tokens (causal mask), so the
    attention part sums 1 + 2 + ... + prompt_len = prompt_len*(prompt_len+1)/2.
    """
    linear = 2 * matmul_params(m) * prompt_len
    attention = 4 * m.n_layers * m.n_heads * m.head_dim * prompt_len * (prompt_len + 1) / 2
    return batch * (linear + attention)


# ---------------------------------------------------------------------------
# Roofline
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    """Result of one roofline evaluation (a prefill, or one decode step)."""

    flops: float
    bytes: float
    compute_time: float
    memory_time: float

    @property
    def time(self) -> float:
        # Compute and memory traffic overlap, so the slower one sets the time.
        return max(self.compute_time, self.memory_time)

    @property
    def bottleneck(self) -> str:
        return "compute-bound" if self.compute_time >= self.memory_time else "memory-bound"

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs done per byte moved."""
        return self.flops / self.bytes


def ridge_point(gpu: GPUSpec, compute_dtype: str, compute_eff: float = 1.0, memory_eff: float = 1.0) -> float:
    """Arithmetic intensity (FLOPs/byte) where the roofline bends.

    Below it a kernel is memory-bound, above it compute-bound.
    With efficiencies = 1 this is the hardware ridge: peak FLOP/s / bandwidth.
    """
    return (gpu.peak_flops(compute_dtype) * compute_eff) / (gpu.bandwidth * memory_eff)


def roofline(
    flops: float,
    bytes_moved: float,
    gpu: GPUSpec,
    compute_dtype: str,
    compute_eff: float = DEFAULT_COMPUTE_EFF,
    memory_eff: float = DEFAULT_MEMORY_EFF,
) -> PhaseResult:
    compute_time = flops / (gpu.peak_flops(compute_dtype) * compute_eff)
    memory_time = bytes_moved / (gpu.bandwidth * memory_eff)
    return PhaseResult(flops, bytes_moved, compute_time, memory_time)


def prefill(
    m: ModelSpec,
    gpu: GPUSpec,
    prompt_len: int,
    batch: int = 1,
    precision: str = "fp16",
    kv_precision: str = "fp16",
    compute_dtype: str | None = None,
    compute_eff: float = DEFAULT_COMPUTE_EFF,
    memory_eff: float = DEFAULT_MEMORY_EFF,
) -> PhaseResult:
    """Process the whole prompt in one forward pass.

    Bytes: read every weight once + write the KV cache for every prompt token.
    Activations are ignored (they mostly stay in on-chip caches).
    """
    compute_dtype = compute_dtype or DEFAULT_COMPUTE_DTYPE[precision]
    flops = prefill_flops(m, prompt_len, batch)
    bytes_moved = weight_bytes_read_per_step(m, precision) + kv_cache_bytes(m, prompt_len, batch, kv_precision)
    return roofline(flops, bytes_moved, gpu, compute_dtype, compute_eff, memory_eff)


def decode_step(
    m: ModelSpec,
    gpu: GPUSpec,
    context_len: int,
    batch: int = 1,
    precision: str = "fp16",
    kv_precision: str = "fp16",
    compute_dtype: str | None = None,
    compute_eff: float = DEFAULT_COMPUTE_EFF,
    memory_eff: float = DEFAULT_MEMORY_EFF,
) -> PhaseResult:
    """Generate one new token for each of `batch` sequences.

    FLOPs: batch x (2 x matmul params + attention over context_len tokens).
    Bytes: all weights once (shared by the whole batch) + each sequence's
           KV cache (context_len tokens) read once.
    Weights are shared across the batch, KV cache is not: that is why
    batching helps decode until the KV cache traffic or compute takes over.
    """
    compute_dtype = compute_dtype or DEFAULT_COMPUTE_DTYPE[precision]
    flops = batch * flops_per_token(m, context_len)
    bytes_moved = weight_bytes_read_per_step(m, precision) + kv_cache_bytes(m, context_len, batch, kv_precision)
    return roofline(flops, bytes_moved, gpu, compute_dtype, compute_eff, memory_eff)


def generation_time(
    m: ModelSpec,
    gpu: GPUSpec,
    prompt_len: int,
    gen_len: int,
    batch: int = 1,
    precision: str = "fp16",
    kv_precision: str = "fp16",
    compute_dtype: str | None = None,
    compute_eff: float = DEFAULT_COMPUTE_EFF,
    memory_eff: float = DEFAULT_MEMORY_EFF,
) -> float:
    """End-to-end time: one prefill, then gen_len decode steps.

    The first output token comes from the prefill pass, then each of the
    remaining gen_len - 1 tokens needs one decode step whose context grows by 1.
    """
    kw = dict(precision=precision, kv_precision=kv_precision, compute_dtype=compute_dtype,
              compute_eff=compute_eff, memory_eff=memory_eff)
    total = prefill(m, gpu, prompt_len, batch, **kw).time
    for step in range(1, gen_len):
        total += decode_step(m, gpu, prompt_len + step, batch, **kw).time
    return total


# ---------------------------------------------------------------------------
# Memory capacity and batch size
# ---------------------------------------------------------------------------


def fits_in_memory(
    m: ModelSpec,
    gpu: GPUSpec,
    seq_len: int,
    batch: int,
    precision: str = "fp16",
    kv_precision: str = "fp16",
    usable_fraction: float = DEFAULT_USABLE_MEMORY_FRACTION,
) -> bool:
    needed = weight_bytes(m, precision) + kv_cache_bytes(m, seq_len, batch, kv_precision)
    return needed <= gpu.memory_gb * GB * usable_fraction


def max_batch(
    m: ModelSpec,
    gpu: GPUSpec,
    seq_len: int,
    precision: str = "fp16",
    kv_precision: str = "fp16",
    usable_fraction: float = DEFAULT_USABLE_MEMORY_FRACTION,
) -> int:
    """Largest batch whose KV cache fits next to the weights (0 if weights don't fit)."""
    free = gpu.memory_gb * GB * usable_fraction - weight_bytes(m, precision)
    if free <= 0:
        return 0
    per_sequence = kv_cache_bytes(m, seq_len, 1, kv_precision)
    return int(free // per_sequence)


def batch_sweep(
    m: ModelSpec,
    gpu: GPUSpec,
    seq_len: int,
    batches: list[int],
    precision: str = "fp16",
    kv_precision: str = "fp16",
    compute_dtype: str | None = None,
    compute_eff: float = DEFAULT_COMPUTE_EFF,
    memory_eff: float = DEFAULT_MEMORY_EFF,
    usable_fraction: float = DEFAULT_USABLE_MEMORY_FRACTION,
) -> list[dict]:
    """Decode throughput for each batch size, at a context of seq_len tokens."""
    limit = max_batch(m, gpu, seq_len, precision, kv_precision, usable_fraction)
    rows = []
    for b in batches:
        step = decode_step(m, gpu, seq_len, b, precision, kv_precision, compute_dtype, compute_eff, memory_eff)
        rows.append({
            "batch": b,
            "step_ms": step.time * 1e3,
            "tokens_per_s": b / step.time,
            "bottleneck": step.bottleneck,
            "fits": b <= limit,
        })
    return rows
