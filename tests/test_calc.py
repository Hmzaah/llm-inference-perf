"""Each test compares a formula with a value worked out by hand (shown in the comment)."""

import pytest

from llmperf import calc
from llmperf.specs import load_gpus, load_models

MODELS = load_models()
GPUS = load_gpus()
LLAMA3 = MODELS["llama-3.1-8b"]
QWEN = MODELS["qwen2.5-7b"]
A100 = GPUS["a100-80gb"]


# ---- parameter counts ------------------------------------------------------

def test_llama31_8b_param_count():
    # embedding 128256*4096 = 525,336,576 (x2: input + lm_head)
    # per layer: q 4096*4096 + k,v 2*4096*1024 + o 4096*4096 = 41,943,040
    #            mlp 3*4096*14336 = 176,160,768, norms 2*4096 = 8,192
    #            -> 218,112,000 x 32 layers = 6,979,584,000
    # total = 6,979,584,000 + 1,050,673,152 + 4,096 (final norm)
    assert calc.count_params(LLAMA3) == 8_030_261_248


def test_qwen25_7b_matches_model_card():
    # card: 7.61B total, 6.53B non-embedding
    total = calc.count_params(QWEN)
    assert total == 7_615_616_512
    non_embedding = total - 2 * QWEN.vocab_size * QWEN.hidden_size
    assert round(non_embedding / 1e9, 2) == 6.53


def test_llama2_7b_matches_published_gguf_count():
    assert calc.count_params(MODELS["llama-2-7b"]) == 6_738_415_616


def test_deepseek_distill_is_same_shape_as_qwen():
    assert calc.count_params(MODELS["deepseek-r1-distill-qwen-7b"]) == calc.count_params(QWEN)


# ---- memory ----------------------------------------------------------------

def test_weight_bytes_by_precision():
    p = 8_030_261_248
    assert calc.weight_bytes(LLAMA3, "fp16") == p * 2
    assert calc.weight_bytes(LLAMA3, "int8") == p * 1
    assert calc.weight_bytes(LLAMA3, "int4") == p * 0.5
    # q4_0: 32 weights x 4 bits + 2-byte scale = 18 bytes per 32 weights
    assert calc.BYTES_PER_PARAM["q4_0"] == 0.5625


def test_kv_cache_bytes():
    # 2 (K,V) x 32 layers x 8 kv heads x 128 head_dim x 2 bytes = 131,072 bytes per token
    assert calc.kv_bytes_per_token(LLAMA3) == 131_072
    # x 2048 tokens x batch 8 = 2,147,483,648 bytes
    assert calc.kv_cache_bytes(LLAMA3, 2048, 8) == 2_147_483_648
    # int8 KV halves it
    assert calc.kv_cache_bytes(LLAMA3, 2048, 8, "int8") == 1_073_741_824


def test_max_batch_a100_fp16_2k():
    # usable = 0.9 * 80e9 = 72e9; weights 16,060,522,496 -> free 55,939,477,504
    # per sequence 131,072 * 2048 = 268,435,456 -> 208.39 -> 208
    assert calc.max_batch(LLAMA3, A100, 2048) == 208


def test_max_batch_zero_when_weights_do_not_fit():
    rtx = GPUS["rtx-4090"]
    # an 8 GB GPU cannot hold 16.06 GB of fp16 weights; a 24 GB RTX 4090 can (limit 21.6 GB)
    tiny = calc.GPUSpec("tiny", memory_gb=8, bandwidth_gb_s=100, fp16_tflops=10)
    assert calc.max_batch(LLAMA3, tiny, 2048) == 0
    assert calc.max_batch(LLAMA3, rtx, 2048) > 0


# ---- FLOPs -----------------------------------------------------------------

def test_flops_per_token():
    # matmul params = 8,030,261,248 - 525,336,576 = 7,504,924,672 -> x2
    assert calc.flops_per_token(LLAMA3, 0) == 15_009_849_344
    # attention term: 4 x 32 layers x 32 heads x 128 x 1000 = 524,288,000
    assert calc.attention_flops_per_token(LLAMA3, 1000) == 524_288_000
    assert calc.flops_per_token(LLAMA3, 1000) == 15_009_849_344 + 524_288_000


def test_prefill_flops_small_prompt():
    # 2 tokens: linear 2 x 7,504,924,672 x 2 = 30,019,698,688
    # attention: token 1 sees 1, token 2 sees 2 -> 3 x 524,288 = 1,572,864
    assert calc.prefill_flops(LLAMA3, 2) == 30_021_271_552
    assert calc.prefill_flops(LLAMA3, 2, batch=3) == 3 * 30_021_271_552


# ---- roofline --------------------------------------------------------------

def test_ridge_point_a100():
    # 312e12 / 2039e9 = 153.016 FLOPs/byte
    assert calc.ridge_point(A100, "fp16") == pytest.approx(153.016, rel=1e-4)


def test_decode_batch1_is_memory_bound_and_matches_hand_value():
    step = calc.decode_step(LLAMA3, A100, context_len=0, batch=1, precision="int4",
                            compute_eff=1.0, memory_eff=1.0)
    # bytes = 7,504,924,672 x 0.5 = 3,752,462,336; time = bytes / 2.039e12 = 1.8403 ms
    assert step.bytes == 3_752_462_336
    assert step.time == pytest.approx(1.8403e-3, rel=1e-4)
    assert step.bottleneck == "memory-bound"
    # intensity ~ 15e9 FLOPs / 3.75e9 bytes = 4 FLOPs/byte, far below the ridge (~153)
    assert step.arithmetic_intensity == pytest.approx(4.0, rel=1e-6)


def test_efficiency_factors_scale_time():
    full = calc.decode_step(LLAMA3, A100, 0, precision="int4", compute_eff=1.0, memory_eff=1.0)
    half = calc.decode_step(LLAMA3, A100, 0, precision="int4", compute_eff=1.0, memory_eff=0.5)
    assert half.time == pytest.approx(2 * full.time)


def test_prefill_long_prompt_is_compute_bound():
    pre = calc.prefill(LLAMA3, A100, prompt_len=2048)
    assert pre.bottleneck == "compute-bound"
    # time = FLOPs / (312e12 x 0.6)
    assert pre.time == pytest.approx(calc.prefill_flops(LLAMA3, 2048) / (312e12 * 0.6))


def test_large_batch_decode_becomes_compute_bound():
    # with short context the KV traffic is small, so a big batch crosses the ridge
    step = calc.decode_step(LLAMA3, A100, context_len=16, batch=512, precision="fp16")
    assert step.bottleneck == "compute-bound"


def test_batch_sweep_throughput_rises_then_flattens():
    rows = calc.batch_sweep(LLAMA3, A100, 512, [1, 2, 4, 8, 16, 32, 64, 128, 256])
    tps = [r["tokens_per_s"] for r in rows]
    assert all(tps[i + 1] > tps[i] for i in range(len(tps) - 1))
    # doubling batch from 1 to 2 nearly doubles throughput (weights shared) ...
    assert tps[1] / tps[0] > 1.9
    # ... but from 128 to 256 it gains much less
    assert tps[-1] / tps[-2] < 1.5


def test_int4_decode_faster_than_fp16():
    fp16 = calc.decode_step(LLAMA3, A100, 1024, precision="fp16").time
    int4 = calc.decode_step(LLAMA3, A100, 1024, precision="int4").time
    assert int4 < fp16


def test_generation_time_is_prefill_plus_decode_steps():
    t = calc.generation_time(LLAMA3, A100, prompt_len=10, gen_len=3)
    expected = (calc.prefill(LLAMA3, A100, 10).time
                + calc.decode_step(LLAMA3, A100, 11).time
                + calc.decode_step(LLAMA3, A100, 12).time)
    assert t == pytest.approx(expected)


def test_missing_peak_raises():
    with pytest.raises(ValueError):
        A100.peak_flops("fp8")
