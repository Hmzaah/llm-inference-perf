# Explain it like I'm interviewing

A study guide for this project. Every number below can be reproduced with the CLI
(`python -m llmperf estimate ...`) or by hand with the formulas in `src/llmperf/calc.py`.
Running example: **LLaMA 3.1 8B** (32 layers, hidden 4096, 32 query heads, 8 KV heads, head dim 128,
8.03B parameters) on an **A100 80GB SXM** (2039 GB/s, 312 TFLOPS dense FP16, 624 TOPS dense INT8).

---

## 1. Prefill vs decode

Generating text has two phases.

- **Prefill**: the whole prompt goes through the model in one forward pass. With a 1024-token prompt, every weight
  matrix is multiplied by a 1024-row matrix of activations. Lots of math per byte of weights read.
  Its time is the **time to first token (TTFT)**.
- **Decode**: after that, the model makes one token at a time. Each new token needs a full forward pass, but
  with only one row of activations per sequence. Its time per step is the **inter-token latency**, and
  batch / step time is the **throughput** in tokens/s.

Same model, same weights, completely different hardware behaviour.

## 2. Why decode is memory-bound

In one decode step at batch 1, each weight is read from GPU memory once and used for exactly one multiply-add
(2 FLOPs). In FP16 a weight is 2 bytes, so the work is about **1 FLOP per byte**. The A100 can do 312e12 FLOP/s but
only move 2039e9 bytes/s, a ratio of **153 FLOPs per byte**. Decode supplies 1, so the compute units wait for memory
more than 99% of the time. The time per token is set by how fast the weights can be streamed:

```
decode time ~ (weight bytes + KV cache bytes) / bandwidth
```

Prefill uses each weight for every prompt token, so its FLOPs per byte is roughly the prompt length; with 1024
tokens it is far above 153 and prefill is compute-bound.

## 3. KV cache

Attention needs the keys (K) and values (V) of every earlier token. Recomputing them every step would be wasteful, so
they are stored: one K and one V vector per layer per KV head per token.

```
KV bytes = 2 x layers x kv_heads x head_dim x tokens x batch x bytes_per_value
LLaMA 3.1 8B, FP16: 2 x 32 x 8 x 128 x 2 = 131,072 bytes per token (128 KiB)
```

The KV cache grows linearly with context length and with batch, and in decode it must be **read in full every step**.
At long contexts and big batches it becomes larger than the weights and dominates both memory capacity and bandwidth.

## 4. GQA (grouped-query attention)

In plain multi-head attention every query head has its own K and V head. GQA lets a group of query heads share one
K/V head. LLaMA 3.1 8B has 32 query heads but only 8 KV heads, so groups of 4 share.

- **Saves**: KV cache memory and KV read bandwidth, 4x here. Also slightly fewer parameters in the k and v projections.
- **Does not save**: attention FLOPs. Every query head still computes its own dot products against the shared keys,
  so the attention FLOP term uses the number of *query* heads.

## 5. Roofline, ridge point, arithmetic intensity

**Arithmetic intensity** = FLOPs done / bytes moved from memory.

The **roofline** says attainable FLOP/s = min(peak FLOP/s, intensity x bandwidth). Plotted on log-log axes it is a
sloped line (memory-bound region) that bends into a flat roof (compute-bound region). The bend is the
**ridge point** = peak FLOP/s / bandwidth:

| GPU | ridge (FP16 dense) |
|---|---|
| A100 80GB | 312e12 / 2039e9 = 153 FLOPs/byte |
| H100 SXM | 989.5e12 / 3350e9 = 295 FLOPs/byte |

A newer GPU usually gains compute faster than bandwidth, so its ridge moves right and *more* workloads become
memory-bound. In this tool each phase gets two times, compute_time and memory_time; the larger one is the estimate and
names the bottleneck. Efficiency factors (0.6 for compute, 0.8 for memory by default) scale the two roofs down, which
moves the effective ridge to 153 x 0.6 / 0.8 = 115 on the A100.

## 6. Batching trade-offs

In decode, a batch of B sequences shares one read of the weights, but each sequence brings its own KV cache.

- At short context: FLOPs grow with B, weight bytes do not, so intensity ~ B (in FP16). Tokens/s grows almost
  linearly until B passes the effective ridge (~115 on the A100), then flattens at the compute roof.
- At long context: KV bytes grow with B as fast as FLOPs, so intensity stops growing (about 60 FLOPs/byte for
  LLaMA 3.1 8B at 2048 tokens) and decode never reaches the compute roof. Memory capacity also caps B:
  max batch = (usable memory - weights) / KV per sequence.
- The cost: each step takes longer as B grows, so latency per user gets worse while total throughput improves.
  Big prefills in the same batch also delay everyone's next token. Serving systems balance throughput against
  per-user latency targets.

## 7. What quantization saves and what it costs

| weights | bytes/param | LLaMA 3.1 8B weights | A100 decode, batch 1, 1024 ctx | max batch at 1024 ctx |
|---|---|---|---|---|
| FP16 | 2 | 16.06 GB | 9.28 ms (108 tok/s) | 416 |
| INT8 | 1 | 8.03 GB | 4.68 ms (214 tok/s) | 476 |
| INT4 | 0.5 | 4.02 GB | 2.38 ms (420 tok/s) | 506 |

(default efficiencies; KV cache kept in FP16)

**Saves**: memory (fits on smaller GPUs, leaves room for more KV cache, so bigger batches) and decode time, because
decode is memory-bound and there are fewer bytes to stream.

**Costs**:
- Accuracy: fewer bits per weight means rounding error; INT4 needs careful methods (group-wise scales, GPTQ, AWQ).
- Compute does not always get cheaper. Weight-only INT4 (W4A16) unpacks weights to FP16 before multiplying, so
  prefill runs at FP16 speed and pays extra work for the unpacking. Only W8A8 (or FP8) uses the faster 8-bit math.
- Weight quantization does nothing for the KV cache. At long context the KV cache dominates, so you also need KV
  cache quantization (FP8/INT8 KV) to keep gaining.
- Scales and zero points add storage: llama.cpp Q4_0 is 4.5 bits per weight, not 4.

---

## 15 likely interview questions

**1. Walk me through this project in two minutes.**
It estimates LLM inference performance from first principles. From a model's config (layers, hidden size, heads, KV
heads, MLP width, vocab) it computes parameter count, weight memory and KV cache size. From a GPU's datasheet
(bandwidth, dense FP16/INT8 peak, memory size) it builds a roofline. For prefill and for each decode step it computes
FLOPs and bytes moved, takes the larger of compute time and memory time, and reports time, tokens/s and which one is
the bottleneck. It sweeps batch size and precision and compares against published numbers: within about -17% to +34%
for large-batch Gaudi 2 serving, but 2-2.5x too optimistic for batch-1 llama.cpp on A100/H100, and I can explain why.

**2. Why does decode get faster with INT4 but prefill doesn't?**
Decode is memory-bound, so its time is roughly bytes / bandwidth; INT4 cuts weight bytes 4x versus FP16. Prefill is
compute-bound, and with weight-only INT4 the multiply still runs in FP16 after unpacking, so the FLOP rate does not
change (it can even get slower from the unpacking work).

**3. Compute the KV cache for LLaMA 3.1 8B at 8k context, batch 16, FP16.**
131,072 bytes per token x 8192 x 16 = **17.2 GB**. Without GQA (32 KV heads instead of 8) it would be 4x, 68.7 GB,
which together with 16 GB of FP16 weights would not fit on an 80 GB GPU.

**4. Worked example: how many tokens/s for LLaMA 3.1 8B, INT4, batch 1, on an A100, and why?**
- Weights read per token: 7.50e9 matmul parameters (8.03e9 minus the 0.53e9 input embedding, which is just a
  lookup) x 0.5 bytes = 3.75 GB.
- KV cache read at 1024 tokens of context: 1024 x 131,072 = 0.13 GB. Total ~3.89 GB per token.
- FLOPs: 2 x 7.50e9 + attention 4 x 32 x 32 x 128 x 1024 = 15.55e9. Intensity = 15.55e9 / 3.89e9 = 4 FLOPs/byte,
  way below the ridge of 153, so memory-bound.
- Time = 3.89e9 / 2039e9 = 1.91 ms at datasheet bandwidth, about **525 tokens/s** as an upper bound.
  With 80% bandwidth efficiency: 2.38 ms, about **420 tokens/s**.
- Compute time is only 15.55e9 / 312e12 = 0.05 ms, so the tensor cores are idle most of the time.
- Caveat from my validation: real engines at batch 1 can land well below this on datacenter GPUs because of fixed
  per-token overheads; llama.cpp reached about a third of A100 bandwidth on a 7B 4-bit model.

**5. What is the ridge point of an H100 and what does it mean?**
989.5 TFLOPS dense FP16 / 3.35 TB/s = about 295 FLOPs/byte. A kernel needs at least that many FLOPs per byte to be
compute-bound. It is higher than the A100's 153, so on an H100 you need an even bigger batch before decode stops
being memory-bound.

**6. Why does the tool use n_heads and not n_kv_heads in the attention FLOPs?**
GQA shares K and V tensors between query heads, but each query head still does its own q.k dot products and its own
weighted sum of V. Memory traffic depends on KV heads; FLOPs depend on query heads.

**7. Where does "FLOPs per token = 2 x parameters" come from, and when is it wrong?**
Each weight takes part in one multiply-add per token. It ignores attention over the context, which costs
4 x layers x heads x head_dim x context per token. For LLaMA 3.1 8B the two are equal at about 28,600 tokens of
context, so at 128k context attention dominates. It also ignores softmax, norms and activations, which are small.

**8. How does batch size change decode throughput, and when does it stop helping?**
Weights are read once per step for the whole batch, so throughput rises almost linearly at first. It stops when
(a) the step becomes compute-bound (batch above the effective ridge, about 115 for FP16 on an A100 at short context),
(b) the KV cache traffic dominates at long context, or (c) the KV cache no longer fits in memory. The plot in the
README shows (a) at 256 tokens and (b)/(c) at 2048 tokens.

**9. Your estimate for llama.cpp on an H100 was 2.5x too fast. Why, and how would you fix the model?**
The measured data says it is not bandwidth: the A100 has about twice the bandwidth of the RTX 4090 but generated
about the same tokens/s. At batch 1 one token is only ~1-2 ms of memory traffic, so fixed per-token costs (many
small kernels per layer, synchronization, small matrices that cannot keep HBM busy) matter. To fix it I would add a
per-token (or per-kernel) overhead term and calibrate it, plus a size-dependent bandwidth efficiency from
microbenchmarks, instead of one constant.

**10. Why are your efficiency factors 0.6 and 0.8, and how would you calibrate them properly?**
They are assumptions: large GEMMs usually reach a good fraction of tensor-core peak but not all of it, and streaming
reads get closer to peak bandwidth than compute does. I fixed them before running validation and did not fit them.
To calibrate: run a bandwidth microbenchmark (big copy) and GEMM benchmarks across the matrix shapes the model uses
on the real GPU, and use measured efficiency per kernel shape instead of one global number.

**11. How would you extend this to tensor parallelism on 2 or 8 GPUs?**
Split each weight matrix and the KV heads across N GPUs, so weight bytes and KV bytes per GPU drop by N and compute
per GPU drops by N. Add communication: with Megatron-style splitting there are two all-reduces per layer of the
activations (batch x tokens x hidden x bytes). Model each with a ring all-reduce cost of about
2 x (N - 1) / N x message size / link bandwidth, plus a latency term, which dominates in decode where messages are tiny.
Then sweep interconnect bandwidth the same way I sweep memory bandwidth.

**12. How would you add a mixture-of-experts model like DeepSeek V3?**
Memory must hold all experts, but each token only uses the router-selected few, so FLOPs per token use active
parameters. Bytes per decode step depend on how many distinct experts the whole batch touches: at batch 1 only the
active experts are read, at large batch almost all experts are read. That makes MoE decode strongly batch-dependent.
DeepSeek V3 also uses multi-head latent attention, which stores a compressed KV, so the KV formula changes too.
That is why I left MoE out of this version.

**13. Time to first token vs inter-token latency: what controls each?**
TTFT is prefill time (plus queueing): compute-bound, so it scales with prompt length, model FLOPs and peak FLOP/s.
Inter-token latency is decode step time: memory-bound, so it scales with weight bytes, KV cache bytes and
bandwidth. Quantization mainly helps inter-token latency; more compute (or smaller models) mainly helps TTFT.

**14. If you could change one hardware spec to speed up batch-1 LLM decode, which would it be?**
Memory bandwidth (or more on-chip memory so weights do not stream from DRAM every token). Decode at batch 1 is at 1-4
FLOPs/byte, so doubling peak FLOPs changes almost nothing, while doubling bandwidth nearly halves the step time.
This is the kind of what-if sweep the model is for: change bandwidth, capacity or compute and see which phase moves.

**15. Does LLaMA 3.1 8B in FP16 fit on an RTX 4090 at 8k context? What would you change?**
Weights are 16.06 GB and the KV cache at 8192 tokens is 1.07 GB per sequence. With 90% of 24 GB usable (21.6 GB) it
fits, but only up to batch 5. Switching weights to INT4 (4.02 GB) raises the max batch to 16; quantizing the KV cache
to 8-bit would roughly double that again. The tool shows this with
`python -m llmperf estimate --gpu rtx-4090 --seq-len 8192 --precision int4`.
