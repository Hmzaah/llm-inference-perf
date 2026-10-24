"""Smoke tests: every CLI command runs and prints the key fields."""

from llmperf.cli import main


def test_estimate_prints_table(capsys):
    main(["estimate", "--model", "llama-3.1-8b", "--gpu", "a100-80gb", "--batch", "8",
          "--seq-len", "2048", "--precision", "int4"])
    out = capsys.readouterr().out
    for field in ["weights", "KV cache", "fits in memory", "prefill time", "decode time per step",
                  "tokens/s", "memory-bound", "compute-bound"]:
        assert field in out


def test_custom_gpu(capsys):
    main(["estimate", "--gpu", "custom", "--memory-gb", "16", "--bandwidth-gb-s", "448", "--fp16-tflops", "50"])
    assert "custom GPU" in capsys.readouterr().out


def test_sweep_quant_list(capsys):
    main(["sweep", "--batches", "1,2,4"])
    main(["quant", "--gpu", "gaudi2"])
    main(["list"])
    out = capsys.readouterr().out
    assert "max batch" in out and "int4" in out and "Intel Gaudi 2" in out
