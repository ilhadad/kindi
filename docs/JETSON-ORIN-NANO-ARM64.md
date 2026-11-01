# Jetson Orin Nano ARM64 results

This report records an independent ARM64/CUDA feasibility run of Kindi 0.3.0 on an
NVIDIA Jetson Orin Nano Super. It is a hardware report, not a claim that the
released x86-64 installer or CUDA 10.2 container supports Jetson.

## Hardware and software

- Board: Jetson Orin Nano Super
- Architecture: `aarch64`
- OS: Ubuntu 24.04.4 LTS
- Kernel: `6.8.12-1021-tegra`
- CUDA: 13.2
- GPU: Orin, compute capability 8.7 (`sm_87`), 8 SMs
- Unified memory: 7.3 GiB
- Storage: NVMe, approximately 147 GiB free at the start
- Python: 3.12.3
- CuPy: `cupy-cuda13x==14.2.0`, prebuilt `manylinux2014_aarch64` wheel
- Kindi revision: `6ac903e5cb00a6d55b6cb60f6ad7445cafe93051` (v0.3.0)

The production llama.cpp service was stopped during experiments. No systemd unit was
created for Kindi. The experiments used `127.0.0.1:18080`; the existing production
port 8080 was not used.

## ARM64 enablement

The stock installer contains an x86-64 check and the release Docker image targets
CUDA 10.2/x86-64. We therefore used the source tree natively on the Jetson. A
minimal CUDA program compiled with `nvcc -O2 -arch=sm_87` and executed successfully.
Kindi imported successfully after installing the ARM64 CUDA 13 wheel.

## Model results

### Ternary-Bonsai-2 2B

- File: `Ternary-Bonsai-2-2B-PTQ1_0.gguf`
- Size: 429,183,456 bytes
- SHA-256: `36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa`
- Kindi pack: approximately 494 MiB
- Verification: 4/4 prompts token-exact with llama.cpp, 40/40 tokens per prompt
- Benchmark at 2,048 context: prompt 108.5 tok/s; generation 92.4–92.7 tok/s
- API smoke test: passed on `127.0.0.1:18080`

### Ternary-Bonsai-2 4B

- File: `Ternary-Bonsai-2-4B-PTQ1_0.gguf`
- Size: 949,549,920 bytes
- SHA-256: `56711263ec9a9ffeaa74a3a0e58dece8127cf9268404746c721113ca76994d60`
- Kindi pack: approximately 1.1 GiB
- Verification: 4/4 prompts token-exact with llama.cpp, 40/40 tokens per prompt
- Benchmark at 2,048 context: prompt 61.0 tok/s; generation 54.3 tok/s at position 100 and 52.5 tok/s at position 2,000
- Lean server at 4,096 context: generation approximately 56.15 tok/s; approximately 4.3 GiB RAM available after startup
- 65,536-context stress test: startup and short request passed; approximately 3.0 GiB RAM available after startup and approximately 3.1 GiB GPU memory free

### Ternary-Bonsai-2 9B

- File: `Ternary-Bonsai-2-9B-PTQ1_0.gguf`
- Size: 1,997,486,976 bytes
- SHA-256: `3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b`
- Kindi pack: approximately 2.3 GiB
- Verification: 4/4 prompts passed; 35.5 tok/s aggregate over verification
- API smoke test at 1,024 context: generation 34.06 tok/s
- API smoke test at 2,048 context: generation 33.91 tok/s
- 65,536-context startup and short request passed, but only approximately 1.9 GiB RAM and 1.8 GiB GPU memory remained

## Memory and context observations

Jetson CPU and GPU share the same physical memory. The 9B runs repeatedly reported
that a 2.23 GB pinned host allocation could not be made and fell back to synchronous
transfers. These runs were functional but should be treated as stress tests, not as
practical deployment recommendations.

The 4B 65K run accepted a roughly 19K-token Hermes prompt, but reading it took
several minutes at Jetson prompt-processing rates. A cache-enabled repeat was not
completed because the first 42K-token test request exceeded the client test timeout
and disconnected while Kindi was still reading the prompt.

Recommended Jetson starting point from this run: 4B, exact head, 4K context for
interactive use. The 4B 64K configuration is technically functional but has limited
headroom.

## Reproduction outline

```text
python3 -m venv ~/jetson-kindi/.venv
~/jetson-kindi/.venv/bin/python -m pip install cupy-cuda13x==14.2.0
PYTHONPATH=~/jetson-kindi/source/kindi \
  ~/jetson-kindi/.venv/bin/python -m kindi convert MODEL.gguf --out PACK
PYTHONPATH=~/jetson-kindi/source/kindi \
  ~/jetson-kindi/.venv/bin/python -m kindi --pack PACK verify
PYTHONPATH=~/jetson-kindi/source/kindi \
  ~/jetson-kindi/.venv/bin/python -m kindi --pack PACK --ctx 4096 serve \
    --host 127.0.0.1 --port 18080 --head exact
```

For low-memory testing, the 4B server used `KINDI_PROMPT_CACHE=0`,
`KINDI_PARK_MAX=0`, `KINDI_CHECKPOINT_BUDGET_MB=256`, and `KINDI_BATCH=1`.

## Disclosure

This report was prepared with AI assistance using Hermes Agent. Measurements were
performed on the Jetson and the commands/results were checked against the local logs.
The model files are Prism ML artifacts; no model files are included in this report.
