# Native ARM64 setup on Jetson Orin Nano

This is the native source-build path used to run Kindi 0.3.0 on a Jetson Orin Nano Super. It is separate from the released one-command installer: the installer and Docker image currently target Linux x86-64/CUDA 10.2.

## Scope and safety

- Tested on JetPack R39 / Ubuntu 24.04.4 LTS / CUDA 13.2 / Python 3.12 / `aarch64`.
- The Jetson has unified CPU/GPU memory. Stop other GPU inference services before loading a larger model.
- Start on a small model. The 2B and 4B models are practical feasibility targets; 9B is memory-sensitive; 27B was not attempted.
- Use a separate experimental port such as `18080`; do not replace an existing production service.
- Do not enable a systemd service until manual validation is complete.

## Jetson Orin Nano 8 GB profile

The board is marketed as an 8 GB device, but the tested system exposed about
7.3 GiB of unified memory to Linux. CPU allocations, CUDA allocations, model
weights, the K/V cache, filesystem cache and the operating system all compete
for this same pool; there is no separate VRAM capacity to add to it.

Recommended starting profile:

| Model | Context | Head | Prompt cache | Expected use |
|---|---:|---|---|---|
| Bonsai-2 2B | 2,048–65,536 | exact | optional | fastest feasibility target |
| Bonsai-2 4B | 4,096 | exact | off initially | recommended interactive target |
| Bonsai-2 4B | 65,536 | exact | off or tightly bounded | stress test; limited headroom |
| Bonsai-2 9B | 1,024–4,096 | exact | off initially | memory-sensitive experiment |
| Bonsai-2 27B | — | — | — | not attempted; not a practical target on this board |

For the 4B low-memory profile, use:

```bash
export KINDI_PROMPT_CACHE=0
export KINDI_PARK_MAX=0
export KINDI_CHECKPOINT_BUDGET_MB=256
export KINDI_BATCH=1
```

These settings trade repeated-prompt speed for memory headroom. Prompt caching
can help after a complete prompt has been processed, but the first long prompt
still has to be read. A 1 GiB checkpoint budget was tested with a 4B/65K run;
it increased memory pressure and the first 42K-token request did not finish
before the client timeout. Do not assume that a 64K context is practical just
because the server accepts it.

## 1. Check prerequisites

```bash
uname -m                         # aarch64
python3 --version                # Python 3.12.x tested
/usr/local/cuda/bin/nvcc --version
free -h
df -h ~
```

Confirm that the CUDA compiler can target the Orin architecture (`sm_87`) before installing Python dependencies.

## 2. Create an isolated workspace

```bash
mkdir -p ~/jetson-kindi/{source,model,pack,state,logs,results,config}
cd ~/jetson-kindi

git clone --branch v0.3.0 --depth 1 \
  https://github.com/ahtx/kindi.git source/kindi
```

Record the revision:

```bash
git -C ~/jetson-kindi/source/kindi rev-parse HEAD
```

## 3. Create a virtual environment

On Ubuntu, install the venv package if needed:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
```

Create the environment without modifying the system Python:

```bash
python3 -m venv ~/jetson-kindi/.venv
```

## 4. Install CuPy using the ARM64 wheel

Do **not** build the generic `cupy` source package on a 7.3 GiB Jetson. Its full C++/CUDA build can exhaust RAM and start using swap.

Use the prebuilt CUDA 13 ARM64 wheel:

```bash
V=$HOME/jetson-kindi/.venv
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}

"$V/bin/python" -m pip install --no-cache-dir \
  cupy-cuda13x==14.2.0
```

The tested wheel was:

```text
cupy_cuda13x-14.2.0-cp312-cp312-manylinux2014_aarch64.whl
```

Verify CUDA access:

```bash
"$V/bin/python" - <<'PY'
import cupy as cp
print("cupy", cp.__version__)
print("devices", cp.cuda.runtime.getDeviceCount())
d = cp.cuda.Device()
print("compute_capability", d.compute_capability)
a = cp.arange(1024, dtype=cp.float32)
print("sum", float(cp.sum(a + 2*a).get()))
PY
```

Expected device capability on Orin is `87`.

## 5. Download and verify a model

Start with the 2B or 4B PTQ1_0 model. Example for 4B:

```bash
ROOT=$HOME/jetson-kindi
FILE=$ROOT/model/Ternary-Bonsai-2-4B-PTQ1_0.gguf
URL=https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf/resolve/main/Ternary-Bonsai-2-4B-PTQ1_0.gguf
wget -O "$FILE" "$URL"
sha256sum "$FILE"
```

Expected 4B metadata:

```text
size: 949549920 bytes
sha256: 56711263ec9a9ffeaa74a3a0e58dece8127cf9268404746c721113ca76994d60
```

Expected 2B metadata:

```text
size: 429183456 bytes
sha256: 36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa
```

## 6. Convert and verify

Set the source and environment:

```bash
ROOT=$HOME/jetson-kindi
V=$ROOT/.venv
SRC=$ROOT/source/kindi
export PYTHONPATH=$SRC
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}
export CUPY_CACHE_DIR=$ROOT/state/cupy-cache
```

Convert the 4B model to a separate pack:

```bash
PACK=$ROOT/pack/bonsai2-4b
mkdir -p "$PACK"
"$V/bin/python" -m kindi convert \
  "$ROOT/model/Ternary-Bonsai-2-4B-PTQ1_0.gguf" \
  --out "$PACK"
```

Run the built-in token-equivalence verification:

```bash
"$V/bin/python" -m kindi --pack "$PACK" verify
```

A successful run reports all four prompts passing token-for-token against llama.cpp.

## 7. Start an isolated server

For a conservative interactive configuration on the Jetson:

```bash
export KINDI_PROMPT_CACHE=0
export KINDI_PARK_MAX=0
export KINDI_CHECKPOINT_BUDGET_MB=256
export KINDI_BATCH=1

"$V/bin/python" -m kindi \
  --pack "$ROOT/pack/bonsai2-4b" \
  --state-dir "$ROOT/state-4b-4096" \
  --ctx 4096 \
  serve \
  --host 127.0.0.1 \
  --port 18080 \
  --head exact
```

The cache settings reduce host-RAM use. Prompt caching can help repeated prompts, but a large first prompt still has to be read once. On a constrained Jetson, use a long client timeout and one request at a time.

Check the service:

```bash
curl http://127.0.0.1:18080/health
curl http://127.0.0.1:18080/v1/models
```

## 8. OpenAI-compatible request

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bonsai2-4b",
    "messages": [{
      "role": "user",
      "content": "Say hello from the Jetson."
    }],
    "max_tokens": 32,
    "temperature": 0
  }'
```

## 9. Optional SSH tunnel

Keep Kindi bound to Jetson localhost and forward it securely:

```bash
ssh -i ~/.ssh/pihole_manage \
  -o IdentitiesOnly=yes \
  -N \
  -L 18081:127.0.0.1:18080 \
  ilhad@10.10.0.38
```

Then use the local endpoint:

```text
http://127.0.0.1:18081/v1
```

## 10. Benchmark and monitor

Record the exact model, revision, context, cache settings, command, prompt rate, generation rate, RAM, swap, temperature, and failures.

Useful commands:

```bash
free -h
swapon --show
tegrastats --interval 1000
curl http://127.0.0.1:18080/health
```

The tested 4B results on a Jetson Orin Nano Super were approximately 61 tok/s prompt processing and 52–54 tok/s generation at 2048 context. A 4B 65536-context stress test started and answered a short request, but left substantially less memory headroom and is not the recommended interactive setting.

During larger-model tests, watch both RAM and swap. A pinned-memory warning such
as `cudaErrorMemoryAllocation` means Kindi has fallen back to synchronous host
transfers; it is functional evidence, not evidence of an ideal deployment.

```bash
free -h
swapon --show
tegrastats --interval 1000
```

## Limitations

- The released installer still rejects `aarch64` and the released Docker image targets x86-64/CUDA 10.2.
- CuPy source compilation was not practical on the 7.3 GiB board; use the ARM64 CUDA 13 wheel or an equivalent Jetson-compatible package.
- CPU and GPU share memory. CPU offload or a larger context does not create an independent capacity pool.
- Swap is a safety net, not a practical inference tier.
- Restore any production service manually after experiments; do not add a permanent Kindi systemd unit as part of this setup.
