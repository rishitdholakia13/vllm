# Cohere MoE GB200 Reproduction Guide

Reproduce the Command A+ FP8 serving benchmark on GB200 using the
`bench-v0.19.1-cohere-moe` branch.

**Public branch:** https://github.com/rishitdholakia13/vllm/tree/bench-v0.19.1-cohere-moe

## What this branch contains

Three commits on top of upstream `v0.19.1`:

| Commit | Description |
|--------|-------------|
| `f2cb78f27` | Cherry-pick: Cohere MoE (#40817) |
| `6b89f017b` | Cherry-pick: Cohere Eagle + MoE fixes (#42078) |
| `a68a391e5` | **Fix:** `SigmoidRenorm` routing + fork sequential MoE pattern |

The fix in `a68a391e5` aligns shared-expert execution with the internal
Cohere fork (`cohere-copy-06-09`) and closes the TTFT gap vs nightly CI.

## Hardware / software requirements

| Requirement | Value used in our run |
|-------------|----------------------|
| GPU | 4x NVIDIA GB200 (TP=4) |
| Driver | 570.172.08 |
| Python | 3.12 |
| torch | 2.10.0+cu129 |
| flashinfer | 0.6.6 |
| vLLM | `0.19.2.dev2+g6b89f017b` (this branch) |

## Model

- Architecture: `Cohere2VisionForConditionalGeneration` (text-only serving)
- Quantization: FP8 (`compressed-tensors`)
- ~218B parameters
- Any valid Command A+ FP8 engine directory works; we used
  `/root/engines/c5-3a30t_fp8` (equivalent to `c4-25a218t_fp8`)

The benchmark uses `--load-format dummy`, so weights are not loaded from
disk, but the model directory is still required for config/tokenizer.

## 1. Clone and install

```bash
git clone https://github.com/rishitdholakia13/vllm.git
cd vllm
git checkout bench-v0.19.1-cohere-moe

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12
source .venv/bin/activate

# Precompiled wheels (Python-only changes on this branch)
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

Verify:

```bash
.venv/bin/python -c "import vllm, torch, flashinfer; \
  print('vllm', vllm.__version__); \
  print('torch', torch.__version__); \
  print('flashinfer', flashinfer.__version__)"
```

## 2. Run the benchmark

Set your model path, then run the repro script:

```bash
export MODEL_PATH=/path/to/your/command_a_plus_fp8_engine
export RESULT_DIR=results

bash benchmarks/cohere_moe_gb200/repro.sh
```

Or run server + client manually (see script for exact flags).

### Expected results (with fix commit `a68a391e5`)

| Metric | Target (approx) |
|--------|-----------------|
| Mean TTFT | ~69 ms |
| Output tok/s | ~208 |
| Mean TPOT | ~4.7 ms |
| torch.compile | ~50 s (first startup) |

Reference internal nightly (Cohere fork `ca9bda971`): TTFT 66.79 ms,
208 tok/s.

## 3. Verify the fix (before vs after)

To see the regression without the fix:

```bash
git checkout 6b89f017b   # cherry-picks only, no fix
# reinstall if needed, restart server, rerun repro.sh
# Expected: TTFT ~102 ms, tok/s ~235

git checkout a68a391e5   # with fix
# Expected: TTFT ~69 ms, tok/s ~208
```

## 4. Key code changes (commit `a68a391e5`)

**`vllm/model_executor/models/cohere2_moe.py`**

- Shared experts run separately *before* routed `FusedMoE`
- `reduce_results=False` on routed experts
- No `shared_experts=` passed into `FusedMoE`
- Fork-style TP all-reduce in decoder layer

**`vllm/model_executor/layers/fused_moe/config.py`**

- Adds `SigmoidRenorm` routing type (required for server startup)

## 5. Troubleshooting

**`AttributeError: RoutingMethodType has no attribute 'SigmoidRenorm'`**
→ You are on `6b89f017b` without the fix. Checkout `a68a391e5`.

**`Free memory on device ... less than desired GPU memory utilization`**
→ Kill stale vLLM processes: `pkill -f "vllm serve"` then retry.

**Server takes ~2-3 min on first start**
→ Normal: `torch.compile` + CUDA graph capture on GB200.

**TTFT still ~100 ms**
→ Confirm you are on `a68a391e5` and server log shows
`Using VLLM_CUTLASS Fp8 MoE backend`.

## 6. Output artifacts

The repro script writes to `$RESULT_DIR/`:

- `server_cohere_moe_gb200.log` — server startup log
- `bench_cohere_moe_gb200_mc1_warmup1.log` — benchmark stdout
- `serving_*_cohere_moe_gb200_warmup1.json` — structured results
- `proof_cohere_moe_gb200_env.txt` — versions / git sha for sharing
