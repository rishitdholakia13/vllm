#!/usr/bin/env bash
# Reproduce Cohere MoE GB200 serving benchmark (mc=1, 1000/1000, 2 prompts).
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/engines/c5-3a30t_fp8}"
RESULT_DIR="${RESULT_DIR:-results}"
PORT="${PORT:-8000}"
TP="${TENSOR_PARALLEL_SIZE:-4}"
VLLM_BIN="${VLLM_BIN:-.venv/bin/vllm}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

if [[ ! -x "$VLLM_BIN" ]]; then
  echo "ERROR: vLLM not found at $VLLM_BIN"
  echo "Run: VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto"
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH"
  echo "Set MODEL_PATH to your Command A+ FP8 engine directory."
  exit 1
fi

mkdir -p "$RESULT_DIR"

SERVER_LOG="$RESULT_DIR/server_cohere_moe_gb200.log"
BENCH_LOG="$RESULT_DIR/bench_cohere_moe_gb200_mc1_warmup1.log"
PROOF_FILE="$RESULT_DIR/proof_cohere_moe_gb200_env.txt"
RESULT_JSON="$RESULT_DIR/serving_cohere_moe_gb200_mc1_warmup1.json"

echo "=== Cohere MoE GB200 repro ==="
echo "MODEL_PATH=$MODEL_PATH"
echo "RESULT_DIR=$RESULT_DIR"
echo "PORT=$PORT TP=$TP"
echo

# Kill stale server on this port
fuser -k "${PORT}/tcp" 2>/dev/null || true
pkill -f "vllm serve.*--port ${PORT}" 2>/dev/null || true
sleep 2

# Write environment proof for sharing
{
  echo "=== git ==="
  git describe --tags --always
  git log -1 --oneline
  echo "=== python packages ==="
  "$PYTHON_BIN" -c "
import vllm, torch
try:
    import flashinfer
    fi = flashinfer.__version__
except Exception:
    fi = 'not installed'
print('vllm', vllm.__version__)
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('flashinfer', fi)
"
  echo "=== nvidia-smi ==="
  nvidia-smi | head -3
  echo "=== model ==="
  echo "$MODEL_PATH"
  if [[ -f "$MODEL_PATH/config.json" ]]; then
    "$PYTHON_BIN" -c "
import json
with open('$MODEL_PATH/config.json') as f:
    c = json.load(f)
print('architectures:', c.get('architectures'))
print('total_parameters:', c.get('total_parameters', 'n/a'))
"
  fi
} > "$PROOF_FILE"
echo "Wrote $PROOF_FILE"

echo "Starting server (log: $SERVER_LOG) ..."
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
"$VLLM_BIN" serve "$MODEL_PATH" \
  --disable-log-stats \
  --tensor-parallel-size "$TP" \
  --load-format dummy \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.95 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --max-model-len 256000 \
  --max-cudagraph-capture-size 128 \
  --attention-backend FLASHINFER \
  --reasoning-config '{"reasoning_start_str":"<|START_THINKING|>","reasoning_end_str":"<|END_THINKING|>"}' \
  --port "$PORT" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  fuser -k "${PORT}/tcp" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for server on port $PORT (up to 10 min) ..."
for i in $(seq 1 300); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: server exited early. Last 30 lines of $SERVER_LOG:"
    tail -30 "$SERVER_LOG"
    exit 1
  fi
  if curl -sf -o /dev/null -X POST "http://127.0.0.1:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL_PATH}\",\"prompt\":\"hi\",\"max_tokens\":1}"; then
    echo "Server ready after ~${i}s"
    break
  fi
  if [[ "$i" -eq 300 ]]; then
    echo "ERROR: server not ready after 10 min. Last 30 lines:"
    tail -30 "$SERVER_LOG"
    exit 1
  fi
  sleep 2
done

GIT_SHA="$(git rev-parse --short HEAD)"
echo "Running benchmark (log: $BENCH_LOG) ..."
"$VLLM_BIN" bench serve \
  --save-result \
  --result-dir "$RESULT_DIR/" \
  --result-filename "$(basename "$RESULT_JSON")" \
  --request-rate inf \
  --max-concurrency 1 \
  --metadata "git_sha=${GIT_SHA}" \
  --metadata "tensor_parallel_size=${TP}" \
  --metadata "attention_backend=FLASHINFER" \
  --metadata "load_format=dummy" \
  --metadata "moe_pattern=fork_sequential" \
  --backend vllm \
  --dataset-name random \
  --ignore-eos \
  --model "$MODEL_PATH" \
  --random-input-len 1000 \
  --random-output-len 1000 \
  --num-warmups 1 \
  --num-prompts 2 \
  2>&1 | tee "$BENCH_LOG"

echo
echo "=== Done ==="
echo "Proof:       $PROOF_FILE"
echo "Server log:  $SERVER_LOG"
echo "Bench log:   $BENCH_LOG"
echo "Results JSON: $RESULT_JSON"
echo
echo "Expected (with fix a68a391e5): TTFT ~69ms, tok/s ~208"
