#!/usr/bin/env bash
# MoE trace smoke test for TensorRT-LLM (PyTorch flow).
#
# Brings up trtllm-serve with MOE_TRACE_ENABLE=1, runs a short aiperf
# burst against it, then prints where the per-rank JSONL traces landed.
#
# Target: GB300 NVL72, 4 GPUs (~288 GB HBM3e each).
# Model:  Qwen/Qwen3.5-397B-A17B.
# A2A:    NVLinkOneSided (default for the PyTorch flow on GB).
#
# The wheel-overlay install (wheel first on tensorrt_llm.__path__ so the
# bundled libs load; source tree appended via .pth) means our edits to
# files that ALSO exist in the wheel are not live by default. Step 2
# replaces each modified file's wheel copy with a symlink to the source
# tree.

set -euo pipefail

ENGINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ENGINE_DIR"

VENV_DIR="$ENGINE_DIR/.venv"
TRACE_DIR="${MOE_TRACE_DIR:-$ENGINE_DIR/moe_traces}"
SERVER_LOG="$ENGINE_DIR/trtllm_server.log"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B}"
TP="${TP:-4}"
PORT="${PORT:-30000}"
TRTLLM_WHEEL_VERSION="${TRTLLM_WHEEL_VERSION:-1.3.0rc5.post2}"

# ---------------------------------------------------------------- 1) venv
if [[ ! -d "$VENV_DIR" ]]; then
  python3.12 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
  "$VENV_DIR/bin/pip" install "tensorrt-llm==$TRTLLM_WHEEL_VERSION" \
    --extra-index-url https://pypi.nvidia.com
  # Wheel's declared deps miss cu13 cublas at this version; install explicitly.
  "$VENV_DIR/bin/pip" install nvidia-cublas \
    --extra-index-url https://pypi.nvidia.com
fi

SP="$VENV_DIR/lib/python3.12/site-packages"

# ------------------------------------------- 2) shadow wheel with source-tree
# Replace each modified file in site-packages with a symlink to the
# checked-out copy so the instrumentation is live at import time.
MODIFIED_FILES=(
  "tensorrt_llm/moe_trace_logger.py"
  "tensorrt_llm/moe_trace_logger_phase.py"
  "tensorrt_llm/_torch/modules/fused_moe/communication/deep_ep.py"
  "tensorrt_llm/_torch/modules/fused_moe/communication/deep_ep_low_latency.py"
  "tensorrt_llm/_torch/modules/fused_moe/communication/nvlink_one_sided.py"
  "tensorrt_llm/_torch/modules/fused_moe/communication/nvlink_two_sided.py"
  "tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py"
  "tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py"
  "tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py"
  "tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py"
  "tensorrt_llm/_torch/modules/fused_moe/fused_moe_wide_ep.py"
)
for rel in "${MODIFIED_FILES[@]}"; do
  src="$ENGINE_DIR/$rel"
  dst="$SP/$rel"
  if [[ ! -f "$src" ]]; then
    echo "WARN: source missing, skip: $src" >&2
    continue
  fi
  mkdir -p "$(dirname "$dst")"
  if [[ -L "$dst" ]] && [[ "$(readlink -f "$dst")" == "$(readlink -f "$src")" ]]; then
    continue
  fi
  if [[ -e "$dst" ]] && [[ ! -L "$dst" ]]; then
    cp -n "$dst" "$dst.wheel.bak"
  fi
  ln -sfn "$src" "$dst"
done
echo "shadowed ${#MODIFIED_FILES[@]} files into $SP"

# ----------------------------------------------------- 3) aiperf for testing
"$VENV_DIR/bin/pip" install --quiet aiperf

# ------------------------------------------------------------- 4) tracer env
mkdir -p "$TRACE_DIR"
rm -f "$TRACE_DIR"/moe_trace_rank*.jsonl
export MOE_TRACE_ENABLE=1
export MOE_TRACE_DIR="$TRACE_DIR"
export MOE_TRACE_FLUSH_EVERY=512

# --------------------------------------------------------- 5) launch server
"$VENV_DIR/bin/trtllm-serve" "$MODEL" \
  --backend pytorch \
  --tp_size "$TP" \
  --host 0.0.0.0 \
  --port "$PORT" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "trtllm-serve launched (PID=$SERVER_PID), log: $SERVER_LOG"

cleanup() {
  echo "shutting down trtllm-serve (PID $SERVER_PID)..."
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ----------------------------------------------------- 6) wait for readiness
echo "waiting for /health on :$PORT (up to 30 min for 397B load)..."
for _ in $(seq 1 360); do
  if curl -s --fail "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "server ready"
    break
  fi
  sleep 5
done
curl -s --fail "http://localhost:$PORT/health" >/dev/null || {
  echo "ERROR: server did not become ready; tail of log:"
  tail -80 "$SERVER_LOG"
  exit 1
}

# -------------------------------------------------------------- 7) aiperf
"$VENV_DIR/bin/aiperf" profile \
  --model "$MODEL" \
  --tokenizer "$MODEL" \
  --service-kind openai \
  --endpoint-type chat \
  --endpoint /v1/chat/completions \
  --url "http://localhost:$PORT" \
  --concurrency 4 \
  --warmup-request-count 2 \
  --request-count 20 \
  --synthetic-input-tokens-mean 1024 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 \
  --output-tokens-stddev 0

# --------------------------------------------------- 8) summarize trace output
echo ""
echo "--- trace files ---"
ls -lh "$TRACE_DIR"/moe_trace_rank*.jsonl 2>/dev/null || echo "no trace files produced"
echo ""
echo "--- event counts per kind, per rank ---"
for f in "$TRACE_DIR"/moe_trace_rank*.jsonl; do
  [[ -f "$f" ]] || continue
  echo "==> $(basename "$f")"
  "$VENV_DIR/bin/python" -c "
import json, sys, collections
c = collections.Counter()
for line in open('$f'):
    try: c[json.loads(line)['kind']] += 1
    except Exception: pass
for k, v in sorted(c.items()): print(f'    {k:12s} {v}')
"
done
echo ""
echo "--- first event of each kind (rank 0) ---"
"$VENV_DIR/bin/python" -c "
import json, glob
seen = set()
for line in open(sorted(glob.glob('$TRACE_DIR/moe_trace_rank0.jsonl'))[0]):
    try: e = json.loads(line)
    except Exception: continue
    if e['kind'] in seen: continue
    seen.add(e['kind'])
    print(json.dumps(e, indent=2))
" 2>/dev/null || true
