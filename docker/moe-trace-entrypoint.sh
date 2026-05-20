#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Entrypoint for the trtllm-moe-trace image. Mirrors the moe_trace.sh
# smoke-test runbook: launches trtllm-serve, waits for /health, fires a short
# aiperf burst, then summarizes the per-rank trace JSONLs.
#
# Knobs (env):
#   MODEL, TP, PORT                 -- serve args
#   CONCURRENCY, REQUEST_COUNT,
#   WARMUP_REQUEST_COUNT, ISL, OSL  -- aiperf args
#   MOE_TRACE_DIR                   -- where trace JSONLs land (default /traces)
#   SKIP_AIPERF=1                   -- skip the aiperf burst (server stays up
#                                       until container is stopped)

set -euo pipefail

mkdir -p "$MOE_TRACE_DIR"
rm -f "$MOE_TRACE_DIR"/moe_trace_rank*.jsonl

SERVER_LOG="${SERVER_LOG:-/tmp/trtllm_server.log}"

trtllm-serve "$MODEL" \
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

if [[ "${SKIP_AIPERF:-0}" == "1" ]]; then
  echo "SKIP_AIPERF=1, keeping server up; Ctrl-C to exit"
  wait "$SERVER_PID"
  exit 0
fi

aiperf profile \
  --model "$MODEL" \
  --tokenizer "$MODEL" \
  --service-kind openai \
  --endpoint-type chat \
  --endpoint /v1/chat/completions \
  --url "http://localhost:$PORT" \
  --concurrency "${CONCURRENCY:-4}" \
  --warmup-request-count "${WARMUP_REQUEST_COUNT:-2}" \
  --request-count "${REQUEST_COUNT:-20}" \
  --synthetic-input-tokens-mean "${ISL:-1024}" \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean "${OSL:-128}" \
  --output-tokens-stddev 0

echo ""
echo "--- trace files ---"
ls -lh "$MOE_TRACE_DIR"/moe_trace_rank*.jsonl 2>/dev/null || echo "no trace files produced"
echo ""
echo "--- event counts per kind, per rank ---"
for f in "$MOE_TRACE_DIR"/moe_trace_rank*.jsonl; do
  [[ -f "$f" ]] || continue
  echo "==> $(basename "$f")"
  python -c "
import json, collections
c = collections.Counter()
for line in open('$f'):
    try: c[json.loads(line)['kind']] += 1
    except Exception: pass
for k, v in sorted(c.items()): print(f'    {k:12s} {v}')
"
done
echo ""
echo "--- first event of each kind (rank 0) ---"
python -c "
import json, glob
files = sorted(glob.glob('$MOE_TRACE_DIR/moe_trace_rank0.jsonl'))
if not files: raise SystemExit
seen = set()
for line in open(files[0]):
    try: e = json.loads(line)
    except Exception: continue
    if e['kind'] in seen: continue
    seen.add(e['kind'])
    print(json.dumps(e, indent=2))
" 2>/dev/null || true
