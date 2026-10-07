#!/usr/bin/env bash
# End-to-end intent + tool routing latency matrix (local only).
set -euo pipefail
ROOT=/workspace/maclaw_reranker
BIN="$ROOT/bin/tool_pipeline.test"
OUT="$ROOT/out/e2e_pipeline_cmp"
LOG="$ROOT/logs"
TOOLS="$ROOT/data/core_tool_defs.json"
TOOL_IN="$OUT/tool_in_test.jsonl"
KIMI150="$ROOT/data/go_in_GK_test_150.jsonl"
GEMMA="$ROOT/models/embeddinggemma-300M-Q8_0.gguf"
QWEN3="$ROOT/models/Qwen3-Embedding-0.6B-Q8_0.gguf"
HEAD_G="$ROOT/heads/head_mlp_gemmacls_GK.json"
HEAD_Q="$ROOT/heads/head_mlp_qwen3go_GK.json"
TOOL_HEAD="$ROOT/heads/tool_head_ovr_gemma.json"

mkdir -p "$OUT" "$LOG"
# tool_in from tool_eval test split
if [[ ! -f "$TOOL_IN" ]]; then
  python3 -c "
import json
out=open('$TOOL_IN','w')
for l in open('$ROOT/data/tool_eval.jsonl'):
  r=json.loads(l)
  if r.get('split')=='test':
    out.write(json.dumps({'id':r['id'],'text':r['text']},ensure_ascii=False)+'\n')
print('wrote', '$TOOL_IN')
"
fi

loadavg() { cut -d' ' -f1-3 /proc/loadavg; }
rss_note() { echo "load=$(loadavg) ts=$(date '+%F %T %z')"; }

run_one() {
  local name="$1"; shift
  echo "=== $name $(rss_note) ===" | tee -a "$LOG/e2e_pipeline_cmp.log"
  # isolate tool embedding disk cache per config (ISA/model)
  local home="$OUT/home_$name"
  mkdir -p "$home/.maclaw/cache"
  rm -f "$home/.maclaw/cache/tool_embeddings.gob"
  env HOME="$home" CGO_ENABLED=0 "$@" \
    ZZ_TOOLS="$TOOLS" \
    "$BIN" -test.run '^TestZZToolRetrieval$' -test.v \
    >"$LOG/pipe_${name}.log" 2>&1 || { echo "FAIL $name"; tail -30 "$LOG/pipe_${name}.log"; return 1; }
  echo "OK $name $(rss_note)" | tee -a "$LOG/e2e_pipeline_cmp.log"
  # capture trailing ZZ done line
  rg "ZZ .* done|ZZ embedder|ZZ tool cache|ZZ intent ready|rss=" "$LOG/pipe_${name}.log" | tee -a "$LOG/e2e_pipeline_cmp.log"
}

# ---- Tool-only Track A (tool_eval test n=262) ----
run_one trackA_gemma \
  ZZ_MODE=trackA ZZ_IN="$TOOL_IN" ZZ_OUT="$OUT/trackA_gemma.jsonl" ZZ_GGUF="$GEMMA"

run_one trackA_qwen3_avx512 \
  MACLAW_Q8P_ISA=avx512 \
  ZZ_MODE=trackA ZZ_IN="$TOOL_IN" ZZ_OUT="$OUT/trackA_qwen3_avx512.jsonl" ZZ_GGUF="$QWEN3"

run_one trackA_qwen3_avx2 \
  MACLAW_Q8P_ISA=avx2 \
  ZZ_MODE=trackA ZZ_IN="$TOOL_IN" ZZ_OUT="$OUT/trackA_qwen3_avx2.jsonl" ZZ_GGUF="$QWEN3"

# ---- Gemma tool head only ----
run_one toolhead_gemma \
  ZZ_MODE=toolhead ZZ_TOOL_HEAD="$TOOL_HEAD" \
  ZZ_IN="$TOOL_IN" ZZ_OUT="$OUT/toolhead_gemma.jsonl" ZZ_GGUF="$GEMMA"

# ---- Combined intent→tool on Kimi 150 ----
run_one combined_gemma_k150 \
  ZZ_MODE=combined ZZ_L3=localhead ZZ_TAU_S=0.95 \
  ZZ_HEAD="$HEAD_G" ZZ_TOOL_HEAD="$TOOL_HEAD" \
  ZZ_IN="$KIMI150" ZZ_OUT="$OUT/combined_gemma_k150.jsonl" ZZ_GGUF="$GEMMA"

run_one combined_qwen3_avx512_k150 \
  MACLAW_Q8P_ISA=avx512 \
  ZZ_MODE=combined ZZ_L3=localhead ZZ_TAU_S=0.95 \
  ZZ_HEAD="$HEAD_Q" \
  ZZ_IN="$KIMI150" ZZ_OUT="$OUT/combined_qwen3_avx512_k150.jsonl" ZZ_GGUF="$QWEN3"

run_one combined_qwen3_avx2_k150 \
  MACLAW_Q8P_ISA=avx2 \
  ZZ_MODE=combined ZZ_L3=localhead ZZ_TAU_S=0.95 \
  ZZ_HEAD="$HEAD_Q" \
  ZZ_IN="$KIMI150" ZZ_OUT="$OUT/combined_qwen3_avx2_k150.jsonl" ZZ_GGUF="$QWEN3"

echo "ALL DONE $(rss_note)" | tee -a "$LOG/e2e_pipeline_cmp.log"
