#!/usr/bin/env bash
# Round 2: Qwen3-only single-model proof (ZZ_EXPECT_SINGLE_GGUF=1), reverse order.
set -euo pipefail
ROOT=/workspace/maclaw_reranker; BIN=$ROOT/bin/tool_pipeline.test; OUT=$ROOT/out/e2e_pipeline_cmp; LOG=$ROOT/logs
QWEN3=$ROOT/models/Qwen3-Embedding-0.6B-Q8_0.gguf; HEAD_Q=$ROOT/heads/head_mlp_qwen3go_GK.json
run() { local name=$1; local isa=$2; local mode=$3; local in=$4; shift 4
  local home=$OUT/home_r2_${isa}; mkdir -p $home/.maclaw/cache; rm -f $home/.maclaw/cache/tool_embeddings.gob
  echo "=== $name load=$(cut -d' ' -f1-3 /proc/loadavg) $(date +%T)" | tee -a $LOG/e2e_pipeline_cmp.log
  env HOME=$home CGO_ENABLED=0 MACLAW_Q8P_ISA=$isa ZZ_EXPECT_SINGLE_GGUF=1 ZZ_MODE=$mode ZZ_IN=$in \
    ZZ_OUT=$OUT/$name.jsonl ZZ_GGUF=$QWEN3 ZZ_TOOLS=$ROOT/data/core_tool_defs.json "$@" \
    $BIN -test.run '^TestZZToolRetrieval$' -test.v > $LOG/pipe_$name.log 2>&1 || { echo FAIL $name; tail -20 $LOG/pipe_$name.log; exit 1; }
  rg "ZZ probe|ZZ .* done" $LOG/pipe_$name.log | tee -a $LOG/e2e_pipeline_cmp.log; }
K=$ROOT/data/go_in_GK_test_150.jsonl; T=$OUT/tool_in_test.jsonl
run combined_qwen3_avx2_k150_r2   avx2   combined $K ZZ_L3=localhead ZZ_TAU_S=0.95 ZZ_HEAD=$HEAD_Q
run combined_qwen3_avx512_k150_r2 avx512 combined $K ZZ_L3=localhead ZZ_TAU_S=0.95 ZZ_HEAD=$HEAD_Q
run trackA_qwen3_avx2_r2   avx2   trackA $T
run trackA_qwen3_avx512_r2 avx512 trackA $T
echo "R2 DONE load=$(cut -d' ' -f1-3 /proc/loadavg) $(date +%T)" | tee -a $LOG/e2e_pipeline_cmp.log
