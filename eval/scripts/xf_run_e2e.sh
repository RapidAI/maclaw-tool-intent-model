#!/bin/bash
# Section 19: Go e2e (real ClassifyContext; L2 Qwen3 anchors + quantile map; local L3 = head via ZZ_L3=head), no network.
R=/workspace/maclaw_reranker
export ZZ_GGUF=$R/models/Qwen3-Embedding-0.6B-Q8_0.gguf ZZ_MODE=eval ZZ_L3=head MACLAW_INTENT_ANCHOR_CACHE_DIR=$R/out/anchor_cache_qwen3
cd /workspace/maclaw-final/corelib/intent
run() { echo "=== $1 start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  ZZ_HEAD=$R/heads/$2 ZZ_TAU_S=$3 ZZ_IN=$4 ZZ_OUT=$R/out/xf/go_$1.jsonl $R/bin/intent_harness_final.test -test.run TestZZLocalEval -test.v -test.timeout 90m > $R/logs/xf/go_$1.log 2>&1
  echo "=== $1 done $(date +%T) rc=$?"; }
run xf_indep839 head_mlp_qwen3go_xf.json 0.95 $R/data/go_in_indep839.jsonl
run xf_kimi150 head_mlp_qwen3go_xf.json 0.95 $R/data/go_in_GK_test_150.jsonl
run GK_kimi150 head_mlp_qwen3go_GK.json 0.95 $R/data/go_in_GK_test_150.jsonl
echo ALLDONE
