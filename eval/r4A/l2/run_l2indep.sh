#!/bin/bash
# L2 scores (top1, runner-up) on indep839 (selection-allowed set), head r3W5F prod.
R=/workspace/maclaw_reranker; A=$R/out/replace_audit
export ZZ_GGUF=$R/models/Qwen3-Embedding-0.6B-Q8_0.gguf ZZ_MODE=eval ZZ_L3=localhead MACLAW_INTENT_ANCHOR_CACHE_DIR=$R/out/anchor_cache_qwen3 MACLAW_INTENT_HEAD_HINT_FIX=1
cd /workspace/maclaw-qwen3only/corelib/intent
echo "start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg)"
ZZ_HEAD=$R/heads/head_mlp_qwen3go_xf.json ZZ_TAU_S=0.97 ZZ_IN=$R/data/go_in_indep839.jsonl ZZ_OUT=$A/r4A/l2/l2_indep.jsonl $R/bin/intent_harness_r4l2.test -test.run TestZZLocalEval -test.timeout 120m > $A/r4A/l2/l2_indep.log 2>&1
echo "done $(date +%T) rc=$? load=$(cut -d' ' -f1-3 /proc/loadavg)"
