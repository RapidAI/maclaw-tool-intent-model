#!/bin/bash
R=/workspace/maclaw_reranker; A=$R/out/replace_audit
export ZZ_GGUF=$R/models/Qwen3-Embedding-0.6B-Q8_0.gguf ZZ_MODE=eval ZZ_L3=head MACLAW_INTENT_ANCHOR_CACHE_DIR=$R/out/anchor_cache_qwen3
cd /workspace/maclaw-final/corelib/intent
echo "start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg)"
ZZ_HEAD=$A/head_mlp_qwen3go_xf.snapshot.json ZZ_TAU_S=0.95 ZZ_IN=$R/data/intent_test_texts.jsonl ZZ_OUT=$A/go_xf_old339.jsonl $R/bin/intent_harness_final.test -test.run TestZZLocalEval -test.v -test.timeout 60m > $A/go_xf_old339.log 2>&1
echo "done $(date +%T) rc=$? load=$(cut -d' ' -f1-3 /proc/loadavg)"
