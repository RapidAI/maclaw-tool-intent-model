#!/bin/bash
R=/workspace/maclaw_reranker
while ! grep -q COLDDONE $R/logs/run_go_cls.log; do sleep 10; done
export ZZ_GGUF=$R/models/embeddinggemma-300M-Q8_0.gguf ZZ_MODE=eval ZZ_L3=head MACLAW_INTENT_ANCHOR_CACHE_DIR=$R/out/anchor_cache_cls
cd /workspace/maclaw/corelib/intent
run() { echo "=== $1 start $(date +%T)"; ZZ_HEAD=$R/heads/$2 ZZ_TAU_S=$3 ZZ_IN=$4 ZZ_OUT=$R/out/go_cls_$1.jsonl \
  $R/bin/intent_harness6.test -test.run TestZZLocalEval -test.v -test.timeout 60m > $R/logs/go_cls_$1.log 2>&1; echo "=== $1 done $(date +%T)"; }
run kimi_GK_t90_ts095 head_mlp_gemmacls_GK_t90.json 0.95 $R/data/go_in_GK_test.jsonl
run gemini_KG_t90_ts095 head_mlp_gemmacls_KG_t90.json 0.95 $R/data/go_in_KG_test.jsonl
run old339_GK_t90_ts095 head_mlp_gemmacls_GK_t90.json 0.95 $R/data/intent_test_texts.jsonl
echo ALLDONE2 $(date +%T)
