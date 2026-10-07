#!/bin/bash
# Section 9: Go end-to-end with classification-prompt L2 (production change) + cls-prompt head as local L3. No network.
R=/workspace/maclaw_reranker
export ZZ_GGUF=$R/models/embeddinggemma-300M-Q8_0.gguf ZZ_MODE=eval ZZ_L3=head
export MACLAW_INTENT_ANCHOR_CACHE_DIR=$R/out/anchor_cache_cls
cd /workspace/maclaw/corelib/intent
run() { # name head tau_s input
  echo "=== $1 start $(date +%T)"
  ZZ_HEAD=$R/heads/$2 ZZ_TAU_S=$3 ZZ_IN=$4 ZZ_OUT=$R/out/go_cls_$1.jsonl \
    $R/bin/intent_harness6.test -test.run TestZZLocalEval -test.v -test.timeout 60m > $R/logs/go_cls_$1.log 2>&1
  echo "=== $1 done $(date +%T) exit=$? $(grep -h 'anchors warm' $R/logs/go_cls_$1.log)"
}
rm -rf $MACLAW_INTENT_ANCHOR_CACHE_DIR; mkdir -p $R/out/go_cls_buggy; mv $R/out/go_cls_*.jsonl $R/out/go_cls_buggy/ 2>/dev/null
run old339_GK head_mlp_gemmacls_GK.json 0.99 $R/data/intent_test_texts.jsonl
run kimi_GK head_mlp_gemmacls_GK.json 0.99 $R/data/go_in_GK_test.jsonl
run gemini_KG head_mlp_gemmacls_KG.json 0.99 $R/data/go_in_KG_test.jsonl
run amb95_GK head_mlp_gemmacls_GK.json 0.99 $R/data/indep_ambiguous.jsonl
run kimi_GK_ts095 head_mlp_gemmacls_GK.json 0.95 $R/data/go_in_GK_test.jsonl
run gemini_KG_ts095 head_mlp_gemmacls_KG.json 0.95 $R/data/go_in_KG_test.jsonl
run old339_GK_ts095 head_mlp_gemmacls_GK.json 0.95 $R/data/intent_test_texts.jsonl
run amb95_GK_ts095 head_mlp_gemmacls_GK.json 0.95 $R/data/indep_ambiguous.jsonl
# ablation: no prompt anywhere (MACLAW_EMBED_PROMPTS=0 -> raw L2 + identity calibration), raw-space heads, same recipe
export MACLAW_EMBED_PROMPTS=0
run raw_kimi_GK_ts095 head_mlp_gemmaraw_GK.json 0.95 $R/data/go_in_GK_test.jsonl
run raw_gemini_KG_ts095 head_mlp_gemmaraw_KG.json 0.95 $R/data/go_in_KG_test.jsonl
unset MACLAW_EMBED_PROMPTS
echo ALLDONE $(date +%T)
bash $R/scripts/coldstart.sh > $R/logs/coldstart.log 2>&1
echo COLDDONE $(date +%T)
