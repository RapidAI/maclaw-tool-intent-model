#!/bin/bash
# Fair no-prefix baseline (section 8.4 config) with the FIXED harness (80bdef59): raw L2 (MACLAW_EMBED_PROMPTS=0, identity map),
# old head heads/head_mlp_gemma.json (tau=0.79), tau_s=0.90, independent 839. Waits for cold-start runs to finish (to not disturb them).
R=/workspace/maclaw_reranker
while ! grep -q COLDDONE $R/logs/run_go_cls.log; do sleep 10; done
export ZZ_GGUF=$R/models/embeddinggemma-300M-Q8_0.gguf ZZ_MODE=eval ZZ_L3=head MACLAW_INTENT_ANCHOR_CACHE_DIR=$R/out/anchor_cache_cls MACLAW_EMBED_PROMPTS=0
cd /workspace/maclaw/corelib/intent
echo "=== base_indep839_ts090 start $(date +%T)"
ZZ_HEAD=$R/heads/head_mlp_gemma.json ZZ_TAU_S=0.90 ZZ_IN=$R/data/go_in_indep839.jsonl ZZ_OUT=$R/out/go_cls_base_indep839_ts090.jsonl \
  $R/bin/intent_harness6.test -test.run TestZZLocalEval -test.v -test.timeout 60m > $R/logs/go_cls_base_indep839_ts090.log 2>&1
echo "=== base_indep839_ts090 done $(date +%T) exit=$?"
echo ALLDONE3
