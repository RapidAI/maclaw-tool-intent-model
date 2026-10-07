#!/bin/bash
# cold start: anchor warmup time with prompt+no cache, prompt+disk cache, raw (no prompt) + no cache
R=/workspace/maclaw_reranker; D=$R/out/anchor_cache_coldtest; rm -rf $D
cd /workspace/maclaw/corelib/intent
for i in 1 2 3; do
 for mode in prompt_nocache prompt_cache raw_nocache; do
  case $mode in
   prompt_nocache) E="MACLAW_INTENT_ANCHOR_CACHE=0";;
   prompt_cache) E="MACLAW_INTENT_ANCHOR_CACHE_DIR=$D";;
   raw_nocache) E="MACLAW_INTENT_ANCHOR_CACHE=0 MACLAW_EMBED_PROMPTS=0";;
  esac
  echo -n "run$i $mode: load=$(cut -d' ' -f1 /proc/loadavg) "
  env $E ZZ_GGUF=$R/models/embeddinggemma-300M-Q8_0.gguf ZZ_MODE=eval ZZ_L3=none ZZ_IN=$R/data/one.jsonl ZZ_OUT=/tmp/cold.jsonl \
    $R/bin/intent_harness6.test -test.run TestZZLocalEval -test.timeout 10m 2>&1 | grep -E "embedder load|anchors warm|disk cache" | tr '\n' ' '; echo
 done
done
