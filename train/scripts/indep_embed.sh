#!/bin/bash
# Embed data/indep_candidates.jsonl with the production Go GemmaEmbedder (ZZ_MODE=embed).
R=/workspace/maclaw_reranker
cd /workspace/maclaw/corelib/intent && ZZ_MODE=embed ZZ_GGUF=$R/models/embeddinggemma-300M-Q8_0.gguf ZZ_IN=$R/data/indep_candidates.jsonl ZZ_OUT=$R/out/emb_indep_gemma.jsonl \
  $R/bin/intent_harness4.test -test.run TestZZLocalEval -test.timeout 30m
