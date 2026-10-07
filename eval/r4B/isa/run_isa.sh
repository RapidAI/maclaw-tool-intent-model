#!/bin/bash
R=/workspace/maclaw_reranker; D=$R/out/replace_audit/r4B/isa; cd /workspace/maclaw-qwen3only/corelib/embedding
for v in avx512 avx2 off; do
  echo "== $v"; MACLAW_Q8P_ISA=$v ZZ_VD_MODEL=$R/models/Qwen3-Embedding-0.6B-Q8_0.gguf ZZ_VD_IN=$D/texts.jsonl ZZ_VD_OUT=$D/v_$v $R/bin/vecdump.test -test.run TestZZVecDump -test.v -test.timeout 60m 2>&1 | grep -iE "isa|q8p|PASS|FAIL" | head -5
done
uptime
