#!/bin/bash
# Track A harness runs (Qwen3-only and Gemma reference). Sequential to limit CPU contention.
cd /workspace/maclaw_reranker
B=bin/tool_harness_qwen3.test; D=out/qwen3_toolhead
run(){ name=$1; shift; echo "== $name start $(date +%T) load $(cut -d' ' -f1 /proc/loadavg)"; env "$@" ZZ_TOOLS=data/core_tool_defs.json $B -test.run TestZZToolRetrieval -test.count=1 2>&1 | tail -3; echo "== $name end $(date +%T)"; }
run q3cls  ZZ_EMBEDDER=qwen3 ZZ_GGUF=models/Qwen3-Embedding-0.6B-Q8_0.gguf ZZ_QROLE=classification ZZ_IN=$D/in_all.jsonl ZZ_OUT=$D/trackA_q3cls.jsonl ZZ_QVEC_OUT=$D/qvec_q3cls_final.jsonl
run q3none ZZ_EMBEDDER=qwen3 ZZ_GGUF=models/Qwen3-Embedding-0.6B-Q8_0.gguf ZZ_QROLE= ZZ_IN=$D/in_test.jsonl ZZ_OUT=$D/trackA_q3none.jsonl
run q3query ZZ_EMBEDDER=qwen3 ZZ_GGUF=models/Qwen3-Embedding-0.6B-Q8_0.gguf ZZ_QROLE=query ZZ_IN=$D/in_test.jsonl ZZ_OUT=$D/trackA_q3query.jsonl
run gemma  ZZ_GGUF=models/embeddinggemma-300M-Q8_0.gguf ZZ_IN=$D/in_indep.jsonl ZZ_OUT=$D/trackA_gemma_indep.jsonl
echo ALLDONE
