#!/bin/bash
cd /workspace/maclaw-replace
for r in qd sim; do for tag in qwen3 gemma; do f=/workspace/maclaw_reranker/models/Qwen3-Embedding-0.6B-Q8_0.gguf; [ $tag = gemma ] && f=/workspace/maclaw_reranker/models/embeddinggemma-300M-Q8_0.gguf
ZZ_CAP3_ROLES=$r ZZ_CAP3_MODEL=$f ZZ_CAP3_DIR=/workspace/maclaw_reranker/out/replace_audit/cap3 ZZ_CAP3_TAG=$tag CGO_ENABLED=0 go test ./corelib/agent -run TestZZCap3Dump -count=1 -timeout 2h -v 2>&1 | grep -v TopicDetector | tail -2; uptime; done; done
