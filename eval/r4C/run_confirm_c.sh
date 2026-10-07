#!/bin/bash
# One-shot C_confirm signals: Gemma production (bm25/bigram/cos via cap3 harness) + Qwen3 CT3 task vectors (vecdump).
R=/workspace/maclaw_reranker; D=$R/out/replace_audit/r4C
cd /workspace/maclaw-qwen3only/corelib/embedding
ZZ_VD_MODEL=$R/models/Qwen3-Embedding-0.6B-Q8_0.gguf ZZ_VD_IN=$D/conf_c_texts.jsonl ZZ_VD_OUT=$D/vconf_c $R/bin/vecdump.test -test.run TestZZVecDump -test.timeout 60m | tail -1
cd /workspace/maclaw-replace
ZZ_CAP3_MODEL=$R/models/embeddinggemma-300M-Q8_0.gguf ZZ_CAP3_DIR=$D/confirm_dir ZZ_CAP3_TAG=gemma CGO_ENABLED=0 GOFLAGS=-p=2 go test ./corelib/agent -run TestZZCap3Dump -count=1 -timeout 2h -v 2>&1 | grep -v TopicDetector | tail -3
uptime
