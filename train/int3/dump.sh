#!/bin/bash
# usage: dump.sh <texts.jsonl> <outprefix>   (Qwen3 Q8_0, pure Go, CGO_ENABLED=0 build)
R=/workspace/maclaw_reranker
echo "start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) n=$(wc -l < $1)"
cd /workspace/maclaw-qwen3only/corelib/embedding
ZZ_VD_MODEL=$R/models/Qwen3-Embedding-0.6B-Q8_0.gguf ZZ_VD_IN=$1 ZZ_VD_OUT=$2 $R/bin/vecdump.test -test.run TestZZVecDump -test.v -test.timeout 120m
echo "done $(date +%T) rc=$? load=$(cut -d' ' -f1-3 /proc/loadavg)"
