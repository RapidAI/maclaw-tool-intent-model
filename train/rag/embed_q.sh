#!/bin/bash
A=/workspace/maclaw_reranker/out/replace_audit; M=/workspace/maclaw_reranker/models; cd $A/rag
for cfg in "q3 Qwen3-Embedding-0.6B-Q8_0.gguf query" "gm embeddinggemma-300M-Q8_0.gguf query" "q3 Qwen3-Embedding-0.6B-Q8_0.gguf none" "gm embeddinggemma-300M-Q8_0.gguf none" "q3 Qwen3-Embedding-0.6B-Q8_0.gguf qa"; do
 set -- $cfg; echo "== $1 $3 start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg)"
 $A/embdumpany.new -model $M/$2 -in queries_in.jsonl -out q_$1_$3.jsonl -role $3 -batch 32 2>&1 | tail -1
done; echo QDONE
