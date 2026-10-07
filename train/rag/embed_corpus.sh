#!/bin/bash
A=/workspace/maclaw_reranker/out/replace_audit; M=/workspace/maclaw_reranker/models; cd $A/rag
for cfg in "q3 Qwen3-Embedding-0.6B-Q8_0.gguf document" "gm embeddinggemma-300M-Q8_0.gguf document" "gm embeddinggemma-300M-Q8_0.gguf none"; do
 set -- $cfg; echo "== $1 $3 start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg)"
 $A/embdumpany -model $M/$2 -in corpus_in.jsonl -out doc_$1_$3.jsonl -role $3 -batch 16 2>&1 | tail -2
done; echo CORPUSDONE
