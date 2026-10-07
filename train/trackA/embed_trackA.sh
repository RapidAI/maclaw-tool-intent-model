#!/bin/bash
A=/workspace/maclaw_reranker/out/replace_audit; M=/workspace/maclaw_reranker/models; cd $A/trackA
while ! grep -q CORPUSDONE $A/rag/embed_corpus.log; do sleep 5; done
Q=/workspace/maclaw_reranker/out/qwen3_toolhead/in_all.jsonl
run(){ echo "== $1 $3 $4 start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg)"; $A/embdumpany -model $M/$2 -in $3 -out $5 -role $4 -batch 32 2>&1 | tail -1; }
for r in none document classification query; do
  run q3 Qwen3-Embedding-0.6B-Q8_0.gguf tooltext.jsonl $r doc_q3_$r.jsonl
  run gm embeddinggemma-300M-Q8_0.gguf tooltext.jsonl $r doc_gm_$r.jsonl
done
for r in none classification query; do run gm embeddinggemma-300M-Q8_0.gguf $Q $r q_gm_$r.jsonl; done
for r in none query; do run q3 Qwen3-Embedding-0.6B-Q8_0.gguf $Q $r q_q3_$r.jsonl; done
echo TRACKADONE
