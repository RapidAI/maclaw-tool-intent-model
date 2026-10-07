#!/bin/bash
# old-339 with the FINAL cross-family head (r2bF, sha f1e101ad…), head's own tau/tau_s (0.87/0.96, chosen by the xf worker on pooled LOFO OOF xf data — no old-339).
R=/workspace/maclaw_reranker; A=$R/out/replace_audit
export ZZ_GGUF=$R/models/Qwen3-Embedding-0.6B-Q8_0.gguf ZZ_MODE=eval ZZ_L3=head MACLAW_INTENT_ANCHOR_CACHE_DIR=$R/out/anchor_cache_qwen3
cd /workspace/maclaw-final/corelib/intent
H=$A/head_mlp_qwen3go_xf.final.snapshot.json
TS=$(python3 -c "import json;print(json.load(open('$H'))['tau_s'])")
echo "start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) tau_s=$TS"
ZZ_HEAD=$H ZZ_TAU_S=$TS ZZ_IN=$R/data/intent_test_texts.jsonl ZZ_OUT=$A/go_xfF_old339.jsonl $R/bin/intent_harness_final.test -test.run TestZZLocalEval -test.v -test.timeout 60m > $A/go_xfF_old339.log 2>&1
echo "done $(date +%T) rc=$? load=$(cut -d' ' -f1-3 /proc/loadavg)"
