#!/bin/bash
R=/workspace/maclaw_reranker
export ZZ_GGUF=$R/models/Qwen3-Embedding-0.6B-Q8_0.gguf ZZ_MODE=eval ZZ_L3=head MACLAW_INTENT_ANCHOR_CACHE_DIR=$R/out/anchor_cache_qwen3
cd /workspace/maclaw-final/corelib/intent
TS=$(python3 -c "import json;print(json.load(open('$R/heads/head_mlp_qwen3go_xf_r3W5F.json'))['tau_s'])")
run(){ echo "=== $1 $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"; ZZ_HEAD=$R/heads/head_mlp_qwen3go_xf_r3W5F.json ZZ_TAU_S=$TS ZZ_IN=$2 ZZ_OUT=$R/out/xf/go_$1.jsonl $R/bin/intent_harness_final.test -test.run TestZZLocalEval -test.v -test.timeout 90m > $R/logs/xf/go_$1.log 2>&1; echo "done $1 rc=$? $(date +%T)"; }
run xf3W_kimi150 $R/data/go_in_GK_test_150.jsonl
run xf3W_indep839 $R/data/go_in_indep839.jsonl
run xf3W_old339 $R/data/intent_test_texts.jsonl
echo ALLDONE
