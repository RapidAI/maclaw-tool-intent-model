#!/bin/bash
# usage: run_hintfix.sh <head.json> <tag> <fix 0|1> <set: old339|indep839> [tau_s override]
# Production LoadLocalHead + LocalTreeFunc (+ LocalHeadProbe) via ZZ_L3=localhead, harness built from local/qwen3-only.
R=/workspace/maclaw_reranker; A=$R/out/replace_audit
H=$1; TAG=$2; FIX=$3; SET=$4
case $SET in old339) IN=$R/data/intent_test_texts.jsonl;; indep839) IN=$R/data/go_in_indep839.jsonl;; esac
TS=${5:-$(python3 -c "import json;print(json.load(open('$H'))['tau_s'])")}
export ZZ_GGUF=$R/models/Qwen3-Embedding-0.6B-Q8_0.gguf ZZ_MODE=eval ZZ_L3=localhead MACLAW_INTENT_ANCHOR_CACHE_DIR=$R/out/anchor_cache_qwen3 MACLAW_INTENT_HEAD_HINT_FIX=$FIX
cd /workspace/maclaw-qwen3only/corelib/intent
OUT=$A/hintfix/go_${TAG}_fix${FIX}_${SET}
echo "start $SET fix=$FIX $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) tau_s=$TS head=$(sha256sum $H|cut -c1-12)"
ZZ_HEAD=$H ZZ_TAU_S=$TS ZZ_IN=$IN ZZ_OUT=$OUT.jsonl $R/bin/${HARNESS:-intent_harness_hintfix.test} -test.run TestZZLocalEval -test.v -test.timeout 90m > $OUT.log 2>&1
echo "done $SET fix=$FIX $(date +%T) rc=$? load=$(cut -d' ' -f1-3 /proc/loadavg)"
