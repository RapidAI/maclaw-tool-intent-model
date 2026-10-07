#!/bin/bash
# A_confirm ONE-SHOT (only after freeze). Verifies sha, builds {id,text} input, runs Qwen3 (frozen head, production harness, fix 1) and Gemma (scripts/run_go_cls.sh config old339_GK_ts095) sequentially.
set -e
R=/workspace/maclaw_reranker; A=$R/out/replace_audit; C=$A/confirm_r4; O=$A/aconf; H=$1
echo "8f48de2a81edd82b960922751e592ba9472292995006182d6b6e801cbff61197  $C/A_confirm.jsonl" | sha256sum -c -
python3 -c "
import json
with open('$O/aconf_in.jsonl','w') as f:
    for l in open('$C/A_confirm.jsonl'): r=json.loads(l); f.write(json.dumps({'id':r['id'],'text':r['text']},ensure_ascii=False)+'\n')"
echo "qwen3 start $(date +%T) head=$(sha256sum $H|cut -c1-12)"
(export ZZ_GGUF=$R/models/Qwen3-Embedding-0.6B-Q8_0.gguf ZZ_MODE=eval ZZ_L3=localhead MACLAW_INTENT_ANCHOR_CACHE_DIR=$R/out/anchor_cache_qwen3 MACLAW_INTENT_HEAD_HINT_FIX=1
 cd /workspace/maclaw-qwen3only/corelib/intent
 ZZ_HEAD=$H ZZ_TAU_S=$(python3 -c "import json;print(json.load(open('$H'))['tau_s'])") ZZ_IN=$O/aconf_in.jsonl ZZ_OUT=$O/qwen3.jsonl $R/bin/intent_harness_hintfix.test -test.run TestZZLocalEval -test.v -test.timeout 90m > $O/qwen3.log 2>&1)
echo "gemma start $(date +%T)"
(export ZZ_GGUF=$R/models/embeddinggemma-300M-Q8_0.gguf ZZ_MODE=eval ZZ_L3=head MACLAW_INTENT_ANCHOR_CACHE_DIR=$R/out/anchor_cache_cls
 cd /workspace/maclaw/corelib/intent
 ZZ_HEAD=$R/heads/head_mlp_gemmacls_GK.json ZZ_TAU_S=0.95 ZZ_IN=$O/aconf_in.jsonl ZZ_OUT=$O/gemma.jsonl $R/bin/intent_harness6.test -test.run TestZZLocalEval -test.v -test.timeout 90m > $O/gemma.log 2>&1)
echo "done $(date +%T)"
