#!/bin/bash
# Embed hard_train.jsonl with Qwen3 pure-Go (RoleClassification via plain Embed of classification-prompted text? 
# Intent head uses RoleClassification. The zz_vecdump uses plain Embed.
# For intent head training, existing emb_*.jsonl were dumped with RoleClassification.
# Check zz_vecdump / how xf embeddings were made.
set -e
R=/workspace/maclaw_reranker; D=$R/out/replace_audit/r4A
# Use the same dump harness as intent: build texts with classification prompt if needed.
python3 - <<'PY'
import json,os
R='/workspace/maclaw_reranker'; D=R+'/out/replace_audit/r4A'
# Intent RoleClassification prompt from qwen3_prompt.go: "Classify the intent of this assistant user request"
INST="Classify the intent of this assistant user request"
rows=[json.loads(l) for l in open(f'{D}/hard_train.jsonl')]
with open(f'{D}/hard_texts.jsonl','w') as f:
    for r in rows:
        f.write(json.dumps({'k':r['id'],'t':f"Instruct: {INST}\nQuery: {r['text']}"},ensure_ascii=False)+'\n')
print(len(rows),'texts')
PY
cd /workspace/maclaw-qwen3only/corelib/embedding
ZZ_VD_MODEL=$R/models/Qwen3-Embedding-0.6B-Q8_0.gguf ZZ_VD_IN=$D/hard_texts.jsonl ZZ_VD_OUT=$D/vec_hard \
  $R/bin/vecdump.test -test.run TestZZVecDump -test.v -test.timeout 60m
# convert to jsonl for xf_train compatibility
python3 - <<'PY'
import numpy as np,json
D='/workspace/maclaw_reranker/out/replace_audit/r4A'
keys=[l.rstrip('\n') for l in open(D+'/vec_hard.keys')]
V=np.fromfile(D+'/vec_hard.f32',dtype='<f4').reshape(len(keys),-1)
with open(D+'/emb_hard_qwen3go.jsonl','w') as f:
    for i,k in enumerate(keys):
        f.write(json.dumps({'id':k,'vec':V[i].tolist()})+'\n')
print('wrote',len(keys))
PY
