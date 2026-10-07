#!/bin/bash
# One-shot B_confirm embedding (frozen set; contents not printed). Qwen3: IS task / IM message; Gemma: plain (production).
set -e
R=/workspace/maclaw_reranker; D=$R/out/replace_audit/r4B; CF=$R/out/replace_audit/confirm_r4/B_confirm.jsonl
python3 - <<PY
import json
IS="Represent the user request by the task it concerns, so that a follow-up message and the task it modifies are close"
IM="Given a message a user sent while an AI agent was executing a task, retrieve the task this message refers to"
P=[json.loads(l) for l in open("$CF")]
q=[];g=[];sq=set();sg=set()
for p in P:
    for k,t in (("IS|"+p['task'],"Instruct: %s\nQuery: %s"%(IS,p['task'])),("IM|"+p['message'],"Instruct: %s\nQuery: %s"%(IM,p['message']))):
        if k not in sq: sq.add(k); q.append({'k':k,'t':t})
    for t in (p['task'],p['message']):
        if t not in sg: sg.add(t); g.append({'k':t,'t':t})
open("$D/conf_q.jsonl","w").write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in q))
open("$D/conf_g.jsonl","w").write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in g))
print('n pairs',len(P),'qwen texts',len(q),'gemma texts',len(g))
PY
cd /workspace/maclaw-qwen3only/corelib/embedding
ZZ_VD_MODEL=$R/models/Qwen3-Embedding-0.6B-Q8_0.gguf ZZ_VD_IN=$D/conf_q.jsonl ZZ_VD_OUT=$D/vconf_q $R/bin/vecdump.test -test.run TestZZVecDump -test.timeout 60m | tail -1
ZZ_VD_MODEL=$R/models/embeddinggemma-300M-Q8_0.gguf ZZ_VD_IN=$D/conf_g.jsonl ZZ_VD_OUT=$D/vconf_g $R/bin/vecdump.test -test.run TestZZVecDump -test.timeout 60m | tail -1
