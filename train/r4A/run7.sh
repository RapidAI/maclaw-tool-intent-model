#!/bin/bash
# r4 s7 pipeline on the GPU server, in ~/maclaw_r4a/ws7 (copy of ws). Needs aug.jsonl/aug.npy/aug.ids.json in ws7 root and the vLLM server stopped (GPU memory).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; set -e; cd ~/maclaw_r4a/ws7; P=out/replace_audit/r4A; PY=~/r4venv/bin/python
echo "start $(date +%T)"
if [ ! -f .aug_applied ]; then
$PY - <<'PYEOF'
import json, numpy as np
A=[json.loads(l) for l in open('aug.jsonl')]; ids=json.load(open('aug.ids.json')); X=np.load('aug.npy')
assert [r['id'] for r in A]==ids
base=json.load(open('allvec.ids.json')); assert not set(ids)&set(base)
np.save('allvec.npy',np.concatenate([np.load('allvec.npy'),X]).astype(np.float32)); json.dump(base+ids,open('allvec.ids.json','w'))
with open('data/train_xfamily/train_xf.jsonl','a') as f:
    for r in A: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print('aug applied',len(A))
PYEOF
touch .aug_applied; fi
LEV_PROCS=18 $PY $P/levers_a.py > lev7.log 2>&1; echo "sklearn done $(date +%T)"
$PY $P/dump_folds.py > folds7.log 2>&1
for s in 0 1 2; do ~/vllm-env/bin/python $P/torch_ls2.py 0.03,0.05,0.1 $s hard $P/ls7_s$s.pkl > ls7_s$s.log 2>&1 & done
$PY $P/lc.py > lc7.log 2>&1; echo "lc done $(date +%T)"
wait; echo "torch done $(date +%T)"
LEV_PKLS=$P/levers_raw.pkl,$P/ls7_s0.pkl,$P/ls7_s1.pkl,$P/ls7_s2.pkl LEV_OUT=$P/levers_eval7.json $PY $P/levers_eval7.py > eval7.log 2>&1
echo "eval done $(date +%T)"; grep "STAGE1 WINNER" eval7.log
