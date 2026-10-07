#!/bin/bash
# r4 lever 3b on ws7: 3 seeds (11 fits each) on allvec.npy ++ allvec2.npy (I2), then stage-1 eval. BLAS pinned.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; cd ~/maclaw_r4a/ws7; P=out/replace_audit/r4A
echo "start $(date +%T)"
for s in 0 1 2; do ~/vllm-env/bin/python $P/torch_2emb.py $s $P/emb2_s$s.pkl > emb2_s$s.log 2>&1 & done; wait
echo "torch done $(date +%T)"
LEV_PKLS=$P/emb2_s0.pkl,$P/emb2_s1.pkl,$P/emb2_s2.pkl LEV_OUT=$P/levers_eval_2emb.json ~/r4venv/bin/python $P/levers_eval7.py > eval_2emb.log 2>&1
echo "eval done $(date +%T)"
