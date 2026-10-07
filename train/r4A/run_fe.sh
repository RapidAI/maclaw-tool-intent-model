#!/bin/bash
# r4 s9 lever 2b on ws7 (after run7's dump_folds). GPU torch; BLAS pinned.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; cd ~/maclaw_r4a/ws7; P=out/replace_audit/r4A
while [ ! -s folds7.log ]; do sleep 20; done
echo "start $(date +%T)"
for s in 0 1 2; do ~/vllm-env/bin/python $P/torch_fe.py 0.5,1,2 $s hard $P/fe_s$s.pkl > fe_s$s.log 2>&1 & done; wait
echo "torch done $(date +%T)"
LEV_PKLS=$P/fe_s0.pkl,$P/fe_s1.pkl,$P/fe_s2.pkl LEV_OUT=$P/levers_eval_fe.json ~/r4venv/bin/python $P/levers_eval7.py > eval_fe.log 2>&1
echo "eval done $(date +%T)"
