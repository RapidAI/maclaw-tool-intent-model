#!/bin/bash
# r4 capacity sweep on ws7: 8 configs x 3 seeds = 24 GPU processes (11 fits each), BLAS pinned.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; cd ~/maclaw_r4a/ws7; P=out/replace_audit/r4A
echo "start $(date +%T)"
for w in 256 512; do for d in 1 2; do for p in 0.1 0.3; do for s in 0 1 2; do
  ~/vllm-env/bin/python $P/torch_cap.py $w $d $p $s $P/cap_w${w}_d${d}_p${p}_s$s.pkl > cap_w${w}_d${d}_p${p}_s$s.log 2>&1 &
done; done; done; done; wait
echo "torch done $(date +%T)"
L=$(ls $P/cap_*.pkl | paste -sd,)
LEV_PKLS=$L LEV_OUT=$P/levers_eval_cap.json ~/r4venv/bin/python $P/levers_eval7.py > eval_cap.log 2>&1
echo "eval done $(date +%T)"
