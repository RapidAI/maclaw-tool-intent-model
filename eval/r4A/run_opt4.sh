#!/bin/bash
# r4 option 4: (a) matryoshka 512/768 ×3 seeds, (b) r10CAP recipe seeds 3-6. BLAS pinned. Then stage-1 evals.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; cd ~/maclaw_r4a/ws7; P=out/replace_audit/r4A
echo "start $(date +%T)"
for d in 512 768; do for s in 0 1 2; do ~/vllm-env/bin/python $P/torch_mat.py $d $s $P/mat${d}_s$s.pkl > mat${d}_s$s.log 2>&1 & done; done
for s in 3 4 5 6; do ~/vllm-env/bin/python $P/torch_cap.py 512 1 0.3 $s $P/cap_w512_d1_p0.3_s$s.pkl > cap_w512_d1_p0.3_s$s.log 2>&1 & done
wait; echo "torch done $(date +%T)"
LEV_PKLS=$(ls $P/mat*_s?.pkl | paste -sd,) LEV_OUT=$P/levers_eval_mat.json ~/r4venv/bin/python $P/levers_eval7.py > eval_mat.log 2>&1
LEV_PKLS=$(ls $P/cap_w512_d1_p0.3_s?.pkl | paste -sd,) LEV_OUT=$P/levers_eval_seeds.json ~/r4venv/bin/python $P/eval_seeds.py > eval_seeds.log 2>&1
echo "eval done $(date +%T)"
