#!/bin/bash
# usage: run_head_pipeline.sh <head.json> <tag>
# fix0 + fix1 on old339 and indep839 (production localhead path), then the out-of-family tau sweep
# (selection on indep839 only) and a Go confirmation run at the selected tau.
A=/workspace/maclaw_reranker/out/replace_audit; H=$1; TAG=$2; cd $A
for s in old339 indep839; do for f in 0 1; do
  [ -s hintfix/go_${TAG}_fix${f}_${s}.jsonl ] || ./run_hintfix.sh $H $TAG $f $s
done; done
cd /workspace/maclaw_reranker
python3 $A/policy_sim_r3.py $A/hintfix/go_${TAG}_fix0_indep839.jsonl $A/hintfix/go_${TAG}_fix0_old339.jsonl $H > $A/hintfix/sweep_${TAG}.txt
SEL=$(sed -n 's/^SELECTED.*: (\([0-9.]*\), 1, .*/\1/p' $A/hintfix/sweep_${TAG}.txt)
echo "selected tau (indep839, fix on) = ${SEL:-none}"
if [ -n "$SEL" ]; then
  python3 -c "import json;d=json.load(open('$H'));d['tau']=float('$SEL');d['tau_s']=max(d['tau_s'],d['tau']);json.dump(d,open('$A/hintfix/head_${TAG}_tau$SEL.json','w'))"
  cd $A; for s in old339 indep839; do ./run_hintfix.sh $A/hintfix/head_${TAG}_tau$SEL.json ${TAG}tau$SEL 1 $s; done
fi
