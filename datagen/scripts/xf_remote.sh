#!/bin/bash
# Sync inputs to the ARM box (<GPU_HOSTNAME>) and run xf_train.py there (CPU sklearn; vectors are the pure-Go Qwen3 ones from this box).
# usage: xf_remote.sh <XF_TAG> <XF_FRACS> [export]   ; results copied back to out/xf/ and heads/
set -e; cd /workspace/maclaw_reranker
H=<GPU_USER>@<GPU_HOST>; RD=maclaw_xfamily
S() { SSHPASS="$GPU_SSH_PASSWORD" sshpass -e "$@"; }
tar czf - scripts/train_heads.py scripts/tool_gold.py scripts/xf_train.py scripts/xf_final.py data/intent_all.jsonl data/intent_meta.json \
  data/indep_test.jsonl data/indep_ambiguous.jsonl data/fix_anchors.jsonl data/train_xfamily/train_xf.jsonl data/train_xfamily/train_xf_r1b.jsonl data/train_xfamily/train_xf_r2a.jsonl data/train_xfamily/train_xf_r2b.jsonl data/train_xfamily/train_xf_r3.jsonl scripts/xf_oldcv.py data/core_tool_defs.json data/tool_eval.jsonl \
  out/emb_qwen3go.npy out/emb_qwen3go.npy.ids.json out/emb_indep_qwen3go.npy out/emb_indep_qwen3go.npy.ids.json out/emb_fix_qwen3go.jsonl \
  out/xf/emb_xf_qwen3go.jsonl heads/head_mlp_qwen3go_GK.json heads/head_mlp_qwen3go_KG.json | S ssh -p <GPU_SSH_PORT> $H "mkdir -p $RD && tar xzf - -C $RD"
SCRIPT=scripts/xf_train.py; [ "$3" = final ] && SCRIPT=scripts/xf_final.py; [ "$3" = oldcv ] && SCRIPT=scripts/xf_oldcv.py
S ssh -p <GPU_SSH_PORT> $H "cd $RD && mkdir -p out/xf logs && OMP_NUM_THREADS=2 XF_OLDW=${XF_OLDW:-1} XF_TAG=$1 XF_FRACS=$2 venv/bin/python -u $SCRIPT $3 > logs/train_$1.log 2>&1; echo rc=\$?"
S scp -P <GPU_SSH_PORT> $H:$RD/logs/train_$1.log logs/xf/train_$1.remote.log
S scp -P <GPU_SSH_PORT> $H:$RD/out/xf/xf_train_$1.json out/xf/ 2>/dev/null || true; S scp -P <GPU_SSH_PORT> $H:$RD/out/xf/xf_final_$1.json out/xf/ 2>/dev/null || true; S scp -P <GPU_SSH_PORT> $H:$RD/out/xf/xf_oldcv_$1.json out/xf/ 2>/dev/null || true
if [ "$3" = export ]; then S scp -P <GPU_SSH_PORT> $H:$RD/heads/head_mlp_qwen3go_xf.json heads/; S scp -P <GPU_SSH_PORT> $H:$RD/heads/tool_head_ovr_qwen3go_xf.json heads/; fi
# final mode never overwrites the published FINAL head; lands as a tagged candidate
if [ "$3" = final ]; then S scp -P <GPU_SSH_PORT> $H:$RD/heads/head_mlp_qwen3go_xf.json heads/head_mlp_qwen3go_xf_$1.json; S scp -P <GPU_SSH_PORT> $H:$RD/heads/tool_head_ovr_qwen3go_xf.json heads/tool_head_ovr_qwen3go_xf_$1.json; python3 scripts/xf_toolhead_compat.py heads/tool_head_ovr_qwen3go_xf_$1.json heads/tool_head_ovr_qwen3go_xf_$1.json; fi
