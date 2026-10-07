"""Section 10.7: paired comparison (same ids) Qwen3-Go e2e vs Gemma-prefix e2e (tau_s=0.95) + bootstrap CI."""
import json, sys, numpy as np
src = open('scripts/analyze_go_indep.py').read().split("f = sys.argv[1]")[0]
exec(src)
def load(f):
    R = {}
    for l in open(f):
        try: r = json.loads(l); R[r['id']] = r
        except Exception: pass
    return R
pairs = [('indep839', 'out/go_q3_indep839_ts095.jsonl', 'out/go_cls_indep839_cls_ts095.jsonl'),
         ('old339', 'out/go_q3_old339_GK.jsonl', 'out/go_cls_old339_GK_ts095.jsonl'),
         ('amb95', 'out/go_q3_amb95_GK.jsonl', 'out/go_cls_amb95_GK_ts095.jsonl')]
rng = np.random.default_rng(0)
for name, fq, fg in pairs:
    Q, G = load(fq), load(fg); ids = sorted(set(Q) & set(G) & set(gold))
    cq = np.array([ok(gold[i]['intent'], Q[i]['cls_primary']) for i in ids]); cg = np.array([ok(gold[i]['intent'], G[i]['cls_primary']) for i in ids])
    d = cq.astype(float) - cg
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    print(f"{name}: n={len(ids)} qwen3={cq.mean():.3f} gemma={cg.mean():.3f} diff={d.mean():+.3f} 95%CI=[{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] q3_only_right={int(((cq)&(~cg)).sum())} gemma_only_right={int(((~cq)&(cg)).sum())}")
