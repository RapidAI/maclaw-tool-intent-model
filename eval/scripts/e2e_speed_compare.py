"""Section 11.4: e2e before/after (Kimi first 150, GK head, tau_s=0.95): latency + identical predictions + accuracy."""
import json, sys, numpy as np, os
exec(open('scripts/analyze_go_indep.py').read().split("f = sys.argv[1]")[0])
L = lambda f: {json.loads(l)['id']: json.loads(l) for l in open(f)}
runs = [r for r in ['before', 'after', 'before2', 'after2'] if os.path.exists(f'out/e2e_speed_{r}.jsonl') and sum(1 for _ in open(f'out/e2e_speed_{r}.jsonl')) == 150]
D = {r: L(f'out/e2e_speed_{r}.jsonl') for r in runs}
ref = L('out/go_q3_kimi_GK.jsonl')
ids = list(D[runs[0]])
P = lambda a, q: float(np.percentile(a, q))
for r in runs:
    R = D[r]; ms = [R[i]['cls_ms'] for i in ids]
    esc = [R[i]['cls_ms'] for i in ids if not R[i]['l2_confident']]
    conf = [R[i]['cls_ms'] for i in ids if R[i]['l2_confident']]
    acc = np.mean([ok(gold[i]['intent'], R[i]['cls_primary']) for i in ids]); unk = np.mean([R[i]['cls_primary'] in ('unknown', 'ambiguous') for i in ids])
    print(f"{r}: n={len(ids)} acc={acc:.3f} degrade={unk:.3f} per-turn p50/p95/p99={P(ms,50):.0f}/{P(ms,95):.0f}/{P(ms,99):.0f} ms; escalated(n={len(esc)}) p50/p95={P(esc,50):.0f}/{P(esc,95):.0f}; L2-confident(n={len(conf)}) p50={P(conf,50) if conf else float('nan'):.0f}; l2_ms p50={P([R[i]['l2_ms'] for i in ids],50):.0f}")
ks = ['cls_primary', 'cls_layer', 'cls_secondary', 'cls_degraded']
for a in runs:
    for b in runs + ['ref10.7']:
        if b <= a and b != 'ref10.7': continue
        B = ref if b == 'ref10.7' else D[b]
        same = sum(all(D[a][i].get(k) == B[i].get(k) for k in ks) for i in ids)
        conf_same = sum(D[a][i]['cls_conf'] == B[i]['cls_conf'] for i in ids)
        maxdc = max(abs(D[a][i]['cls_conf'] - B[i]['cls_conf']) for i in ids)
        print(f"  {a} vs {b}: identical {ks} {same}/{len(ids)}; cls_conf bit-identical {conf_same}/{len(ids)} (max |diff| {maxdc:.2e})")
