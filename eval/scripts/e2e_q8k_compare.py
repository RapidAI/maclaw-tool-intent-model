"""Section 12.5: e2e Q8P variants vs section 11 float-path run (out/e2e_speed_after{,2}.jsonl), Kimi first 150, GK head, tau_s=0.95."""
import json, sys, numpy as np, os
exec(open('scripts/analyze_go_indep.py').read().split("f = sys.argv[1]")[0])
L = lambda f: {json.loads(l)['id']: json.loads(l) for l in open(f)}
refs = {'sec11_after': 'out/e2e_speed_after.jsonl', 'sec11_after2': 'out/e2e_speed_after2.jsonl'}
vars_ = ['q8p_warm', 'q8p', 'q8p2_warm', 'q8p2', 'q8p_oldcache', 'q8p_b']
D = {}
for k, f in refs.items(): D[k] = L(f)
for v in vars_:
    f = f'out/e2e_q8k_{v}.jsonl'
    if os.path.exists(f) and sum(1 for _ in open(f)) == 150: D[v] = L(f)
    else: print(f"{v}: missing/incomplete")
ids = list(D['sec11_after'])
P = lambda a, q: float(np.percentile(a, q))
for r, R in D.items():
    ms = [R[i]['cls_ms'] for i in ids]
    esc = [R[i]['cls_ms'] for i in ids if not R[i]['l2_confident']]
    g = [gold[i]['intent'] for i in ids]
    acc = np.mean([ok(a, R[i]['cls_primary']) for a, i in zip(g, ids)]); unk = np.mean([R[i]['cls_primary'] in ('unknown', 'ambiguous') for i in ids])
    fe = sum(1 for a, i in zip(g, ids) if R[i]['cls_primary'] in SENS and R[i]['cls_primary'] != a)
    print(f"{r}: acc={acc:.3f} degrade={unk:.3f} sens_false_exposure={fe} L2_confident={sum(R[i]['l2_confident'] for i in ids)} per-turn p50/p95/p99={P(ms,50):.0f}/{P(ms,95):.0f}/{P(ms,99):.0f} ms; escalated(n={len(esc)}) p50/p95={P(esc,50):.0f}/{P(esc,95):.0f}; head_embed p50={P([R[i].get('head_embed_ms',np.nan) for i in ids],50):.0f}; l2_ms p50={P([R[i]['l2_ms'] for i in ids],50):.0f}")
ks = ['cls_primary', 'cls_layer', 'cls_secondary', 'cls_degraded']
for v in [x for x in vars_ if x in D]:
    for b in ['sec11_after', 'q8p', 'q8p2']:
        if b == v or b not in D: continue
        B = D[b]
        diff = [i for i in ids if any(D[v][i].get(k) != B[i].get(k) for k in ks)]
        pflip = [i for i in ids if D[v][i]['cls_primary'] != B[i]['cls_primary']]
        l2flip = sum(D[v][i]['l2_confident'] != B[i]['l2_confident'] for i in ids)
        l2p = sum(D[v][i]['l2_primary'] != B[i]['l2_primary'] for i in ids)
        h1 = sum(D[v][i]['head_top'][0]['L'] != B[i]['head_top'][0]['L'] for i in ids if 'head_top' in D[v][i] and 'head_top' in B[i])
        maxdc = max(abs(D[v][i]['cls_conf'] - B[i]['cls_conf']) for i in ids)
        print(f"  {v} vs {b}: fields-identical {150-len(diff)}/150, cls_primary flips={len(pflip)}, l2_primary flips={l2p}, l2_confident flips={l2flip}, head top1 flips={h1}, max|d cls_conf|={maxdc:.2e}")
        for i in pflip:
            print(f"     {i} gold={gold[i]['intent']} {b}={B[i]['cls_primary']}(L{B[i]['cls_layer']},{B[i]['cls_conf']:.3f}) -> {v}={D[v][i]['cls_primary']}(L{D[v][i]['cls_layer']},{D[v][i]['cls_conf']:.3f})")
