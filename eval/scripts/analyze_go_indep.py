"""Section 8.4: analyze full offline ClassifyContext (L2 + head as local L3) on the independent set."""
import json, sys, numpy as np, collections
meta = json.load(open('data/intent_meta.json')); SENS = set(meta['sensitive']); SAFEU = set(meta['safe_unknown'])
gold = {}
for f in ('data/intent_all.jsonl', 'data/indep_test.jsonl'):
    for l in open(f): r = json.loads(l); gold[r['id']] = r
amb = {json.loads(l)['id'] for l in open('data/indep_ambiguous.jsonl')}
for l in open('data/indep_ambiguous.jsonl'): r = json.loads(l); gold[r['id']] = r
ok = lambda g, p: p == g or (g == 'unknown' and p in SAFEU)
pct = lambda a, q: float(np.percentile(a, q)) if len(a) else float('nan')
def rep(name, R):
    if not R: return
    g = [gold[r['id']]['intent'] for r in R]
    c = [ok(a, r['cls_primary']) for a, r in zip(g, R)]; unk = [r['cls_primary'] in ('unknown', 'ambiguous') for r in R]
    l2 = [ok(a, r['l2_primary']) for a, r in zip(g, R)]; l2c = [r['l2_confident'] for r in R]
    fe = [(gold[r['id']]['intent'], r['cls_primary']) for r in R if r['cls_primary'] in SENS and r['cls_primary'] != gold[r['id']]['intent']]
    ms = [r['cls_ms'] for r in R]; em = [r.get('head_embed_ms', np.nan) for r in R]; hm = [r.get('head_ms', np.nan) for r in R]
    hok = [ok(a, r['head_top'][0]['L']) for a, r in zip(g, R)] if 'head_top' in R[0] else []
    print(f"== {name} n={len(R)}")
    print(f" primary acc={np.mean(c):.3f} unknown/degrade={np.mean(unk):.3f} acc|non-unknown={np.mean([x for x, u in zip(c, unk) if not u]):.3f} (n={sum(not u for u in unk)}) sens false exposure={len(fe)} ({len(fe)/len(R)*100:.1f}%)")
    print(f" L2 raw top1={np.mean(l2):.3f} L2 confident rate={np.mean(l2c):.3f} acc|L2 confident={np.mean([x for x, k in zip(l2, l2c) if k]) if any(l2c) else float('nan'):.3f}; Go head top1={np.mean(hok) if hok else float('nan'):.3f}")
    print(f" layers={dict(collections.Counter(r['cls_layer'] for r in R))}")
    for L in ('zh', 'mix', 'en'):
        k = [i for i, r in enumerate(R) if gold[r['id']].get('lang') == L]
        if k: print(f"  {L}: n={len(k)} acc={np.mean([c[i] for i in k]):.3f} unk={np.mean([unk[i] for i in k]):.3f}")
    for G in ('gemini', 'kimi'):
        k = [i for i, r in enumerate(R) if gold[r['id']].get('gen') == G]
        if k: print(f"  {G}: n={len(k)} acc={np.mean([c[i] for i in k]):.3f} unk={np.mean([unk[i] for i in k]):.3f}")
    print(f" sens FE pairs: {collections.Counter(fe).most_common(10)}")
    print(f" cls latency ms p50={pct(ms,50):.1f} p95={pct(ms,95):.1f} p99={pct(ms,99):.1f} max={max(ms):.1f}; embed p50={pct(em,50):.1f} p95={pct(em,95):.1f}; head p50={pct(hm,50):.3f} ms")
f = sys.argv[1] if len(sys.argv) > 1 else 'out/go_eval_indep_head.jsonl'
R = []
for l in open(f):
    try: R.append(json.loads(l))
    except Exception: pass
rep('indep_test (839 target)', [r for r in R if r['id'] in gold and r['id'].startswith('indep') and r['id'] not in amb])
rep('indep_ambiguous', [r for r in R if r['id'] in amb])
