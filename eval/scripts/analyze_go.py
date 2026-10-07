import json, sys, numpy as np
from collections import Counter
rows = {json.loads(l)['id']: json.loads(l) for l in open('data/intent_all.jsonl')}
meta = json.load(open('data/intent_meta.json')); SENS = set(meta['sensitive']); SAFEU = set(meta['safe_unknown'])
def ok(g, p): return p == g or (g == 'unknown' and p in SAFEU)
def pct(a, q): return float(np.percentile(a, q)) if len(a) else float('nan')
for f in sys.argv[1:]:
    R = []
    for l in open(f):
        try: R.append(json.loads(l))
        except Exception: pass
    g = [rows[r['id']]['intent'] for r in R]
    conf = [r['l2_confident'] for r in R]
    l2ok = [ok(gg, r['l2_primary']) for gg, r in zip(g, R)]
    clsok = [ok(gg, r['cls_primary']) for gg, r in zip(g, R)]
    unk = [r['cls_primary'] in ('unknown', 'ambiguous') for r in R]
    # "routed" = classification produced a governed capability (non-unknown primary with conf); wrong & sensitive = false exposure
    fexp = [(r['id'], rows[r['id']]['text'], gg, r['cls_primary']) for gg, r in zip(g, R) if r['cls_primary'] in SENS and r['cls_primary'] != gg]
    ms = [r['cls_ms'] for r in R]
    print(f"== {f}  n={len(R)}")
    print(f" L2 raw top1 acc={np.mean(l2ok):.3f}  L2 confident rate={np.mean(conf):.3f}  acc|confident={np.mean([o for o,c in zip(l2ok,conf) if c]):.3f}")
    print(f" ClassifyContext primary acc={np.mean(clsok):.3f}  unknown/ambiguous rate={np.mean(unk):.3f}  degraded={np.mean([r['cls_degraded'] for r in R]):.3f}")
    print(f" acc on non-unknown outputs={np.mean([o for o,u in zip(clsok,unk) if not u]):.3f}  (n={sum(not u for u in unk)})")
    print(f" layers={Counter(r['cls_layer'] for r in R)}")
    print(f" sensitive false exposure (primary sensitive & wrong): {len(fexp)}")
    for x in fexp[:12]: print("   ", x)
    print(f" cls latency ms p50={pct(ms,50):.1f} p95={pct(ms,95):.1f} p99={pct(ms,99):.1f} max={max(ms):.1f}")
    l2ms = [r['l2_ms'] for r in R]; print(f" l2 latency ms p50={pct(l2ms,50):.1f} p95={pct(l2ms,95):.1f} p99={pct(l2ms,99):.1f}")
    if 'head_top' in R[0]:
        hm = [r['head_ms'] for r in R]; em = [r['head_embed_ms'] for r in R]
        hok = [ok(gg, r['head_top'][0]['L']) for gg, r in zip(g, R)]
        print(f" Go head top1 acc={np.mean(hok):.3f} head-only ms p50={pct(hm,50):.3f} p99={pct(hm,99):.3f}; embed ms p50={pct(em,50):.1f} p95={pct(em,95):.1f} p99={pct(em,99):.1f}")
