# Paired comparison Qwen3 run vs Gemma run on common ids: primary acc, accepted, acc|accepted, sensitive FE; bootstrap CI of Δprimary.
import json, sys, random
R='/workspace/maclaw_reranker'
rows={json.loads(l)['id']:json.loads(l) for f in (f'{R}/data/intent_all.jsonl',f'{R}/data/indep_test.jsonl') for l in open(f)}
meta=json.load(open(f'{R}/data/intent_meta.json')); SAFEU=set(meta['safe_unknown']); SENS=set(meta['sensitive'])
ok=lambda g,p: p==g or (g=='unknown' and p in SAFEU)
def load(fs): 
    d={}
    for f in fs.split(','):
        for l in open(f): r=json.loads(l); d[r['id']]=r['cls_primary']
    return d
q=load(sys.argv[1]); g=load(sys.argv[2]); ids=sorted(set(q)&set(g))
if len(sys.argv)>3: keep={json.loads(l)['id'] for l in open(sys.argv[3])}; ids=[i for i in ids if i in keep]
def st(d,I):
    a=sum(ok(rows[i]['intent'],d[i]) for i in I); acc=[i for i in I if d[i] not in ('unknown','ambiguous')]
    return a/len(I), len(acc), sum(ok(rows[i]['intent'],d[i]) for i in acc)/max(1,len(acc)), sum(d[i] in SENS and d[i]!=rows[i]['intent'] for i in acc)
sq,sg=st(q,ids),st(g,ids)
r=random.Random(7); v=[]
x=[ok(rows[i]['intent'],q[i])-ok(rows[i]['intent'],g[i]) for i in ids]
for _ in range(2000): s=[x[r.randrange(len(x))] for _ in x]; v.append(sum(s)/len(s))
v.sort()
print(f"n={len(ids)} qwen3 {sq[0]:.3f}/{sq[2]:.3f}({sq[1]})/{sq[3]}  gemma {sg[0]:.3f}/{sg[2]:.3f}({sg[1]})/{sg[3]}  Δprimary {sq[0]-sg[0]:+.3f} [{v[50]:+.3f},{v[1949]:+.3f}]")
