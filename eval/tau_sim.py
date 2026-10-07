import json, numpy as np
rows={json.loads(l)['id']:json.loads(l) for f in ('data/intent_all.jsonl','data/indep_test.jsonl') for l in open(f)}
meta=json.load(open('data/intent_meta.json')); SAFEU=set(meta['safe_unknown']); SENS=set(meta['sensitive'])
def ok(g,p): return p==g or (g=='unknown' and p in SAFEU)
def load(f): return [json.loads(l) for l in open(f)]
def sim(R,tau):
    n=len(R); acc=0; rel=0; relok=0; fe=0
    for r in R:
        g=rows[r['id']]['intent']; p=r['cls_primary']
        if r['cls_layer']==3 and not r['cls_degraded'] and p not in ('unknown','ambiguous') and r['cls_conf']<tau: p='unknown'
        acc+=ok(g,p)
        if p not in ('unknown','ambiguous'): rel+=1; relok+=ok(g,p); fe+= (p in SENS and p!=g)
    return acc/n, rel, relok/max(rel,1), fe
cal=load('out/xf/go_xf_indep839.jsonl'); tst=load('out/replace_audit/go_xf_old339.jsonl')
print('tau | indep839 acc/rel/sel/fe | old339 acc/rel/sel/fe')
pick=None
for tau in np.round(np.arange(0.56,1.0,0.02),2):
    a=sim(cal,tau); b=sim(tst,tau)
    if pick is None and a[2]>=0.95: pick=tau
    print(tau, '%.3f %d %.3f %d'%a, '| %.3f %d %.3f %d'%b)
print('picked on indep839 (sel>=0.95):',pick, 'old339 ->', sim(tst,pick))
