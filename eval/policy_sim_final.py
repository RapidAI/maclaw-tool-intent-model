# Why tau alone cannot reach 0.976, and which pipeline change would (SIMULATION ONLY, no code change).
# Selection uses indep839 only (out-of-family); old-339 is reported, never used for selection.
import json, itertools
rows={json.loads(l)['id']:json.loads(l) for f in ('data/intent_all.jsonl','data/indep_test.jsonl') for l in open(f)}
meta=json.load(open('data/intent_meta.json')); SAFEU=set(meta['safe_unknown']); SENS=set(meta['sensitive'])
H=json.load(open('out/replace_audit/head_mlp_qwen3go_xf.final.snapshot.json')); T0,TS0=H['tau'],H['tau_s']
ok=lambda g,p: p==g or (g=='unknown' and p in SAFEU)
def sim(R,tau,veto,override):
    ts=max(TS0,tau); n=len(R); acc=rel=relok=fe=0
    for r in R:
        g=rows[r['id']]['intent']; p=r['cls_primary']; ht=r.get('head_top') or []
        top=ht[0]['L'] if ht else None; tp=ht[0]['P'] if ht else 0
        if r['cls_layer']==3 and not r['cls_degraded'] and p not in ('unknown','ambiguous'):
            if r['cls_conf']<(ts if p in SENS else tau): p='unknown'
        elif r['cls_layer']==2 and p not in ('unknown','ambiguous') and ht:
            if r['cls_degraded'] and veto and top!=p: p='unknown'          # head contradicts a degraded L2 hint -> abstain
            elif not r['cls_degraded'] and override and top!=p and top not in (r.get('cls_secondary') or []) and tp>=(ts if top in SENS else override):
                p=top                                                     # confident head overrides a contradicting L2 answer
        acc+=ok(g,p)
        if p not in ('unknown','ambiguous'): rel+=1; relok+=ok(g,p); fe+=(p in SENS and p!=g)
    return acc/n, rel, relok/max(rel,1), fe
cal=[json.loads(l) for l in open('out/xf/go_xfF_indep839.jsonl')]
import sys
tst=[json.loads(l) for l in open(sys.argv[1] if len(sys.argv)>1 else 'out/xf/go_xfF_old339.jsonl')]
print('policy                      | indep839 prim/acc(n)/acc|acc/sensFE | old339 prim/acc(n)/acc|acc/sensFE')
res=[]
for tau,veto,ov in itertools.product([0.87,0.90,0.93,0.96],[False,True],[0,0.90,0.95,0.98]):
    a=sim(cal,tau,veto,ov); b=sim(tst,tau,veto,ov); res.append((tau,veto,ov,a,b))
    print(f"tau={tau:.2f} veto={int(veto)} ovr={ov:<4} | {a[0]:.3f}/{a[1]}/{a[2]:.3f}/{a[3]} | {b[0]:.3f}/{b[1]}/{b[2]:.3f}/{b[3]}")
good=[x for x in res if x[3][2]>=0.976 and x[3][3]<=2]
best=max(good,key=lambda x:x[3][0]) if good else None
print('selected on indep839 (acc|acc>=0.976, sensFE<=2, max primary):', best and best[:3], '-> old339', best and best[4])
