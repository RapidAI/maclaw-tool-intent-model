# Round-3 policy simulation for the hint fix (selection on indep839 ONLY; old-339 reported).
#  veto: head uncertain (L3 failed) and L2 degraded hint (search/live_data/office) != head top -> unknown
#  ovr:  L2 confident (non-degraded, accepted) and head top P >= t and head top != L2 primary -> head top label
import json, sys
rows={json.loads(l)['id']:json.loads(l) for f in ('data/intent_all.jsonl','data/indep_test.jsonl') for l in open(f)}
meta=json.load(open('data/intent_meta.json')); SAFEU=set(meta['safe_unknown']); SENS=set(meta['sensitive'])
H=json.load(open(sys.argv[3] if len(sys.argv)>3 else 'out/replace_audit/head_mlp_qwen3go_xf.final.snapshot.json')); T0,TS0=H['tau'],H['tau_s']
ok=lambda g,p: p==g or (g=='unknown' and p in SAFEU)
def sim(R,tau,veto,ovr):
    ts=max(TS0,tau); n=len(R); acc=rel=relok=fe=0; nov=0
    for r in R:
        g=rows[r['id']]['intent']; p=r['cls_primary']; ht=r.get('head_top') or []
        top=ht[0]['L'] if ht else None; tp=ht[0]['P'] if ht else 0
        if r['cls_layer']==3 and not r['cls_degraded'] and p not in ('unknown','ambiguous'):
            if r['cls_conf']<(ts if p in SENS else tau): p='unknown'
        elif r['cls_layer']==2 and p not in ('unknown','ambiguous') and ht:
            if r['cls_degraded']:
                if veto and top!=p: p='unknown'
            elif ovr and top!=p and top!='unknown' and tp>=max(ovr, ts if top in SENS else 0):
                p=top; nov+=1
        acc+=ok(g,p)
        if p not in ('unknown','ambiguous'): rel+=1; relok+=ok(g,p); fe+=(p in SENS and p!=g)
    return dict(acc=round(acc/n,3),n_acc=rel,acc_acc=round(relok/max(rel,1),3),sens=fe,overrides=nov)
cal=[json.loads(l) for l in open(sys.argv[1])]; tst=[json.loads(l) for l in open(sys.argv[2])]
res=[]
for tau in [T0,0.90,0.93,0.96]:
  for veto in (0,1):
    for ovr in (0,0.90,0.95,0.97,0.99):
        a=sim(cal,tau,veto,ovr); b=sim(tst,tau,veto,ovr); res.append(((tau,veto,ovr),a,b))
        print(f"tau={tau:.2f} veto={veto} ovr={ovr:<4} | indep {a} | old339 {b}")
good=[x for x in res if x[1]['acc_acc']>=0.976]
best=max(good,key=lambda x:(x[1]['acc'],-x[1]['sens'])) if good else None
print('SELECTED on indep839 (acc|acc>=0.976, max primary):',best and best[0],'-> old339',best and best[2])
