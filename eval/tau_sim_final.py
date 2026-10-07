# tau selection for the FINAL xf head (r2bF) using out-of-family data only.
# Calibration candidates: (a) the head's own tau=0.87 (xf worker: pooled LOFO OOF over 4071 xf items, accepted acc >= 0.976)
#                         (b) indep839 (Gemini+Kimi, unseen families for every xf head, not in training/selection)
# old-339 is ONLY reported, never used for selection. Exact for tau >= head tau (raising tau turns accepted
# layer-3 answers with conf < thr into unknown); thr = max(tau, tau_s) for sensitive labels.
import json, sys, numpy as np
rows={json.loads(l)['id']:json.loads(l) for f in ('data/intent_all.jsonl','data/indep_test.jsonl') for l in open(f)}
meta=json.load(open('data/intent_meta.json')); SAFEU=set(meta['safe_unknown']); SENS=set(meta['sensitive'])
H=json.load(open('out/replace_audit/head_mlp_qwen3go_xf.final.snapshot.json')); T0,TS0=H['tau'],H['tau_s']
def ok(g,p): return p==g or (g=='unknown' and p in SAFEU)
def load(f): return [json.loads(l) for l in open(f)]
def sim(R,tau):
    ts=max(TS0,tau); n=len(R); acc=rel=relok=fe=0
    for r in R:
        g=rows[r['id']]['intent']; p=r['cls_primary']
        if r['cls_layer']==3 and not r['cls_degraded'] and p not in ('unknown','ambiguous'):
            thr= ts if p in SENS else tau
            if r['cls_conf']<thr: p='unknown'
        acc+=ok(g,p)
        if p not in ('unknown','ambiguous'): rel+=1; relok+=ok(g,p); fe+= (p in SENS and p!=g)
    return dict(acc=acc/n, accepted=rel, acc_accepted=relok/max(rel,1), sens_fe=fe, n=n)
cal=load(sys.argv[1] if len(sys.argv)>1 else 'out/xf/go_xfF_indep839.jsonl')
tst=load(sys.argv[2] if len(sys.argv)>2 else 'out/replace_audit/go_xfF_old339.jsonl')
print(f'head tau={T0} tau_s={TS0}; calibration file n={len(cal)}; report file n={len(tst)}')
print(' tau  | indep839 primary / accepted(n) / acc|accepted / sensFE | old339 primary / accepted(n) / acc|accepted / sensFE')
taus=sorted(set([T0]+list(np.round(np.arange(0.88,1.0,0.01),3))+[0.995,0.999]))
out=[]; pick=None
for tau in taus:
    a=sim(cal,tau); b=sim(tst,tau); out.append((tau,a,b))
    # target: accepted acc >= 0.976 and sensitive exposure <= 1 per 339 (scaled: <= 839/339 ≈ 2.47 → <= 2 on indep839)
    if pick is None and a['acc_accepted']>=0.976 and a['sens_fe']<=2: pick=tau
    print(f" {tau:.3f} | {a['acc']:.3f} / {a['accepted']:3d} / {a['acc_accepted']:.3f} / {a['sens_fe']} | {b['acc']:.3f} / {b['accepted']:3d} / {b['acc_accepted']:.3f} / {b['sens_fe']}")
print('(a) head tau (xf LOFO OOF):', T0, '-> old339', sim(tst,T0))
print('(b) picked on indep839 (acc|accepted>=0.976, sensFE<=2):', pick, '-> old339', sim(tst,pick) if pick else None)
json.dump({'head_tau':T0,'head_tau_s':TS0,'pick_indep839':pick,'sweep':[(t,a,b) for t,a,b in out]},open('out/replace_audit/tau_sim_final.json','w'),indent=1)
