# Lever 2: L2 "rescue" rules for items the head declines and strict L2 does not grant. Tuned on TRAIN only
# (old_tr: head = old-train CV probs; xf sample: head = LOFO OOF probs), then reported on indep839/Kimi150 (allowed for choosing).
# Pipeline sim: strict L2 confident -> L2 label; elif head conf >= tau(_s) -> head top1; elif rescue -> label; else unknown.
import json, sys, numpy as np, itertools
R='/workspace/maclaw_reranker'
meta=json.load(open(f'{R}/data/intent_meta.json')); SENS=set(meta['sensitive']); SAFEU=set(meta['safe_unknown'])
ok=lambda g,p: p==g or (g=='unknown' and p in SAFEU)
H=json.load(open(sys.argv[1]))            # from dump_probs.py: train {text:[gold,top1,P]}, indep {id:[gold,top1,P]}
t2id={json.loads(l)['text']:json.loads(l)['id'] for l in open(f'{R}/out/replace_audit/r4A/l2/train_in.jsonl')}
gold={}; HP={}
for t,(g,l,p) in H['train'].items():
    if t in t2id: gold[t2id[t]]=g; HP[t2id[t]]=(l,p)
for i,(g,l,p) in H['indep'].items(): gold[i]=g; HP[i]=(l,p)
L2={}
for f in sys.argv[2].split(','):
    for l in open(f): r=json.loads(l); L2[r['id']]=r
tau,ts=H['tau'],H['tau_s']
def outcome(i,rule):
    r=L2[i]; hl,hp=HP[i]
    if r['l2_confident']: return r['l2_primary'],'L2'
    need=max(tau,ts) if hl in SENS else tau
    if hp>=need: return hl,'L3'
    if rule:
        kind,a,s,g=rule; l2p=r['l2_primary']; c=r['l2_conf']; gap=c-(r.get('l2_runnerup_score') or 0)
        if l2p in SENS: return 'unknown','none'          # never rescue sensitive labels
        if kind=='agree' and hl==l2p and hp>=a and c>=s and gap>=g: return l2p,'R'
        if kind=='l2' and c>=s and gap>=g: return l2p,'R'
    return 'unknown','none'
def stats(ids,rule):
    o=[outcome(i,rule) for i in ids]; g=[gold[i] for i in ids]
    corr=np.array([ok(gg,p) for gg,(p,_) in zip(g,o)]); acc=np.array([p not in ('unknown',) for p,_ in o])
    resc=np.array([w=='R' for _,w in o]); fe=sum(1 for gg,(p,_) in zip(g,o) if p in SENS and p!=gg)
    return dict(n=len(ids),primary=float(corr.mean()),acc_sel=float(corr[acc].mean()) if acc.any() else 0,rescued=int(resc.sum()),
                resc_prec=float(corr[resc].mean()) if resc.any() else 1.0,sensFE=fe)
tr=[i for i in HP if i in L2 and not i.startswith('indep')]; ind=[i for i in HP if i in L2 and i.startswith('indep')]
K150={json.loads(l)['id'] for l in open(f'{R}/data/go_in_GK_test_150.jsonl')}; kim=[i for i in ind if i in K150]
grid=[None]+[('agree',a,s,0.0) for a in (0.3,0.4,0.5,0.6,0.7) for s in (0.55,0.60,0.65,0.70,0.75)]+[('l2',0,s,g) for s in (0.65,0.70,0.75,0.78) for g in (0.03,0.05,0.07,0.10)]
rows=[]
for rule in grid:
    st=stats(tr,rule); rows.append((rule,st))
base=rows[0][1]
# TRAIN rule: rescued precision >= 0.976 AND train acc_sel >= base acc_sel - 0.002 → max train primary (ties: stricter rule)
elig=[(r,s) for r,s in rows[1:] if s['resc_prec']>=0.976 and s['acc_sel']>=base['acc_sel']-0.002]
best=max(elig,key=lambda x:(round(x[1]['primary'],6),x[0][1],x[0][2])) if elig else (None,base)
print('TRAIN base',base); 
for r,s in sorted(rows[1:],key=lambda x:-x[1]['primary'])[:8]: print('  ',r,s)
print('TRAIN-SELECTED',best)
if ind:
    for name,ids in (('indep839',ind),('Kimi150',kim)):
        print(name,'base',stats(ids,None),'| rule',stats(ids,best[0]))
json.dump(dict(train_base=base,selected=best,grid=[(r,s) for r,s in rows]),open('sim_rescue.json','w'),indent=1,default=str)
