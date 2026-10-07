# Session-5 stage-1 evaluation with the rule fixed in STATUS_r4.md. Merges several levers pkl files (LEV_PKLS, comma-separated).
import os, json, numpy as np, pickle, itertools
TAG='le2'; OLDW=5
src=open('scripts/xf_train.py').read(); exec(src.split("RES = dict(")[0])
res={}; FAMSv=None
for p in os.environ['LEV_PKLS'].split(','):
    D=pickle.load(open(p,'rb')); res.update(D['res']); FAMSv=D['FAMS']
LAB=list(next(iter(res.values()))[0]); assert all(list(v[0])==LAB for v in res.values())
K150={json.loads(l)['id'] for l in open('data/go_in_GK_test_150.jsonl')}
rng=np.random.RandomState(0); idx=rng.permutation(len(old_tr)); OF=np.array_split(idx,5)
R_oof=sum([[r for r in XF if r['fam']==F] for F in FAMSv],[]); R_cv=sum([[old_tr[i] for i in f] for f in OF],[])
kmask=np.array([r['id'] in K150 for r in IND])
def Zs(mem,kind):
    if kind=='final': return np.mean([res[(r,s,'final',0)][1] for r,s in mem],0)
    n=len(FAMSv) if kind=='lofo' else 5
    return np.mean([np.concatenate([res[(r,s,kind,k)][1] for k in range(n)]) for r,s in mem],0)
def rule(Z,R):
    y=np.array([LAB.index(r['intent']) for r in R]); Tm=T.fit_temp(Z,y); P=T.softmax(Z/Tm); conf=P.max(1); pred=[LAB[j] for j in P.argmax(1)]
    corr=np.array([ok(r['intent'],p) for r,p in zip(R,pred)]); isS=np.array([p in SENS for p in pred]); sw=[]
    for ts in (0.95,0.97,0.98,0.99):
        for tau in (0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.97,0.98,0.99):
            a=conf>=np.where(isS,max(tau,ts),tau); sa=a&isS
            sw.append(dict(tau=tau,tau_s=ts,cov=float(a.mean()),sel=float(corr[a].mean()) if a.any() else 0,sensFE=int((a&isS&~corr).sum()),sens_prec=float(corr[sa].mean()) if sa.any() else 1.0))
    c=[d for d in sw if d['sel']>=0.976 and d['sens_prec']>=0.997]; return Tm,(max(c,key=lambda d:(d['cov'],-d['tau_s'],-d['tau'])) if c else None)
def gate(Z,R,Tm,tau,ts,mask=None):
    P=T.softmax(Z/Tm); conf=P.max(1); pred=[LAB[j] for j in P.argmax(1)]; corr=np.array([ok(r['intent'],p) for r,p in zip(R,pred)]); isS=np.array([p in SENS for p in pred])
    a=conf>=np.where(isS,max(tau,ts),tau); m=np.ones(len(R),bool) if mask is None else mask
    return dict(cov=float(a[m].mean()),sel=float(corr[a&m].mean()),sensFE=int((a&isS&~corr&m).sum()),primary=float((a&corr&m).mean()))
keys=sorted({(k[0],k[1]) for k in res})
groups={}
for r,s in keys: groups.setdefault(r,[]).append((r,s))
cands=[[k] for k in keys]+[g for g in groups.values() if len(g)>1]+([keys] if len(groups)>1 else [])
rows=[]
for mem in cands:
    Tm,b=rule(Zs(mem,'lofo'),R_oof)
    if b is None: rows.append(dict(members=mem,eligible=False,why='no rule point')); continue
    cv=gate(Zs(mem,'cv'),R_cv,Tm,b['tau'],b['tau_s']); Zf=Zs(mem,'final')
    ind=gate(Zf,IND,Tm,b['tau'],b['tau_s']); k=gate(Zf,IND,Tm,b['tau'],b['tau_s'],kmask)
    el=cv['sensFE']<=1 and ind['sensFE']<=3 and k['sensFE']<=1
    rows.append(dict(members=mem,T=Tm,oof=b,oldcv=cv,indep=ind,kimi150=k,eligible=el,score=b['cov']+ind['primary']))
    print(('*' if el else ' '),mem if len(mem)<4 else f'{mem[0][0]} x{len(mem)}' if len({m[0] for m in mem})==1 else f'ALL x{len(mem)}','T=%.3f'%Tm,'τ %.2f/%.2f OOFcov %.3f FE %d'%(b['tau'],b['tau_s'],b['cov'],b['sensFE']),
          '| oldCV FE %d prim %.3f'%(cv['sensFE'],cv['primary']),'| indep prim %.3f FE %d'%(ind['primary'],ind['sensFE']),'| K150 prim %.3f FE %d'%(k['primary'],k['sensFE']),'| score %.3f'%(b['cov']+ind['primary']),flush=True)
el=[r for r in rows if r.get('eligible')]
best=max(el,key=lambda r:(round(r['score'],6),-len(r['members']))) if el else None
print('STAGE1 WINNER:',best and (best['members'],round(best['score'],4),best['oof']['tau'],best['oof']['tau_s'],round(best['T'],4)))
json.dump(dict(rows=rows,winner=best),open(os.environ.get('LEV_OUT','out/replace_audit/r4A/levers_eval2.json'),'w'),indent=1,default=str)
