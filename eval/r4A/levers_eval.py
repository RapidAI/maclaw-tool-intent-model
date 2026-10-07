# Evaluate single heads / logit-average ensembles from levers_raw.pkl with the LOCKED r4 rule. Selection: OOF (rule) + old-CV sensFE/sel; indep839 = report gate.
import os, json, numpy as np, pickle, itertools
TAG='leveval'; OLDW=5
src=open('scripts/xf_train.py').read(); exec(src.split("RES = dict(")[0])
D=pickle.load(open(os.environ.get('LEV_PKL','out/replace_audit/r4A/levers_raw.pkl'),'rb')); res=D['res']; FAMSv=D['FAMS']
labs={v[0] and tuple(v[0]) for v in res.values()}; assert len(labs)==1; LAB=list(labs.pop())
rng=np.random.RandomState(0); idx=rng.permutation(len(old_tr)); OF=np.array_split(idx,5)
R_oof=sum([[r for r in XF if r['fam']==F] for F in FAMSv],[]); R_cv=sum([[old_tr[i] for i in f] for f in OF],[])
recs=sorted({(k[0],k[1]) for k in res})
def Zs(members,kind):
    n=len(FAMSv) if kind=='lofo' else 5
    if kind=='final': return np.mean([res[(r,s,'final',0)][1] for r,s in members],0)
    return np.mean([np.concatenate([res[(r,s,kind,k)][1] for k in range(n)]) for r,s in members],0)
def rule(Z,R):
    y=np.array([LAB.index(r['intent']) for r in R]); Tm=T.fit_temp(Z,y); P=T.softmax(Z/Tm); conf=P.max(1); pred=[LAB[j] for j in P.argmax(1)]
    corr=np.array([ok(r['intent'],p) for r,p in zip(R,pred)]); isS=np.array([p in SENS for p in pred]); sw=[]
    for ts in (0.95,0.97,0.98,0.99):
        for tau in (0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.97,0.98,0.99):
            a=conf>=np.where(isS,max(tau,ts),tau); sa=a&isS
            sw.append(dict(tau=tau,tau_s=ts,cov=float(a.mean()),sel=float(corr[a].mean()) if a.any() else 0,sensFE=int((a&isS&~corr).sum()),sens_prec=float(corr[sa].mean()) if sa.any() else 1.0))
    c=[d for d in sw if d['sel']>=0.976 and d['sens_prec']>=0.997]; b=max(c,key=lambda d:(d['cov'],-d['tau_s'],-d['tau'])) if c else None
    return Tm,b,float(corr.mean())
def gate(Z,R,Tm,tau,ts):
    P=T.softmax(Z/Tm); conf=P.max(1); pred=[LAB[j] for j in P.argmax(1)]; corr=np.array([ok(r['intent'],p) for r,p in zip(R,pred)]); isS=np.array([p in SENS for p in pred])
    a=conf>=np.where(isS,max(tau,ts),tau); return dict(top1=float(corr.mean()),cov=float(a.mean()),sel=float(corr[a].mean()),sensFE=int((a&isS&~corr).sum()),primary=float((a&corr).mean()))
rows=[]
cands=[[m] for m in recs]+[list(c) for c in itertools.combinations(recs,2)]+[[m for m in recs if m[0]==r] for r in ('base','hard')]+[recs]
seen=set()
for mem in cands:
    key=tuple(sorted(mem))
    if key in seen or not mem: continue
    seen.add(key)
    Zo=Zs(mem,'lofo'); Tm,b,top1=rule(Zo,R_oof)
    if b is None: rows.append(dict(members=key,oof=None)); continue
    cv=gate(Zs(mem,'cv'),R_cv,Tm,b['tau'],b['tau_s']); ind=gate(Zs(mem,'final'),IND,Tm,b['tau'],b['tau_s'])
    rows.append(dict(members=key,T=Tm,oof_top1=top1,oof=b,oldcv=cv,indep=ind))
    print(key,'T=%.3f'%Tm,'OOF',{k:round(v,3) if isinstance(v,float) else v for k,v in b.items()},'| oldCV',{k:round(v,3) if isinstance(v,float) else v for k,v in cv.items()},'| indep',{k:round(v,3) if isinstance(v,float) else v for k,v in ind.items()},flush=True)
json.dump(rows,open(os.environ.get('LEV_OUT','out/replace_audit/r4A/levers_eval.json'),'w'),indent=1,default=str)
