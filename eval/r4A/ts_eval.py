# r4 LEVER 2a (rule in STATUS s9): sensitive-label temperature on the 2b winner (hard_fe0.5 x3). T_s = NLL fit on pooled OOF items whose argmax is SENS,
# T = NLL fit on the rest; locked tau rule on two-temperature probs; same stage-1 eligibility/score as levers_eval7.
import os, json, numpy as np, pickle
os.environ.setdefault('LEV_OUT','/tmp/le_ts_dummy.json')
exec(open('out/replace_audit/r4A/levers_eval7.py').read().split("keys=sorted(")[0])
mem=[('hard_fe0.5',0),('hard_fe0.5',1),('hard_fe0.5',2)]
Zo=Zs(mem,'lofo'); y=np.array([LAB.index(r['intent']) for r in R_oof]); Sidx=np.array([l in SENS for l in LAB])
am=Zo.argmax(1); sm=Sidx[am]
Tn=T.fit_temp(Zo[~sm],y[~sm]); Tsn=T.fit_temp(Zo[sm],y[sm]); T1=T.fit_temp(Zo,y)
def probs(Z,Tn,Ts):
    s=Sidx[Z.argmax(1)]; return np.where(s[:,None],T.softmax(Z/Ts),T.softmax(Z/Tn))
def sweep(P,R):
    conf=P.max(1); pred=[LAB[j] for j in P.argmax(1)]; corr=np.array([ok(r['intent'],p) for r,p in zip(R,pred)]); isS=np.array([p in SENS for p in pred]); sw=[]
    for ts in (0.95,0.97,0.98,0.99):
        for tau in (0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.97,0.98,0.99):
            a=conf>=np.where(isS,max(tau,ts),tau); sa=a&isS
            sw.append(dict(tau=tau,tau_s=ts,cov=float(a.mean()),sel=float(corr[a].mean()) if a.any() else 0,sensFE=int((a&isS&~corr).sum()),sens_prec=float(corr[sa].mean()) if sa.any() else 1.0))
    c=[d for d in sw if d['sel']>=0.976 and d['sens_prec']>=0.997]; return max(c,key=lambda d:(d['cov'],-d['tau_s'],-d['tau'])) if c else None
def gate2(P,R,tau,ts,mask=None):
    conf=P.max(1); pred=[LAB[j] for j in P.argmax(1)]; corr=np.array([ok(r['intent'],p) for r,p in zip(R,pred)]); isS=np.array([p in SENS for p in pred])
    a=conf>=np.where(isS,max(tau,ts),tau); m=np.ones(len(R),bool) if mask is None else mask
    return dict(cov=float(a[m].mean()),sel=float(corr[a&m].mean()),sensFE=int((a&isS&~corr&m).sum()),primary=float((a&corr&m).sum()/m.sum()))
out={}
for name,(tn,tsv) in {'single_T(2b)':(T1,T1),'two_T(2a)':(Tn,Tsn)}.items():
    b=sweep(probs(Zo,tn,tsv),R_oof)
    if b is None: print(name,'no rule point'); out[name]=None; continue
    cv=gate2(probs(Zs(mem,'cv'),tn,tsv),R_cv,b['tau'],b['tau_s']); Pf=probs(Zs(mem,'final'),tn,tsv)
    ind=gate2(Pf,IND,b['tau'],b['tau_s']); k=gate2(Pf,IND,b['tau'],b['tau_s'],kmask)
    el=cv['sensFE']<=1 and ind['sensFE']<=3 and k['sensFE']<=1; sc=b['cov']+ind['primary']
    out[name]=dict(T=tn,T_s=tsv,oof=b,oldcv=cv,indep=ind,kimi150=k,eligible=el,score=sc)
    print(name,'T %.3f T_s %.3f'%(tn,tsv),'τ %.2f/%.2f OOFcov %.3f FE %d sens_prec %.4f'%(b['tau'],b['tau_s'],b['cov'],b['sensFE'],b['sens_prec']),'| oldCV FE %d | indep prim %.3f FE %d | K150 FE %d | eligible %s score %.4f'%(cv['sensFE'],ind['primary'],ind['sensFE'],k['sensFE'],el,sc),flush=True)
w=out['two_T(2a)']; base=out['single_T(2b)']
print('2a WINS' if (w and w['eligible'] and w['score']>base['score']) else '2a DOES NOT WIN')
json.dump(out,open('out/replace_audit/r4A/ts_eval.json','w'),indent=1)
