# Lever PLT (per-label thresholds; rule fixed in STATUS_r4.md 18:07). env LEV_PKLS, PLT_CANDS="name=rec:s,rec:s;name2=...", LEV_OUT.
import os, json, numpy as np
src=open('out/replace_audit/r4A/levers_eval7.py').read().split("keys=sorted(")[0]; exec(src)
G=(0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.97,0.98,0.99)
def prep(Z,R,Tm):
    P=T.softmax(Z/Tm); conf=P.max(1); pred=[LAB[j] for j in P.argmax(1)]
    corr=np.array([ok(r['intent'],p) for r,p in zip(R,pred)]); return conf,np.array(pred),corr
def taumap(conf,pred,corr,tau,ts):
    M={}
    for c in LAB:
        m=pred==c; n=int(m.sum()); sens=c in SENS; base=max(tau,ts) if sens else tau
        if n<50: M[c]=base; continue
        tgt=0.997 if sens else 0.976; pick=0.99
        for t in G:
            a=m&(conf>=t)
            if a.sum()>=20 and corr[a].mean()>=tgt: pick=t; break
        M[c]=max(ts,pick) if sens else pick
    return M
def gateM(conf,pred,corr,M,mask=None):
    th=np.array([M[p] for p in pred]); a=conf>=th; isS=np.array([p in SENS for p in pred]); m=np.ones(len(pred),bool) if mask is None else mask
    return dict(cov=float(a[m].mean()),sel=float(corr[a&m].mean()) if (a&m).any() else 0,sensFE=int((a&isS&~corr&m).sum()),
                sens_prec=float(corr[a&isS&m].mean()) if (a&isS&m).any() else 1.0,primary=float((a&corr&m).sum()/m.sum()))
rows=[]
for spec in os.environ['PLT_CANDS'].split(';'):
    name,mem=spec.split('='); mem=[(m.split(':')[0],int(m.split(':')[1])) for m in mem.split(',')]
    Tm,b=rule(Zs(mem,'lofo'),R_oof)
    if b is None: rows.append(dict(name=name,eligible=False,why='base has no rule point')); print(name,'no base rule point'); continue
    co,pr,cr=prep(Zs(mem,'lofo'),R_oof,Tm); M=taumap(co,pr,cr,b['tau'],b['tau_s']); oof=gateM(co,pr,cr,M)
    valid=oof['sel']>=0.976 and oof['sens_prec']>=0.997
    cv=gateM(*prep(Zs(mem,'cv'),R_cv,Tm),M); Zf=Zs(mem,'final'); fi=prep(Zf,IND,Tm); ind=gateM(*fi,M); k=gateM(*fi,M,kmask)
    el=valid and cv['sensFE']<=1 and ind['sensFE']<=3 and k['sensFE']<=1
    rows.append(dict(name=name,members=mem,T=Tm,base=b,tau_by_label=M,oof=oof,valid=valid,oldcv=cv,indep=ind,kimi150=k,eligible=el,score=oof['cov']+ind['primary']))
    nch=sum(1 for c in LAB if M[c]!=(max(b['tau'],b['tau_s']) if c in SENS else b['tau']))
    print(('*' if el else ' '),name,'T=%.3f base τ %.2f/%.2f | labels changed %d | OOF cov %.3f sel %.4f sens_prec %.4f FE %d valid %s | oldCV FE %d prim %.3f | indep prim %.3f FE %d | K150 prim %.3f FE %d | score %.3f'%(
        Tm,b['tau'],b['tau_s'],nch,oof['cov'],oof['sel'],oof['sens_prec'],oof['sensFE'],valid,cv['sensFE'],cv['primary'],ind['primary'],ind['sensFE'],k['primary'],k['sensFE'],oof['cov']+ind['primary']),flush=True)
json.dump(dict(rows=rows),open(os.environ['LEV_OUT'],'w'),indent=1,default=str)
