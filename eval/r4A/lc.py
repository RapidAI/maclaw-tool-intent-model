# r4 s7 learning curve (REPORT ONLY): base recipe (r3W5F) seed 0 with {0,25,50,100}% of the added (aug) data. LOFO over the ORIGINAL 4 families
# (held-out family's own aug items also excluded from training), OOF metrics on original items only; final head -> indep839 head-only.
import os, sys, json, numpy as np, multiprocessing as mp
os.environ.setdefault('OMP_NUM_THREADS','1'); TAG='lc'; OLDW=5
exec(open('scripts/xf_train.py').read().split("RES = dict(")[0])
ORIGF=['glm','gptoss','llama','mistral']; ORIG=[r for r in XF if not r.get('aug')]; AUG=[r for r in XF if r.get('aug')]
perm=np.random.RandomState(0).permutation(len(AUG)); FR=[0,0.25,0.5,1.0]
def sub(fr): return [AUG[i] for i in perm[:int(round(fr*len(AUG)))]]
def job(spec):
    fr,k=spec; A=sub(fr)
    if k<4:
        F=ORIGF[k]; te=[r for r in ORIG if r['fam']==F]; tr=old_tr*OLDW+FIX+[r for r in ORIG if r['fam']!=F]+[r for r in A if r['fam']!=F]
    else: te=IND; tr=old_tr*OLDW+FIX+ORIG+A
    h=Head(tr,seed=0); return spec,h.labels,h.logits(te)
if __name__=='__main__':
    specs=[(fr,k) for fr in FR for k in range(5)]
    with mp.get_context('fork').Pool(20) as p: res={s:(l,z) for s,l,z in p.imap_unordered(job,specs)}
    out=[]
    for fr in FR:
        LAB=res[(fr,0)][0]; R=sum([[r for r in ORIG if r['fam']==F] for F in ORIGF],[]); Z=np.concatenate([res[(fr,k)][1] for k in range(4)])
        y=np.array([LAB.index(r['intent']) for r in R]); Tm=T.fit_temp(Z,y); P=T.softmax(Z/Tm); conf=P.max(1); pred=[LAB[j] for j in P.argmax(1)]
        corr=np.array([ok(r['intent'],p) for r,p in zip(R,pred)]); isS=np.array([p in SENS for p in pred]); best=None
        for ts in (0.95,0.97,0.98,0.99):
            for tau in (0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.97,0.98,0.99):
                a=conf>=np.where(isS,max(tau,ts),tau); sa=a&isS; sel=corr[a].mean() if a.any() else 0; sp=corr[sa].mean() if sa.any() else 1
                if sel>=0.976 and sp>=0.997 and (best is None or a.mean()>best['cov']): best=dict(tau=tau,tau_s=ts,cov=float(a.mean()),sel=float(sel),FE=int((sa&~corr).sum()))
        Pi=T.softmax(res[(fr,4)][1]/Tm); ci=Pi.max(1); pi=[LAB[j] for j in Pi.argmax(1)]; cri=np.array([ok(r['intent'],p) for r,p in zip(IND,pi)]); iS=np.array([p in SENS for p in pi])
        a=ci>=np.where(iS,max(best['tau'],best['tau_s']),best['tau']) if best else np.zeros(len(IND),bool)
        # DIAGNOSTIC (not selection): fixed (tau,tau_s) = r3W5F production head 0.85/0.95
        fx=conf>=np.where(isS,0.95,0.85); fsa=fx&isS; fxi=ci>=np.where(iS,0.95,0.85)
        diag=dict(oof_cov=float(fx.mean()),oof_sel=float(corr[fx].mean()),oof_sensFE=int((fsa&~corr).sum()),indep_cov=float(fxi.mean()),indep_primary=float((fxi&cri).mean()),indep_FE=int((fxi&iS&~cri).sum()))
        row=dict(diag_fixed_085_095=diag,frac=fr,n_aug=int(round(fr*len(AUG))),T=float(Tm),oof4=best,top1_oof4=float(corr.mean()),indep_primary=float((a&cri).mean()),indep_FE=int((a&iS&~cri).sum()))
        out.append(row); print(row,flush=True)
    json.dump(out,open('out/replace_audit/r4A/lc.json','w'),indent=1)
