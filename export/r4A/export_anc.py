# Export a logit-average ensemble of lever-3a anchor-feature heads as ONE MLP; feature standardisation folded into w1/b1; adds "anchor_features".
import os, json, sys, pickle, numpy as np, hashlib
mem=[(m.split(':')[0],int(m.split(':')[1])) for m in sys.argv[1].split(',')]; tag=sys.argv[2]; T_,tau,tau_s=map(float,sys.argv[3:6])
D={}
for p in os.environ['LEV_PKLS'].split(','): D.update(pickle.load(open(p,'rb'))['res'])
k=len(mem); W1=[];B1=[];W2=[];B2=0; lab=None; flab=None
for r,s in mem:
    l,_,(meta,w1,b1,w2,b2)=D[(r,s,'final',0)]
    lab=lab or l; assert l==lab; flab=flab or meta['labels']; assert meta['labels']==flab
    ed=w1.shape[0]-len(flab); mu=np.asarray(meta['mu'],np.float64); sd=np.asarray(meta['sd'],np.float64)
    w1=w1.copy(); b1=b1-(mu/sd)@w1[ed:,:]; w1[ed:,:]=w1[ed:,:]/sd[:,None]
    W1.append(w1); B1.append(b1); W2.append(w2/k); B2=B2+b2/k
W1=np.concatenate(W1,1); B1=np.concatenate(B1); W2=np.concatenate(W2,0)
meta=json.load(open('meta_head.json')); f=lambda a:[float('%.7g'%x) for x in a]
out=dict(meta,labels=lab,temperature=T_,tau=tau,tau_s=tau_s,anchor_features=dict(labels=list(flab)),
         calibration=f'r4 lever 3a ensemble {mem}: logit average merged (hidden {W1.shape[1]}); input = [l2norm(v), raw L2 max anchor cosine per label]; T on pooled LOFO OOF; locked tau rule',
         note=f'r4 A lever 3a (anchor features + FE-cost lambda 0.5 + LS 0.05); chosen on OOF/old-CV/indep only',
         w1=[f(row) for row in W1.T],b1=f(B1),w2=[f(row) for row in W2.T],b2=f(B2))
out.pop('oof_tau_sweep',None)
p=f'out/replace_audit/r4A/head_{tag}.json'; json.dump(out,open(p,'w')); print(p,hashlib.sha256(open(p,'rb').read()).hexdigest(),W1.shape)
