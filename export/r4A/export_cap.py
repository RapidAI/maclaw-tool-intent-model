# Export a logit-average ensemble of capacity-sweep heads (depth 1 or 2) as ONE MLP:
# first layers concatenated, middle layers block-diagonal ("hidden" JSON field), output averaged.
import os, json, sys, pickle, numpy as np, hashlib
from scipy.linalg import block_diag
mem=[(m.split(':')[0],int(m.split(':')[1])) for m in sys.argv[1].split(',')]; tag=sys.argv[2]; T_,tau,tau_s=map(float,sys.argv[3:6])
D={}
for p in os.environ['LEV_PKLS'].split(','): D.update(pickle.load(open(p,'rb'))['res'])
k=len(mem); Ws=[]; lab=None
for r,s in mem:
    l,_,W=D[(r,s,'final',0)]; lab=lab or l; assert l==lab; Ws.append(W)
nl=len(Ws[0])//2
W1=np.concatenate([W[0] for W in Ws],1); B1=np.concatenate([W[1] for W in Ws])
mids=[(block_diag(*[W[2*i] for W in Ws]),np.concatenate([W[2*i+1] for W in Ws])) for i in range(1,nl-1)]
W2=np.concatenate([W[-2]/k for W in Ws],0); B2=sum(W[-1] for W in Ws)/k
meta=json.load(open('meta_head.json')); f=lambda a:[float('%.7g'%x) for x in a]
out=dict(meta,labels=lab,temperature=T_,tau=tau,tau_s=tau_s,
         calibration=f'r4 capacity ensemble {mem}: logit average merged (first hidden {W1.shape[1]}, depth {nl-1}); T on pooled LOFO OOF; locked tau rule',
         note='r4 A capacity sweep (FE-cost lambda 0.5 + LS 0.05, dropout); chosen on OOF/old-CV/indep only',
         w1=[f(row) for row in W1.T],b1=f(B1),w2=[f(row) for row in W2.T],b2=f(B2))
if mids: out['hidden']=[dict(w=[f(row) for row in w.T],b=f(b)) for w,b in mids]
out.pop('oof_tau_sweep',None)
p=f'out/replace_audit/r4A/head_{tag}.json'; json.dump(out,open(p,'w')); print(p,hashlib.sha256(open(p,'rb').read()).hexdigest(),W1.shape,[m[0].shape for m in mids])
