# Export a logit-average ensemble of final heads as ONE MLP (hidden concatenated, output weights / k): exact for logit averaging.
import os, json, sys, pickle, numpy as np, hashlib
mem=[tuple(x.split(':')) for x in sys.argv[1].split(',')]; mem=[(r,int(s)) for r,s in mem]; tag=sys.argv[2]; T_,tau,tau_s=map(float,sys.argv[3:6])
D=pickle.load(open(os.environ.get('LEV_PKL','out/replace_audit/r4A/levers_raw.pkl'),'rb'))['res']; k=len(mem)
W1=[];B1=[];W2=[];B2=0; lab=None
for r,s in mem:
    l,_,(w1,b1,w2,b2)=D[(r,s,'final',0)][0],None,D[(r,s,'final',0)][2]
    lab=lab or l; assert l==lab
    W1.append(w1); B1.append(b1); W2.append(w2/k); B2=B2+b2/k
W1=np.concatenate(W1,1); B1=np.concatenate(B1); W2=np.concatenate(W2,0)
meta=json.load(open('meta_head.json'))
f=lambda a:[float('%.7g'%x) for x in a]
out=dict(meta, labels=lab, temperature=T_, tau=tau, tau_s=tau_s,
         calibration=f'r4 ensemble {mem}: logit average merged into one MLP (hidden {W1.shape[1]}); T on pooled LOFO OOF of the ensemble; tau/tau_s by LOCKED r4 rule',
         note=f'r4 A lever: logit-average of {mem} (base=r3W5F recipe, hard=r3W5F+re-judged hard pairs); chosen on OOF/old-CV/indep only',
         w1=[f(row) for row in W1.T], b1=f(B1), w2=[f(row) for row in W2.T], b2=f(B2))
out.pop('oof_tau_sweep',None)
p=f'out/replace_audit/r4A/head_{tag}.json'; json.dump(out,open(p,'w')); print(p,hashlib.sha256(open(p,'rb').read()).hexdigest(),W1.shape)
