# Fold standardisation into layer 1, recompute ETR AUC in numpy (must match GPU 0.9195), ETR bands (same rule as r3), export Go JSON.
import json,math,hashlib,numpy as np
z=np.load('mlp_ISIM.npz'); mu,sd,W1,b1,W2,b2=[z[k].astype(np.float64) for k in ('mu','sd','W1','b1','W2','b2')]
W1f=W1/sd; b1f=b1-(W1*mu/sd).sum(1)
def load(prefix):
    keys=[l.rstrip('\n') for l in open(prefix+'.keys')]; V=np.fromfile(prefix+'.f32','<f4').reshape(len(keys),-1); return {k:V[i] for i,k in enumerate(keys)}
VE=load('vec_eval'); E=[json.loads(l) for l in open('interrupt.jsonl')]
split=lambda i:'train' if int(hashlib.md5(i.encode()).hexdigest(),16)%2==0 else 'test'
ETR=[p for p in E if split(p['id'])=='train']
nz=lambda X:X/np.linalg.norm(X,axis=1,keepdims=True)
def score(P,V):
    U=nz(np.stack([V['IS|'+p['task']] for p in P]).astype(np.float64)); M=nz(np.stack([V['IM|'+p['message']] for p in P]).astype(np.float64))
    X=np.hstack([U,M,np.abs(U-M),U*M,(U*M).sum(1,keepdims=True)]); H=np.maximum(0,X@W1f.T+b1f); return 1/(1+np.exp(-(H@W2+b2[0])))
def auc(s,y):
    s=np.asarray(s);y=np.asarray(y,bool);o=np.argsort(s,kind='mergesort');r=np.empty(len(s));r[o]=np.arange(1,len(s)+1)
    _,inv,c=np.unique(s,return_inverse=True,return_counts=True);r=(np.bincount(inv,r)/c)[inv];n1=y.sum();return (r[y].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1))
sR=score(ETR,VE); yR=np.array([p['label']=='related' for p in ETR])
print('ETR AUC numpy (folded):',round(auc(sR,yR),4))
un=sorted(sR[~yR]); rl=sorted(sR[yR]); hi=un[min(len(un)-1,math.ceil(0.95*len(un)))]; lo=min(rl[int(0.05*len(rl))],hi)
print('ETR bands high',hi,'low',lo)
r3=json.load(open('../int3/interrupt_relevance_qwen3go.json'))
m=dict(format='maclaw-interrupt-relevance-v1',kind='mlp',embedder_model_id=r3['embedder_model_id'],dims=1024,
       task_instruction=r3['task_instruction'],message_instruction=r3['message_instruction'],hidden=int(W1.shape[0]),
       w=[float('%.7g'%x) for x in W1f.ravel()],b1=[float('%.7g'%x) for x in b1f],w2=[float('%.7g'%x) for x in W2],b=float(b2[0]),
       high=float(hi),low=float(lo),
       note="interrupt relevance r4: MLP(4097->64 ReLU->1, AdamW wd 0.1, 60 ep, seed 0) over [u,v,|u-v|,u*v,cos] of IS/IM-instructed Qwen3 vectors; trained on fresh Llama/GLM pairs (n=1222) only; chosen by eval-train-half AUC (0.919) among 2 cosine-bases x {cosine, logistic, mlp}; bands on eval train half")
json.dump(m,open('interrupt_relevance_qwen3go_r4.json','w'),separators=(',',':'))
# parity file: ETR raw scores from the exported (rounded) weights
W1r=np.array(m['w']).reshape(m['hidden'],-1); b1r=np.array(m['b1']); W2r=np.array(m['w2'])
W1f,b1f,W2=W1r,b1r,W2r; sR2=score(ETR,VE); print('max|d| after rounding',np.abs(sR2-sR).max(),'AUC',round(auc(sR2,yR),4))
with open('parity_relevance_r4.jsonl','w') as f:
    for p,s in zip(ETR,sR2): f.write(json.dumps({'id':p['id'],'task':'Instruct: %s\nQuery: %s'%(m['task_instruction'],p['task']),'message':'Instruct: %s\nQuery: %s'%(m['message_instruction'],p['message']),
        'u':VE['IS|'+p['task']].tolist(),'v':VE['IM|'+p['message']].tolist(),'raw':float(s)},ensure_ascii=False)+'\n')
