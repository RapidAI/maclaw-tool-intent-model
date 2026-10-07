# ONE-SHOT B eval (config frozen: interrupt_relevance_qwen3go_r4.json, sha 7d549478...). Existing test half (ETE) + B_confirm.
import json,math,hashlib,random,numpy as np
m=json.load(open('interrupt_relevance_qwen3go_r4.json')); H=m['hidden']; W1=np.array(m['w']).reshape(H,-1); b1=np.array(m['b1']); W2=np.array(m['w2']); b=m['b']
nz=lambda X:X/np.linalg.norm(X,axis=1,keepdims=True)
def mlp(U,M):
    U=nz(U.astype(np.float64));M=nz(M.astype(np.float64)); X=np.hstack([U,M,np.abs(U-M),U*M,(U*M).sum(1,keepdims=True)])
    return 1/(1+np.exp(-(np.maximum(0,X@W1.T+b1)@W2+b)))
def load(prefix):
    keys=[l.rstrip('\n') for l in open(prefix+'.keys')]; V=np.fromfile(prefix+'.f32','<f4').reshape(len(keys),-1); return {k:V[i] for i,k in enumerate(keys)}
def auc(s,y):
    s=np.asarray(s);y=np.asarray(y,bool);o=np.argsort(s,kind='mergesort');r=np.empty(len(s));r[o]=np.arange(1,len(s)+1)
    _,inv,c=np.unique(s,return_inverse=True,return_counts=True);r=(np.bincount(inv,r)/c)[inv];n1=y.sum();return (r[y].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1))
def boot(q,g,y,B=2000):
    rnd=np.random.RandomState(7);d=[];n=len(y)
    for _ in range(B):
        i=rnd.randint(0,n,n)
        if y[i].all() or (~y[i]).all(): continue
        d.append(auc(q[i],y[i])-auc(g[i],y[i]))
    d=np.sort(d);return d[int(.025*len(d))],d[int(.975*len(d))-1]
def bands(s,y,hi,lo): return dict(rel_high=float((s[y]>=hi).mean()),rel_low=float((s[y]<lo).mean()),un_low=float((s[~y]<lo).mean()),un_high=float((s[~y]>=hi).mean()))
out={}
# existing test half
E=[json.loads(l) for l in open('interrupt.jsonl')]; split=lambda i:'train' if int(hashlib.md5(i.encode()).hexdigest(),16)%2==0 else 'test'
ETE=[p for p in E if split(p['id'])=='test']; VE=load('vec_eval')
G={r['id']:r['rel'] for r in map(json.loads,open('../cap3/sig_interrupt_gemma.jsonl'))}
q=mlp(np.stack([VE['IS|'+p['task']] for p in ETE]),np.stack([VE['IM|'+p['message']] for p in ETE])); g=np.array([G[p['id']] for p in ETE]); y=np.array([p['label']=='related' for p in ETE])
qc=(nz(np.stack([VE['IS|'+p['task']] for p in ETE]))*nz(np.stack([VE['IM|'+p['message']] for p in ETE]))).sum(1)
out['ETE']=dict(n=len(y),qwen3_r4=auc(q,y),qwen3_r3_cos=auc(qc,y),gemma=auc(g,y),dAUC=auc(q,y)-auc(g,y),ci=boot(q,g,y),bands_q=bands(q,y,m['high'],m['low']),bands_g=bands(g,y,0.60,0.30))
# confirm
C=[json.loads(l) for l in open('../confirm_r4/B_confirm.jsonl')]; VQ=load('vconf_q'); VG=load('vconf_g')
q=mlp(np.stack([VQ['IS|'+p['task']] for p in C]),np.stack([VQ['IM|'+p['message']] for p in C]))
qc=(nz(np.stack([VQ['IS|'+p['task']] for p in C]))*nz(np.stack([VQ['IM|'+p['message']] for p in C]))).sum(1)
g=(nz(np.stack([VG[p['task']] for p in C]))*nz(np.stack([VG[p['message']] for p in C]))).sum(1); y=np.array([p['label']=='related' for p in C])
out['B_confirm']=dict(n=len(y),qwen3_r4=auc(q,y),qwen3_r3_cos=auc(qc,y),gemma=auc(g,y),dAUC=auc(q,y)-auc(g,y),ci=boot(q,g,y),bands_q=bands(q,y,m['high'],m['low']),bands_g=bands(g,y,0.60,0.30))
json.dump(out,open('oneshot_b.json','w'),indent=1,default=float); print(json.dumps(out,indent=1,default=float))
