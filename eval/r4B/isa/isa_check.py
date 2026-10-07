# Band robustness: r4 MLP scores across Qwen3 kernels (avx512 2-pass [ref], avx2 1-pass, Q8P off=float act., arm64 sdot 1-pass) and float32 vs float64 math.
import json,sys,os,numpy as np
m=json.load(open('../interrupt_relevance_qwen3go_r4.json')); H=m['hidden']; W1=np.array(m['w']).reshape(H,-1); b1=np.array(m['b1']); W2=np.array(m['w2']); b=m['b']; hi,lo=m['high'],m['low']
def load(p):
    k=[l.rstrip('\n') for l in open(p+'.keys')]; V=np.fromfile(p+'.f32','<f4').reshape(len(k),-1); return {x:V[i] for i,x in enumerate(k)}
P=[json.loads(l) for l in open('../interrupt.jsonl')]; y=np.array([p['label']=='related' for p in P])
def z_of(V,dt=np.float64):
    U=np.stack([V['IS|'+p['task']] for p in P]).astype(dt); M=np.stack([V['IM|'+p['message']] for p in P]).astype(dt)
    U/=np.linalg.norm(U,axis=1,keepdims=True); M/=np.linalg.norm(M,axis=1,keepdims=True)
    X=np.hstack([U,M,np.abs(U-M),U*M,(U*M).sum(1,keepdims=True)]).astype(dt)
    return (np.maximum(0,X@W1.T.astype(dt)+b1.astype(dt))@W2.astype(dt)+dt(b)).astype(np.float64)
sig=lambda z:1/(1+np.exp(-z))
band=lambda s:np.where(s>=hi,2,np.where(s<lo,0,1))
def auc(s,yy):
    o=np.argsort(s,kind='mergesort');r=np.empty(len(s));r[o]=np.arange(1,len(s)+1);_,inv,c=np.unique(s,return_inverse=True,return_counts=True);r=(np.bincount(inv,r)/c)[inv];n1=yy.sum();return (r[yy].sum()-n1*(n1+1)/2)/(n1*(len(yy)-n1))
ref=load('v_avx512'); zr=z_of(ref); sr=sig(zr); br=band(sr)
print('ref avx512: AUC %.4f  logit(high)=%.3f logit(low)=%.3f  frac high/mid/low %s  ties at p==1.0: %d'%(auc(sr,y),np.log(hi/(1-hi)),np.log(lo/(1-lo)),np.bincount(br,minlength=3)/len(br),int((sr==1.0).sum())))
# distance of each item's logit to nearest band edge
zh,zl=np.log(hi/(1-hi)),np.log(lo/(1-lo)); dmin=np.minimum(abs(zr-zh),abs(zr-zl)); print('logit margin to nearest edge: p1 %.3f p5 %.3f median %.3f'%tuple(np.percentile(dmin,[1,5,50])))
z32=z_of(ref,np.float32); print('float32 math vs float64: max|dlogit| %.2e  max|dp| %.2e  band flips %d'%(abs(z32-zr).max(),abs(sig(z32)-sr).max(),int((band(sig(z32))!=br).sum())))
out={}
for v in ('avx2','off','arm64'):
    f='v_'+v
    if not os.path.exists(f+'.f32') or os.path.getsize(f+'.f32')!=os.path.getsize('v_avx512.f32'): print(v,'missing/incomplete'); continue
    V=load(f); cs=[float(np.dot(V[k],ref[k])/np.linalg.norm(V[k])/np.linalg.norm(ref[k])) for k in ref]
    z=z_of(V); s=sig(z); bb=band(s)
    r=dict(vec_cos_min=min(cs),vec_cos_mean=float(np.mean(cs)),max_dlogit=float(abs(z-zr).max()),p95_dlogit=float(np.percentile(abs(z-zr),95)),max_dp=float(abs(s-sr).max()),
           band_flips=int((bb!=br).sum()),n=len(P),AUC=float(auc(s,y)))
    cr=(lambda VV:np.array([np.dot(VV['IS|'+q['task']],VV['IM|'+q['message']])/np.linalg.norm(VV['IS|'+q['task']])/np.linalg.norm(VV['IM|'+q['message']]) for q in P]))
    c0,c1=cr(ref),cr(V); cb=lambda c:np.where(c>=0.6335,2,np.where(c<0.3651,0,1)); r['r3cos_band_flips']=int((cb(c0)!=cb(c1)).sum()); r['r3cos_max_d']=float(abs(c0-c1).max())
    out[v]=r; print(v,json.dumps(r))
json.dump(out,open('isa_check.json','w'),indent=1)
