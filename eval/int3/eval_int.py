# Stage 2 (after select_int.py): best candidate by ETR AUC (ETE never used for selection) -> fit on F (logistic) ->
# bands on ETR (≤5% unrelated at/above high, ≤5% related below low; same rule as Gemma's recalibration in
# cap3/eval_cap3.py) -> report ETE (n=124, primary) and all-262 (includes the selection half; secondary).
# Exports the production RelevanceModel JSON (standardisation folded into w, b).
import json, hashlib, math, random, sys, numpy as np
sys.path.insert(0,'.'); from vec import load; from lr import auc
from variants import INT
def fit(X,y,lam,iters=400):
    # same objective as lr.fit; Lipschitz const by power iteration (lr.fit's exact SVD norm was the bottleneck)
    v=np.random.default_rng(0).standard_normal(X.shape[1])
    for _ in range(30): v=X.T@(X@v); v/=np.linalg.norm(v)
    s2=np.linalg.norm(X@v)**2*1.05
    n,d=X.shape; yf=y.astype(float); w=np.zeros(d); b=0.0; vw=np.zeros(d); vb=0.0; step=1.0/(0.25*s2/n+lam/n)
    for t in range(iters):
        wl=w+0.9*vw; bl=b+0.9*vb; p=1/(1+np.exp(-(X@wl+bl))); g=X.T@(p-yf)/n+lam/n*wl; gb=(p-yf).mean()
        vw=0.9*vw-step*g; vb=0.9*vb-step*gb; w=w+vw; b=b+vb
    return w,b; from variants import INT
C='/workspace/maclaw_reranker/out/replace_audit/cap3'
split=lambda i: 'train' if int(hashlib.md5(i.encode()).hexdigest(),16)%2==0 else 'test'
E=[json.loads(l) for l in open(f'{C}/interrupt.jsonl')]; F=[json.loads(l) for l in open('train.jsonl')]
VE=load('vec_eval'); VF=load('vec_train')
G={r['id']:r['rel'] for r in map(json.loads,open(f'{C}/sig_interrupt_gemma.jsonl'))}
Q0={r['id']:r['rel'] for r in map(json.loads,open(f'{C}/sig_interrupt_qwen3.jsonl'))}
ETR=[p for p in E if split(p['id'])=='train']; ETE=[p for p in E if split(p['id'])=='test']
nz=lambda x: x/np.linalg.norm(x)
def vecs(P,V,tv,mv):
    U=np.stack([nz(V[f'{tv}|{p["task"]}']) for p in P])
    M=np.stack([nz(V[f'CTX|{p["task"]}|{p["message"]}'] if mv=='CTX' else V[f'{mv}|{p["message"]}']) for p in P])
    return U,M
FS={'dp':['ad','pr','cos'],'full':['u','v','ad','pr','cos'],'pr':['pr','cos']}
def feats(U,M,fs):
    parts={'u':U,'v':M,'ad':np.abs(U-M),'pr':U*M,'cos':(U*M).sum(1,keepdims=True)}
    return np.hstack([parts[k] for k in FS[fs]])
S=json.load(open('select_int.json')); allc=S['cosine']+S['logistic']
pick=sys.argv[1] if len(sys.argv)>1 else None
best=max(allc,key=lambda c:c['aucR']) if not pick else [c for c in allc if f"{c['kind']}:{c['tv']}:{c['mv']}:{c.get('fs','')}"==pick][0]
out=[]; P=lambda *a: (out.append(' '.join(map(str,a))), print(*a,flush=True))
P('selected (max ETR AUC over %d candidates):'%len(allc), {k:v for k,v in best.items()})
tv,mv=best['tv'],best['mv']
if best['kind']=='cosine':
    score=lambda P_,V: (lambda U,M:(U*M).sum(1))(*vecs(P_,V,tv,mv)); model=None
else:
    XF=feats(*vecs(F,VF,tv,mv),best['fs']); mu=XF.mean(0); sd=XF.std(0)+1e-6
    w,b=fit((XF-mu)/sd,np.array([p['label']=='related' for p in F]),best['lam'])
    wr=w/sd; br=b-(w*mu/sd).sum()
    score=lambda P_,V: 1/(1+np.exp(-(feats(*vecs(P_,V,tv,mv),best['fs'])@wr+br)))
sR=dict(zip([p['id'] for p in ETR],score(ETR,VE))); sT=dict(zip([p['id'] for p in ETE],score(ETE,VE)))
lab={p['id']:p['label']=='related' for p in E}
def calib(s,ids):
    un=sorted(s[i] for i in ids if not lab[i]); rl=sorted(s[i] for i in ids if lab[i])
    hi=un[min(len(un)-1,math.ceil(0.95*len(un)))]; lo=rl[int(0.05*len(rl))]; return hi,min(lo,hi)
def bands(s,ids,hi,lo):
    rel=[i for i in ids if lab[i]]; un=[i for i in ids if not lab[i]]
    return dict(rel_high=sum(s[i]>=hi for i in rel)/len(rel), rel_low=sum(s[i]<lo for i in rel)/len(rel),
                un_low=sum(s[i]<lo for i in un)/len(un), un_high=sum(s[i]>=hi for i in un)/len(un))
trI=[p['id'] for p in ETR]; teI=[p['id'] for p in ETE]
hq,lq=calib(sR,trI); hg,lg=calib(G,trI); h0,l0=calib(Q0,trI)
def A(s,ids): return auc([s[i] for i in ids],[lab[i] for i in ids])
res=dict(selected=best,bands_qwen3=(hq,lq),bands_gemma=(hg,lg),bands_qwen3_raw=(h0,l0))
P(f"\n== TEST half n={len(teI)} (related {sum(lab[i] for i in teI)})")
for name,s,(h,l) in (('qwen3 new',sT,(hq,lq)),('qwen3 production raw cos',Q0,(h0,l0)),('gemma production cos',G,(hg,lg))):
    bt=bands(s,teI,h,l); res[name]=dict(auc=A(s,teI),**bt)
    P(f"  {name:26s} AUC {A(s,teI):.3f} | ETR bands high>={h:.3f} low<{l:.3f}: related→high {bt['rel_high']:.3f} related→low {bt['rel_low']:.3f} (wrong split) | unrelated→low {bt['un_low']:.3f} unrelated→high {bt['un_high']:.3f} (wrong merge)")
gp=bands(G,teI,0.60,0.30); P(f"  gemma production bands .60/.30: related→high {gp['rel_high']:.3f} related→low {gp['rel_low']:.3f} | unrelated→low {gp['un_low']:.3f} unrelated→high {gp['un_high']:.3f}")
def boot(fn,ids,B=2000,seed=7):
    r=random.Random(seed); v=[]
    for _ in range(B):
        smp=[ids[r.randrange(len(ids))] for _ in ids]; v.append(fn(smp))
    v.sort(); return v[int(.025*B)],v[int(.975*B)]
def dA(smp):
    y=[lab[i] for i in smp]
    if all(y) or not any(y): return 0.0
    return auc([sT[i] for i in smp],y)-auc([G[i] for i in smp],y)
d=A(sT,teI)-A(G,teI); lo_,hi_=boot(dA,teI); P(f"  ΔAUC qwen3new − gemma = {d:+.3f}  95% CI [{lo_:+.3f},{hi_:+.3f}]"); res['dAUC']=(d,lo_,hi_)
for key,nm in (('un_high','wrong-merge (unrelated→high)'),('rel_low','wrong-split (related→low)'),('rel_high','related→high'),('un_low','unrelated→low')):
    def fn(smp,key=key):
        rq=[i for i in smp if (lab[i] if key.startswith('rel') else not lab[i])]
        if not rq: return 0.0
        return bands(sT,smp,hq,lq)[key]-bands(G,smp,hg,lg)[key] if any(lab[i] for i in smp) and not all(lab[i] for i in smp) else 0.0
    dd=res['qwen3 new'][key]-res['gemma production cos'][key]; a_,b_=boot(fn,teI,B=1000); P(f"  Δ {nm:30s} {dd:+.3f}  95% CI [{a_:+.3f},{b_:+.3f}]"); res['d_'+key]=(dd,a_,b_)
allI=trI+teI; sA={**sR,**sT}
P(f"\n== all 262 (includes the ETR selection half): qwen3 new AUC {A(sA,allI):.3f} | qwen3 raw {A(Q0,allI):.3f} | gemma {A(G,allI):.3f}")
res['all262']=dict(qwen3=A(sA,allI),qwen3raw=A(Q0,allI),gemma=A(G,allI))
inst=lambda v: INT[v] if v in INT else None
doc=dict(format='maclaw-interrupt-relevance-v1',kind=best['kind'],embedder_model_id='Qwen3 Embedding 0.6b:1024:prompt-v1:qwen3-instruct-v1',dims=1024,
         task_instruction=inst(tv) or '',message_instruction=inst(mv) or '',high=float(hq),low=float(lq),
         note=f"interrupt relevance r3: {best}; trained on fresh Llama/GLM pairs (n={len(F)}); bands on eval train half")
if mv=='CTX': doc['note']+=' (CTX message format not supported in Go)'
if best['kind']=='logistic':
    # Go layout is always [u, v, |u-v|, u*v, cos] (4*d+1); absent blocks get zero weights.
    d=1024; full=np.zeros(4*d+1); off={'u':0,'v':d,'ad':2*d,'pr':3*d,'cos':4*d}; k=0
    for blk in FS[best['fs']]:
        n=1 if blk=='cos' else d; full[off[blk]:off[blk]+n]=wr[k:k+n]; k+=n
    assert k==len(wr)
    doc['w']=[float(x) for x in full]; doc['b']=float(br)
json.dump(doc,open('interrupt_relevance_qwen3go.json','w'))
json.dump(res,open('eval_int.json','w'),indent=1,default=float); open('eval_int.txt','w').write('\n'.join(out)+'\n')
# Go parity fixture: 40 test-half pairs, raw (unnormalised) vectors + Python raw score.
U,M=vecs(ETE[:40],VE,tv,mv); sc=score(ETE[:40],VE)
with open('parity_relevance.jsonl','w') as f:
    for p,u,m,s_ in zip(ETE[:40],U,M,sc):
        f.write(json.dumps({'id':p['id'],'u':[float(x) for x in VE[f'{tv}|{p["task"]}']],'v':[float(x) for x in (VE[f'CTX|{p["task"]}|{p["message"]}'] if mv=='CTX' else VE[f'{mv}|{p["message"]}'])],'raw':float(s_)})+'\n')
