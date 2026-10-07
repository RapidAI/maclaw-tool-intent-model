# Interrupt relevance round 3: choose Qwen3 relevance scorer WITHOUT touching the eval test half.
#  F   = fresh training pairs (train.jsonl; Llama/GLM generated, cross-family judged, deduped vs the 262 eval items)
#  ETR = eval train half (md5 split, n≈138), ETE = eval test half (n≈124, report only)
# Candidates: cosine on instruction-formatted vectors (task-inst × msg-inst, incl. CTX), and tiny logistic scorers
# trained on F only (features on L2-normalised u,v; standardised; λ by task-grouped 5-fold CV on F).
# Selection = AUC on ETR (scorer weights never see eval); bands calibrated on ETR (≤5% wrong-merge / wrong-split).
import json, hashlib, math, random, sys, numpy as np
sys.path.insert(0,'.'); from vec import load; from lr import auc
import lr
def fit(X,y,lam,iters=400):
    # same objective as lr.fit; Lipschitz const by power iteration (lr.fit's exact SVD norm was the bottleneck)
    v=np.random.default_rng(0).standard_normal(X.shape[1])
    for _ in range(30): v=X.T@(X@v); v/=np.linalg.norm(v)
    s2=np.linalg.norm(X@v)**2*1.05
    n,d=X.shape; yf=y.astype(float); w=np.zeros(d); b=0.0; vw=np.zeros(d); vb=0.0; step=1.0/(0.25*s2/n+lam/n)
    for t in range(iters):
        wl=w+0.9*vw; bl=b+0.9*vb; p=1/(1+np.exp(-(X@wl+bl))); g=X.T@(p-yf)/n+lam/n*wl; gb=(p-yf).mean()
        vw=0.9*vw-step*g; vb=0.9*vb-step*gb; w=w+vw; b=b+vb
    return w,b
from variants import INT
C='/workspace/maclaw_reranker/out/replace_audit/cap3'
split=lambda i: 'train' if int(hashlib.md5(i.encode()).hexdigest(),16)%2==0 else 'test'
E=[json.loads(l) for l in open(f'{C}/interrupt.jsonl')]; F=[json.loads(l) for l in open('train.jsonl')]
VE=load('vec_eval'); VF=load('vec_train')
G={r['id']:r for r in map(json.loads,open(f'{C}/sig_interrupt_gemma.jsonl'))}
ETR=[p for p in E if split(p['id'])=='train']; ETE=[p for p in E if split(p['id'])=='test']
nz=lambda x: x/np.linalg.norm(x)
def vecs(P,V,tv,mv):
    U=np.stack([nz(V[f'{tv}|{p["task"]}']) for p in P])
    M=np.stack([nz(V[f'CTX|{p["task"]}|{p["message"]}'] if mv=='CTX' else V[f'{mv}|{p["message"]}']) for p in P])
    return U,M
Y=lambda P: np.array([p['label']=='related' for p in P])
def feats(U,M,fs):
    cos=(U*M).sum(1,keepdims=True)
    parts={'u':U,'v':M,'ad':np.abs(U-M),'pr':U*M,'cos':cos}
    return np.hstack([parts[k] for k in fs])
FS={'dp':['ad','pr','cos'],'full':['u','v','ad','pr','cos'],'pr':['pr','cos']}
out=[]; Pp=lambda *a: (out.append(' '.join(map(str,a))), print(*a, flush=True))
yF,yR,yT=Y(F),Y(ETR),Y(ETE)
Pp(f'F n={len(F)} (related {yF.sum()}), ETR n={len(ETR)} (related {yR.sum()}), ETE n={len(ETE)} (related {yT.sum()})')
# ---- cosine candidates
cands=[]
for tv in INT:
    for mv in list(INT)+['CTX']:
        sF=(lambda U,M:(U*M).sum(1))(*vecs(F,VF,tv,mv)); sR=(lambda U,M:(U*M).sum(1))(*vecs(ETR,VE,tv,mv))
        cands.append(dict(kind='cosine',tv=tv,mv=mv,aucF=auc(sF,yF),aucR=auc(sR,yR)))
cands.sort(key=lambda c:-c['aucR'])
Pp('\n== cosine candidates (AUC on F / ETR), top 12 by ETR'); [Pp(f"  task={c['tv']:4s} msg={c['mv']:4s}  F {c['aucF']:.3f}  ETR {c['aucR']:.3f}") for c in cands[:12]]
raw=[c for c in cands if c['tv']=='raw' and c['mv']=='raw'][0]; Pp(f"  production (raw/raw): F {raw['aucF']:.3f}  ETR {raw['aucR']:.3f}")
# ---- logistic candidates (base variants: raw/raw + top-3 cosine combos by F AUC, symmetric ones included)
base=[('raw','raw'),(cands[0]['tv'],cands[0]['mv'])]  # production + best cosine combo (by ETR); time-boxed
groups={}; gF=np.array([groups.setdefault(p['task'],len(groups)) for p in F])
rng=np.random.default_rng(0); perm=rng.permutation(len(groups)); fold=perm[gF]%5
def cvF(X,lam):
    s=np.zeros(len(yF))
    for k in range(5):
        tr=fold!=k; mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6
        w,b=fit((X[tr]-mu)/sd,yF[tr],lam); s[~tr]=((X[~tr]-mu)/sd)@w+b
    return auc(s,yF),s
logs=[]
for tv,mv in base:
    UF,MF=vecs(F,VF,tv,mv); UR,MR=vecs(ETR,VE,tv,mv)
    for fs in ('dp','full'):
        XF=feats(UF,MF,FS[fs]); XR=feats(UR,MR,FS[fs])
        best=None
        for lam in (10,100,1000):
            a,_=cvF(XF,lam)
            if best is None or a>best[0]: best=(a,lam)
        mu=XF.mean(0); sd=XF.std(0)+1e-6; w,b=fit((XF-mu)/sd,yF,best[1])
        sR=((XR-mu)/sd)@w+b
        logs.append(dict(kind='logistic',tv=tv,mv=mv,fs=fs,lam=best[1],cvF=best[0],aucR=auc(sR,yR)))
        Pp(f"  logistic task={tv:4s} msg={mv:4s} feats={fs:4s} λ={best[1]:<6} CV-AUC(F) {best[0]:.3f}  ETR {logs[-1]['aucR']:.3f}")
json.dump(dict(cosine=cands,logistic=logs),open('select_int.json','w'),indent=1)
open('select_int.txt','w').write('\n'.join(out)+'\n')
