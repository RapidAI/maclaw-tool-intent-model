# r4 B: pair-scorer search on GPU (torch). Train ONLY on fresh F (int3/train.jsonl, Llama/GLM gen);
# select ONLY on ETR (eval train half, md5 split); ETE (test half) and confirm_r4 B never touched here.
# Candidates: (task-inst, msg-inst) bases x scorer {cosine, logistic(full|dp), mlp(h=64|256)} x ensemble-with-cosine weight a.
# Output: search_b.json (ETR AUC per candidate, per-lang ETR AUC) + best weights for export.
import json, hashlib, sys, numpy as np, torch, itertools, math, os
torch.manual_seed(0); np.random.seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
D = os.path.dirname(os.path.abspath(__file__))
def load(prefix):
    keys=[l.rstrip('\n') for l in open(prefix+'.keys')]
    V=np.fromfile(prefix+'.f32',dtype='<f4').reshape(len(keys),-1); return {k:V[i] for i,k in enumerate(keys)}
VE=load(f'{D}/vec_eval'); VF=load(f'{D}/vec_train')
E=[json.loads(l) for l in open(f'{D}/interrupt.jsonl')]; F=[json.loads(l) for l in open(f'{D}/train.jsonl')]
split=lambda i: 'train' if int(hashlib.md5(i.encode()).hexdigest(),16)%2==0 else 'test'
ETR=[p for p in E if split(p['id'])=='train']
INT=['raw','IT','IM','IS','SIM','CLS']
nz=lambda X: X/np.linalg.norm(X,axis=1,keepdims=True)
def vecs(P,V,tv,mv):
    U=nz(np.stack([V[f'{tv}|{p["task"]}'] for p in P])); M=nz(np.stack([V[f'{mv}|{p["message"]}'] for p in P])); return U,M
Y=lambda P: np.array([p['label']=='related' for p in P],dtype=np.float32)
def auc(s,y):
    s=np.asarray(s,float); y=np.asarray(y,bool); o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    # average ties
    _,inv,cnt=np.unique(s,return_inverse=True,return_counts=True); sums=np.bincount(inv,r); r=(sums/cnt)[inv]
    n1=y.sum(); n0=len(y)-n1; return (r[y].sum()-n1*(n1+1)/2)/(n1*n0)
def feats(U,M,fs):
    parts={'u':U,'v':M,'ad':np.abs(U-M),'pr':U*M,'cos':(U*M).sum(1,keepdims=True)}
    return np.hstack([parts[k] for k in fs]).astype(np.float32)
FS={'full':['u','v','ad','pr','cos'],'dp':['ad','pr','cos']}
yF=Y(F); yR=Y(ETR); langR=np.array([p.get('lang','?') for p in ETR])
# group k-fold on F by task (for early-stopping / λ choice inside F only)
gid={}; g=np.array([gid.setdefault(p['task'],len(gid)) for p in F]); fold=np.random.RandomState(0).permutation(len(gid))[g]%5
def train_torch(X,y,kind,wd,epochs=60,hid=64,drop=0.2,seed=0):
    torch.manual_seed(seed)
    X=torch.tensor(X,device=dev); y=torch.tensor(y,device=dev)
    mu=X.mean(0,keepdim=True); sd=X.std(0,keepdim=True)+1e-6; Xs=(X-mu)/sd
    d=X.shape[1]
    if kind=='logistic': net=torch.nn.Linear(d,1)
    else: net=torch.nn.Sequential(torch.nn.Dropout(drop),torch.nn.Linear(d,hid),torch.nn.ReLU(),torch.nn.Dropout(drop),torch.nn.Linear(hid,1))
    net=net.to(dev); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=wd)
    n=len(y); bs=128
    for ep in range(epochs):
        net.train(); perm=torch.randperm(n,device=dev)
        for i in range(0,n,bs):
            idx=perm[i:i+bs]; opt.zero_grad()
            loss=torch.nn.functional.binary_cross_entropy_with_logits(net(Xs[idx]).squeeze(1),y[idx]); loss.backward(); opt.step()
    net.eval()
    def score(Xn):
        with torch.no_grad(): return torch.sigmoid(net((torch.tensor(Xn,device=dev)-mu)/sd).squeeze(1)).cpu().numpy()
    return net,mu,sd,score
def cvF(X,kind,wd,**kw):
    s=np.zeros(len(yF))
    for k in range(5):
        tr=fold!=k; _,_,_,sc=train_torch(X[tr],yF[tr],kind,wd,**kw); s[~tr]=sc(X[~tr])
    return auc(s,yF)
res=[]
# bases: production raw/raw, round-3 winner IS/IM, plus top cosine combos by F AUC
cos_rows=[]
for tv in INT:
    for mv in INT:
        UF,MF=vecs(F,VF,tv,mv); UR,MR=vecs(ETR,VE,tv,mv)
        cos_rows.append(dict(tv=tv,mv=mv,aucF=auc((UF*MF).sum(1),yF),aucR=auc((UR*MR).sum(1),yR)))
cos_rows.sort(key=lambda r:-r['aucF'])
bases=[('IS','IM'),('raw','raw')]+[(r['tv'],r['mv']) for r in cos_rows[:4] if (r['tv'],r['mv']) not in (('IS','IM'),('raw','raw'))][:3]
print('bases',bases,flush=True)
def z(x): x=np.asarray(x,float); return (x-x.mean())/(x.std()+1e-9)
for tv,mv in bases:
    UF,MF=vecs(F,VF,tv,mv); UR,MR=vecs(ETR,VE,tv,mv); cR=(UR*MR).sum(1); cF=(UF*MF).sum(1)
    res.append(dict(tv=tv,mv=mv,kind='cosine',aucR=auc(cR,yR),aucF=auc(cF,yF),
                    aucR_lang={L:float(auc(cR[langR==L],yR[langR==L])) for L in set(langR) if (yR[langR==L].min()!=yR[langR==L].max())}))
    for fs in ('dp','full'):
        XF=feats(UF,MF,FS[fs]); XR=feats(UR,MR,FS[fs])
        for kind,wd,extra in [('logistic',1e-2,{}),('logistic',1e-1,{}),('logistic',1.0,{}),
                              ('mlp',1e-2,dict(hid=64)),('mlp',1e-1,dict(hid=64)),('mlp',1e-2,dict(hid=256))]:
            a_cv=cvF(XF,kind,wd,**extra)
            net,mu,sd,sc=train_torch(XF,yF,kind,wd,**extra); sR=sc(XR)
            row=dict(tv=tv,mv=mv,kind=kind,fs=fs,wd=wd,hid=extra.get('hid'),cvF=float(a_cv),aucR=float(auc(sR,yR)))
            # ensemble with cosine (z-scored on ETR... use rank-free: a*z(cos)+(1-a)*z(model)); a grid chosen on ETR
            best=(row['aucR'],0.0)
            for a in (0.25,0.5,0.75):
                e=a*z(cR)+(1-a)*z(sR); best=max(best,(float(auc(e,yR)),a))
            row['ens_aucR'],row['ens_a']=best
            row['aucR_lang']={L:float(auc(sR[langR==L],yR[langR==L])) for L in set(langR) if (yR[langR==L].min()!=yR[langR==L].max())}
            res.append(row); print({k:(round(v,3) if isinstance(v,float) else v) for k,v in row.items() if k!='aucR_lang'},flush=True)
json.dump(dict(cosine_scan=cos_rows,cands=res,langs_ETR={L:int((langR==L).sum()) for L in set(langR)}),open(f'{D}/search_b.json','w'),indent=1)
best=max(res,key=lambda r:max(r['aucR'],r.get('ens_aucR',0)))
print('BEST by ETR (incl. ensemble):',best,flush=True)
