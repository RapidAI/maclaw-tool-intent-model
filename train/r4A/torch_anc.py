# r4 s9 LEVER 3a: + 48 L2-anchor max-cos features (standardised; self/near-dup anchors cos>=0.98 excluded for TRAIN rows only) + FE-cost. Torch MLP 1024->256 ReLU, Adam lr 1e-3, batch 200, L2 alpha 1e-3 (sklearn-style), 200 epochs, + label smoothing eps. Outputs levers-format pkl.
import json, pickle, sys, numpy as np, torch
lam_list=[float(e) for e in sys.argv[1].split(',')]; eps_list=[0.05]; seeds=[int(x) for x in sys.argv[2].split(',')]; recs=sys.argv[3].split(','); outp=sys.argv[4]
X=np.load('allvec.npy'); ids=json.load(open('allvec.ids.json')); X/=np.linalg.norm(X,axis=1,keepdims=True); pos={i:k for k,i in enumerate(ids)}
D=pickle.load(open('out/replace_audit/r4A/folds.pkl','rb')); LAB=D['LAB']; dev='cuda'
SENS=set(json.load(open('data/intent_meta.json'))['sensitive']); Smask=torch.tensor([l in SENS for l in LAB],device=dev,dtype=torch.float32)
A=json.load(open('anchors_qwen3.json')); AV=np.array(A['vecs'],np.float32); AL=sum([[l]*c for l,c in zip(A['labels'],A['counts'])],[]); AU=list(A['labels'])
G=np.array([[1.0 if a==u else 0.0 for u in AU] for a in AL],np.float32)  # anchor->label one-hot
S=X@AV.T  # (N,534) cos (X already L2-normalised)
def feats(S,train):
    S=S.copy()
    if train: S[S>=0.98]=-1.0
    return np.stack([np.where(G[:,j]>0,S,-1).max(1) for j in range(len(AU))],1)
Ftr=feats(S,True); Fte=feats(S,False)
Xt=torch.tensor(X,device=dev); Ftr_t=torch.tensor(Ftr,device=dev); Fte_t=torch.tensor(Fte,device=dev)
res={}
for eps in eps_list:
 for lam in lam_list:
  for seed in seeds:
    for (rec,kind,k),((tr_ids,y),te_ids) in D['folds'].items():
        if rec not in recs: continue
        torch.manual_seed(seed); n=len(y); C=len(LAB)
        xi=torch.tensor([pos[i] for i in tr_ids],device=dev); yt=torch.tensor(y,device=dev)
        mu=Ftr_t[xi].mean(0); sd=Ftr_t[xi].std(0)+1e-6
        net=torch.nn.Sequential(torch.nn.Linear(1024+len(AU),256),torch.nn.ReLU(),torch.nn.Linear(256,C)).to(dev)
        opt=torch.optim.Adam(net.parameters(),lr=1e-3)
        for ep in range(200):
            perm=torch.randperm(n,device=dev)
            for b in range(0,n,200):
                j=perm[b:b+200]; opt.zero_grad()
                z=net(torch.cat([Xt[xi[j]],(Ftr_t[xi[j]]-mu)/sd],1)); loss=torch.nn.functional.cross_entropy(z,yt[j],label_smoothing=eps)
                pr=torch.softmax(z,1); m=Smask.unsqueeze(0).expand_as(pr).clone(); m[torch.arange(len(j),device=dev),yt[j]]=0
                loss=loss+lam*(pr*m).sum(1).mean()
                loss=loss+1e-3/2*sum((p**2).sum() for nm,p in net.named_parameters() if 'weight' in nm)/n
                loss.backward(); opt.step()
        with torch.no_grad():
            ti=torch.tensor([pos[i] for i in te_ids],device=dev); Z=net(torch.cat([Xt[ti],(Fte_t[ti]-mu)/sd],1)).cpu().numpy().astype(np.float64)
        W=None
        if kind=='final':
            W=(dict(mu=mu.cpu().numpy(),sd=sd.cpu().numpy(),labels=AU),net[0].weight.detach().cpu().numpy().T.astype(np.float64),net[0].bias.detach().cpu().numpy().astype(np.float64),net[2].weight.detach().cpu().numpy().T.astype(np.float64),net[2].bias.detach().cpu().numpy().astype(np.float64))
        res[(f'{rec}_anc_fe{lam}',seed,kind,k)]=(LAB,Z,W); print('done',rec,lam,seed,kind,k,flush=True)
pickle.dump(dict(res=res,FAMS=D['FAMS']),open(outp,'wb')); print('saved')
