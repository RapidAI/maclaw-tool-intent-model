# r4 s12 OPTION 4a MATRYOSHKA: input = l2norm(v[:MD]) (r10CAP recipe). Derived from CAPACITY SWEEP (FE-cost lambda 0.5, LS 0.05): width/depth/dropout. Torch MLP 1024->256 ReLU, Adam lr 1e-3, batch 200, L2 alpha 1e-3 (sklearn-style), 200 epochs, + label smoothing eps. Outputs levers-format pkl.
import json, pickle, sys, numpy as np, torch
MD=int(sys.argv[1]); WID=512; DEP=1; DROP=0.3; seeds=[int(x) for x in sys.argv[2].split(',')]; recs=['hard']; outp=sys.argv[3]; lam_list=[0.5]; eps_list=[0.05]
X=np.load('allvec.npy'); ids=json.load(open('allvec.ids.json')); X/=np.linalg.norm(X,axis=1,keepdims=True); X=np.ascontiguousarray(X[:,:MD]); X/=np.linalg.norm(X,axis=1,keepdims=True); pos={i:k for k,i in enumerate(ids)}
D=pickle.load(open('out/replace_audit/r4A/folds.pkl','rb')); LAB=D['LAB']; dev='cuda'
SENS=set(json.load(open('data/intent_meta.json'))['sensitive']); Smask=torch.tensor([l in SENS for l in LAB],device=dev,dtype=torch.float32)
Xt=torch.tensor(X,device=dev)
res={}
for eps in eps_list:
 for lam in lam_list:
  for seed in seeds:
    for (rec,kind,k),((tr_ids,y),te_ids) in D['folds'].items():
        if rec not in recs: continue
        torch.manual_seed(seed); n=len(y); C=len(LAB)
        xi=torch.tensor([pos[i] for i in tr_ids],device=dev); yt=torch.tensor(y,device=dev)
        layers=[torch.nn.Linear(X.shape[1],WID),torch.nn.ReLU(),torch.nn.Dropout(DROP)]+([torch.nn.Linear(WID,WID),torch.nn.ReLU(),torch.nn.Dropout(DROP)] if DEP==2 else [])+[torch.nn.Linear(WID,C)]
        net=torch.nn.Sequential(*layers).to(dev)
        opt=torch.optim.Adam(net.parameters(),lr=1e-3)
        for ep in range(200):
            net.train(); perm=torch.randperm(n,device=dev)
            for b in range(0,n,200):
                j=perm[b:b+200]; opt.zero_grad()
                z=net(Xt[xi[j]]); loss=torch.nn.functional.cross_entropy(z,yt[j],label_smoothing=eps)
                pr=torch.softmax(z,1); m=Smask.unsqueeze(0).expand_as(pr).clone(); m[torch.arange(len(j),device=dev),yt[j]]=0
                loss=loss+lam*(pr*m).sum(1).mean()
                loss=loss+1e-3/2*sum((p**2).sum() for nm,p in net.named_parameters() if 'weight' in nm)/n
                loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            Z=net(Xt[torch.tensor([pos[i] for i in te_ids],device=dev)]).cpu().numpy().astype(np.float64)
        W=None
        if kind=='final':
            W=tuple(a for m in net if isinstance(m,torch.nn.Linear) for a in (m.weight.detach().cpu().numpy().T.astype(np.float64),m.bias.detach().cpu().numpy().astype(np.float64)))
        res[(f'mat{MD}_w{WID}_d{DEP}_p{DROP}',seed,kind,k)]=(LAB,Z,W); print('done',rec,lam,seed,kind,k,flush=True)
pickle.dump(dict(res=res,FAMS=D['FAMS']),open(outp,'wb')); print('saved')
