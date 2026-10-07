# Torch MLP 1024->256 ReLU, Adam lr 1e-3, batch 200, L2 alpha 1e-3 (sklearn-style), 200 epochs, + label smoothing eps. Outputs levers-format pkl.
import json, pickle, sys, numpy as np, torch
eps_list=[float(e) for e in sys.argv[1].split(',')]; seed=0
X=np.load('allvec.npy'); ids=json.load(open('allvec.ids.json')); X/=np.linalg.norm(X,axis=1,keepdims=True); pos={i:k for k,i in enumerate(ids)}
D=pickle.load(open('out/replace_audit/r4A/folds.pkl','rb')); LAB=D['LAB']; dev='cuda'
Xt=torch.tensor(X,device=dev)
res={}
for eps in eps_list:
    for (rec,kind,k),((tr_ids,y),te_ids) in D['folds'].items():
        torch.manual_seed(seed); n=len(y); C=len(LAB)
        xi=torch.tensor([pos[i] for i in tr_ids],device=dev); yt=torch.tensor(y,device=dev)
        net=torch.nn.Sequential(torch.nn.Linear(1024,256),torch.nn.ReLU(),torch.nn.Linear(256,C)).to(dev)
        opt=torch.optim.Adam(net.parameters(),lr=1e-3)
        for ep in range(200):
            perm=torch.randperm(n,device=dev)
            for b in range(0,n,200):
                j=perm[b:b+200]; opt.zero_grad()
                loss=torch.nn.functional.cross_entropy(net(Xt[xi[j]]),yt[j],label_smoothing=eps)
                loss=loss+1e-3/2*sum((p**2).sum() for nm,p in net.named_parameters() if 'weight' in nm)/n
                loss.backward(); opt.step()
        with torch.no_grad():
            Z=net(Xt[torch.tensor([pos[i] for i in te_ids],device=dev)]).cpu().numpy().astype(np.float64)
        W=None
        if kind=='final':
            W=(net[0].weight.detach().cpu().numpy().T.astype(np.float64),net[0].bias.detach().cpu().numpy().astype(np.float64),net[2].weight.detach().cpu().numpy().T.astype(np.float64),net[2].bias.detach().cpu().numpy().astype(np.float64))
        res[(f'{rec}_ls{eps}',seed,kind,k)]=(LAB,Z,W); print('done',rec,eps,kind,k,flush=True)
pickle.dump(dict(res=res,FAMS=D['FAMS']),open('out/replace_audit/r4A/levers_ls_raw.pkl','wb')); print('saved')
