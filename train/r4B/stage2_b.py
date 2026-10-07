# r4 B stage 2 (train-only choices): final fits on F; ETR AUC for (a) IS/IM MLP, (b) mean-prob ensemble IS/IM MLP + raw/raw MLP,
# (c) language routing (no-CJK pair -> raw/raw MLP), (d) 5-seed variance of (a). Exports weights (npz) of the fitted models.
import numpy as np, json, re, torch
exec(open('search_b.py').read().split('res=[]')[0])
cjk=re.compile(r'[\u3400-\u9fff]')
isen=lambda p: not (cjk.search(p['task']) or cjk.search(p['message']))
enR=np.array([isen(p) for p in ETR]); print('ETR no-CJK',enR.sum(),'vs lang=en',(langR=='en').sum(),'agree',int(((langR=='en')==enR).sum()),flush=True)
def fit(tv,mv,wd,hid,seed=0):
    UF,MF=vecs(F,VF,tv,mv); UR,MR=vecs(ETR,VE,tv,mv)
    net,mu,sd,sc=train_torch(feats(UF,MF,FS['full']),yF,'mlp',wd,hid=hid,seed=seed); return net,mu,sd,sc(feats(UR,MR,FS['full']))
out={}
nA,muA,sdA,sA=fit('IS','IM',0.1,64); nB,muB,sdB,sB=fit('raw','raw',0.1,64)
out['A_ISIM']=auc(sA,yR); out['B_rawraw']=auc(sB,yR)
for w in (0.25,0.5,0.75): out[f'ens_w{w}']=auc(w*sA+(1-w)*sB,yR)
out['route_noCJK_raw']=auc(np.where(enR,sB,sA),yR)
out['A_en']=auc(sA[enR],yR[enR]); out['B_en']=auc(sB[enR],yR[enR])
seeds=[auc(fit('IS','IM',0.1,64,seed=s)[3],yR) for s in range(1,5)]; out['A_seeds1-4']=seeds
print(json.dumps(out,indent=1),flush=True)
def export(net,mu,sd,path):
    L=[m for m in net if isinstance(m,torch.nn.Linear)]
    np.savez(path,mu=mu.cpu().numpy()[0],sd=sd.cpu().numpy()[0],W1=L[0].weight.detach().cpu().numpy(),b1=L[0].bias.detach().cpu().numpy(),
             W2=L[1].weight.detach().cpu().numpy()[0],b2=L[1].bias.detach().cpu().numpy())
export(nA,muA,sdA,f'{D}/mlp_ISIM.npz'); export(nB,muB,sdB,f'{D}/mlp_rawraw.npz')
np.save(f'{D}/sA_ETR.npy',sA); np.save(f'{D}/sB_ETR.npy',sB)
json.dump(out,open(f'{D}/stage2_b.json','w'),indent=1)
