# r4 C selection on the coding TRAIN half only (md5 split, n=140). Metric = cosine-ranking MRR (same def as scorecard),
# constraint: train-calibrated fusion F1 >= train F1 of the r3 config (task=WQ doc=raw base=0.40) - 0 (must not drop).
# Also tunes fusion: score=max(wl*bm25, wb*bigram, max(0,cos-base)) with wl,wb in grid (thr 0.15, k 5), chosen on train.
# TEST/C_confirm are NOT touched here.
import json,hashlib,sys,numpy as np,itertools
C='/workspace/maclaw_reranker/out/replace_audit/cap3'
keys=[l.rstrip('\n') for l in open('vec_c.keys')]; V=np.fromfile('vec_c.f32','<f4').reshape(len(keys),-1); V={k:V[i] for i,k in enumerate(keys)}
cands=json.load(open(f'{C}/coding_cands.json')); names=[c['name'] for c in cands]
split=lambda i: 'train' if int(hashlib.md5(i.encode()).hexdigest(),16)%2==0 else 'test'
Q={r['id']:r for r in map(json.loads,open(f'{C}/sig_coding_qwen3.jsonl'))}
ids=sorted(Q); tr=[i for i in ids if split(i)=='train']
nz=lambda x:x/np.linalg.norm(x,axis=-1,keepdims=True)
def rows_for(tv,dv):
    D=nz(np.stack([V[f'doc:{dv}|{n}'] for n in names])); out={}
    for i in tr:
        r=dict(Q[i]); r['cos']=list((D@nz(V[f'cod:{tv}|{i}'])).astype(float)); out[i]=r
    return out
def select(r,base,wl=1,wb=1,thr=0.15,k=5):
    sc=[(max(wl*r['bm25'][j],wb*r['bigram'][j],max(0.0,r['cos'][j]-base)),j) for j in range(len(names))]
    sc=[x for x in sc if x[0]>=thr]; sc.sort(key=lambda x:-x[0]); return [names[j] for _,j in sc[:k]]
def F1(rows,base,wl=1,wb=1):
    tp=fp=fn=0
    for r in rows:
        g=set(r['gold']); s=set(select(r,base,wl,wb)); tp+=len(g&s); fp+=len(s-g); fn+=len(g-s)
    p=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn); return 2*p*rc/(p+rc) if p+rc else 0
def MRR(rows):
    m=0
    for r in rows:
        o=np.argsort(-np.array(r['cos'])); g=set(r['gold']); m+=1/(min(k for k,j in enumerate(o) if names[j] in g)+1)
    return m/len(rows)
calib=lambda R,wl=1,wb=1: max((F1(list(R.values()),b/100,wl,wb),b/100) for b in range(0,80,2))
res=[]
TV=sorted({k.split('|')[0][4:] for k in keys if k.startswith('cod:')}); DV=sorted({k.split('|')[0][4:] for k in keys if k.startswith('doc:')})
for tv in TV:
    for dv in DV:
        R=rows_for(tv,dv); f,b=calib(R); m=MRR(list(R.values())); res.append(dict(tv=tv,dv=dv,F1=f,base=b,MRR=m))
        print(f'task={tv:4s} doc={dv:6s} MRR={m:.3f} F1={f:.3f} base={b:.2f}',flush=True)
ref=[r for r in res if r['tv']=='WQ' and r['dv']=='raw'][0]
ok=[r for r in res if r['F1']>=ref['F1']-1e-9]
best=max(ok,key=lambda r:(r['MRR'],r['F1']))
print('REF r3 (WQ/raw):',ref); print('SELECTED (train, MRR s.t. F1>=ref):',best)
# fusion weight retune for selected config (F1 only; MRR is cosine-only so unaffected)
R=rows_for(best['tv'],best['dv']); fw=[]
for wl,wb in itertools.product((0.6,0.8,1.0,1.2),(0.6,0.8,1.0,1.2)):
    f,b=calib(R,wl,wb); fw.append(dict(wl=wl,wb=wb,F1=f,base=b))
fw.sort(key=lambda r:-r['F1']); print('fusion top:',fw[:5]); print('fusion (1,1):',[r for r in fw if r['wl']==1 and r['wb']==1])
json.dump(dict(variants=res,ref=ref,selected=best,fusion=fw),open('select_c.json','w'),indent=1)
