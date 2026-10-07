# Embedding-ensemble variants (mean cosine over task-instruction set x doc-format set), TRAIN half only.
import itertools,json,numpy as np
exec(open('select_c.py').read().split('res=[]')[0])
def rows_ens(TVs,DVs):
    out={}
    for i in tr:
        c=np.zeros(len(names))
        for tv in TVs:
            q=nz(V[f'cod:{tv}|{i}'])
            for dv in DVs: c+=nz(np.stack([V[f'doc:{dv}|{n}'] for n in names]))@q
        r=dict(Q[i]); r['cos']=list(c/(len(TVs)*len(DVs))); out[i]=r
    return out
T=['CQ','CT2','CT3','CT4','WQ']; D=['DT','tool','name2','raw']
res=[]
for nt in (1,2,3):
    for TVs in itertools.combinations(T,nt):
        for nd in (1,2,3):
            for DVs in itertools.combinations(D,nd):
                if nt==1 and nd==1: continue
                R=rows_ens(TVs,DVs); m=MRR(list(R.values())); f,b=calib(R); res.append(dict(TVs=TVs,DVs=DVs,MRR=m,F1=f,base=b))
res.sort(key=lambda r:(-r['MRR'],-r['F1']))
for r in res[:12]: print(r)
json.dump(res,open('ens_c.json','w'),indent=1)
