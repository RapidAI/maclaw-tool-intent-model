import json, sys, numpy as np, itertools
sys.path.insert(0,'/workspace/maclaw_reranker/scripts'); from tool_gold import GOLD
R='/workspace/maclaw_reranker'; D=f'{R}/out/replace_audit/trackA'
lab={}
for f in ('data/intent_all.jsonl','data/indep_test.jsonl'):
    for l in open(f'{R}/{f}'):
        x=json.loads(l); lab[x['id']]=x['intent']
bm={}
for l in open(f'{D}/bm25_all.jsonl'):
    x=json.loads(l); bm[x['id']]=x['bm25'] or {}
tools=[json.loads(l)['id'] for l in open(f'{D}/tooltext.jsonl')]
def lv(f):
    out={}
    for l in open(f):
        x=json.loads(l); v=np.array(x['vec'],dtype=np.float32); out[x['id']]=v/np.linalg.norm(v)
    return out
def docmat(f):
    d=lv(f); return np.stack([d[t] for t in tools])
sets={'train668':[i for i in bm if i.startswith('train-') and lab.get(i) in GOLD],
      'old262':[i for i in bm if i.startswith('test-') and lab.get(i) in GOLD],
      'indep644':[i for i in bm if i.startswith('indep-') and lab.get(i) in GOLD],
      'kimi':[i for i in bm if i.startswith('indep-kimi') and lab.get(i) in GOLD]}
def rank(qv, DM, ids, alpha):
    out={}
    for i in ids:
        cos=DM@qv[i]; b=bm[i]; mx=max(b.values()) if b else 0
        nb=np.array([ (b.get(t,0)/mx if mx>0 else 0) for t in tools])
        s=alpha*nb+(1-alpha)*cos
        order=sorted(range(len(tools)),key=lambda k:(-s[k],tools[k]))
        out[i]=[tools[k] for k in order]
    return out
def rec(rk, ids, k): return np.mean([any(t in GOLD[lab[i]] for t in rk[i][:k]) for i in ids])
def mrr(rk, ids):
    r=[]
    for i in ids:
        g=GOLD[lab[i]]; p=next((j for j,t in enumerate(rk[i]) if t in g),None); r.append(0 if p is None else 1/(p+1))
    return np.mean(r)
if __name__=='__main__':
    res=[]
    cfgs=[]
    for fam,qroles,droles in (('q3',['cls','none','query'],['none','classification','query']),('gm',['none','classification','query'],['none','document','classification','query'])):
        for qr in qroles:
            qf={'q3':{'cls':f'{R}/out/qwen3_toolhead/qvec_q3cls_final.jsonl'}}.get(fam,{}).get(qr, f'{D}/q_{fam}_{qr}.jsonl')
            try: qv=lv(qf)
            except (FileNotFoundError, json.JSONDecodeError): print('missing',qf); continue
            if any(i not in qv for s_ in sets.values() for i in s_): print('incomplete',qf); continue
            for dr in droles:
                try: DM=docmat(f'{D}/doc_{fam}_{dr}.jsonl')
                except FileNotFoundError: print('missing doc',fam,dr); continue
                for a in [0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
                    row={'fam':fam,'q':qr,'d':dr,'alpha':a}
                    for s,ids in sets.items():
                        rk=rank(qv,DM,ids,a)
                        for k in (1,3,5,10,20): row[f'{s}@{k}']=round(float(rec(rk,ids,k)),4)
                        row[f'{s}_mrr']=round(float(mrr(rk,ids)),4)
                    res.append(row); print(json.dumps(row))
    json.dump(res,open(f'{D}/grid.json','w'))
