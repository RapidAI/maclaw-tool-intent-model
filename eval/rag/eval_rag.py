import json, sys, numpy as np, re
D='/workspace/maclaw_reranker/out/replace_audit/rag'
corpus=json.load(open(f'{D}/corpus.json')); pz={x['id']:x['zh'] for x in corpus}
Q=[json.loads(l) for l in open(f'{D}/queries.jsonl')]
def lv(f):
    ids=[];V=[]
    for l in open(f):
        x=json.loads(l); ids.append(x['id']); V.append(x['vec'])
    V=np.array(V,dtype=np.float32); V/=np.linalg.norm(V,axis=1,keepdims=True); return ids,V
def ev(qf,df):
    qids,QV=lv(qf); dids,DV=lv(df); di={d:i for i,d in enumerate(dids)}
    qm={q['qid']:q for q in Q}
    S=QV@DV.T; ranks=[]
    for r,qid in enumerate(qids):
        g=di[qm[qid]['pid']]; s=S[r]; ranks.append(int((s>s[g]).sum())+1)
    ranks=np.array(ranks); meta=[qm[q] for q in qids]
    def m(mask):
        rr=ranks[mask]
        return {'n':int(mask.sum()),'R@1':float((rr<=1).mean()),'R@5':float((rr<=5).mean()),'R@10':float((rr<=10).mean()),
                'MRR@10':float(np.where(rr<=10,1/rr,0).mean()),'nDCG@10':float(np.where(rr<=10,1/np.log2(rr+1),0).mean())}
    out={'all':m(np.ones(len(ranks),bool))}
    for lg in ('zh','en','mix'): out[lg]=m(np.array([x['lang']==lg for x in meta]))
    out['passage_zh']=m(np.array([pz[x['pid']]>0.2 for x in meta])); out['passage_en']=m(np.array([pz[x['pid']]<=0.2 for x in meta]))
    return out, ranks
if __name__=='__main__':
    cfgs=[l.split() for l in sys.argv[1:]]
    res={}
    for name,qf,df in [c.split(':') for c in sys.argv[1:]]:
        o,_=ev(f'{D}/{qf}',f'{D}/{df}'); res[name]=o
        print(f"{name:34s}", ' '.join(f"{k}:{o[k]['R@1']:.3f}/{o[k]['R@10']:.3f}/{o[k]['MRR@10']:.3f}/{o[k]['nDCG@10']:.3f}" for k in ('all','zh','en','mix','passage_zh','passage_en')))
    json.dump(res,open(f'{D}/rag_results.json','w'),indent=1)
