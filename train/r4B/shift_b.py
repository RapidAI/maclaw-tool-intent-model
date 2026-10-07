# Aggregate-only diagnostics: why Gemma AUC drops on B_confirm (0.636) vs ETE (0.851). No per-item content printed.
import json,hashlib,re,numpy as np
exec(open('oneshot_b.py').read().split('out={}')[0])
cjk=re.compile(r'[\u3400-\u9fff]')
def stats(tag,P,g,q,qc):
    y=np.array([p['label']=='related' for p in P]); L=np.array([p.get('lang','?') for p in P])
    noc=np.array([not(cjk.search(p['task']) or cjk.search(p['message'])) for p in P])
    tl=np.array([len(p['task']) for p in P]); ml=np.array([len(p['message']) for p in P])
    # lexical overlap (char bigram Jaccard task vs message)
    def bg(s): s=s.lower(); return {s[i:i+2] for i in range(len(s)-1)}
    jac=np.array([len(bg(p['task'])&bg(p['message']))/max(1,len(bg(p['task'])|bg(p['message']))) for p in P])
    r=dict(n=len(P),rel_frac=float(y.mean()),lang={k:int((L==k).sum()) for k in sorted(set(L))},noCJK_frac=float(noc.mean()),
      task_len_med=float(np.median(tl)),msg_len_med=float(np.median(ml)),
      bigramJac_rel=float(jac[y].mean()),bigramJac_unrel=float(jac[~y].mean()),AUC_bigramJac=float(auc(jac,y)),
      gemma=dict(AUC=float(auc(g,y)),cos_rel=float(g[y].mean()),cos_unrel=float(g[~y].mean()),unrel_ge_060=float((g[~y]>=0.6).mean())),
      gemma_AUC_by_lang={k:float(auc(g[L==k],y[L==k])) for k in sorted(set(L)) if 0<y[L==k].sum()<(L==k).sum()},
      gemma_AUC_noCJK=float(auc(g[noc],y[noc])) if 0<y[noc].sum()<noc.sum() else None, gemma_AUC_CJK=float(auc(g[~noc],y[~noc])),
      qwen3_r3cos_AUC=float(auc(qc,y)),qwen3_r4_AUC=float(auc(q,y)),
      gens={k:int(v) for k,v in zip(*np.unique([p.get('gen','?').split('/')[-1] for p in P],return_counts=True))})
    print(tag,json.dumps(r,indent=1)); return r
E=[json.loads(l) for l in open('interrupt.jsonl')]; split=lambda i:'train' if int(hashlib.md5(i.encode()).hexdigest(),16)%2==0 else 'test'
ETE=[p for p in E if split(p['id'])=='test']; VE=load('vec_eval'); G={r['id']:r['rel'] for r in map(json.loads,open('../cap3/sig_interrupt_gemma.jsonl'))}
U=np.stack([VE['IS|'+p['task']] for p in ETE]);M=np.stack([VE['IM|'+p['message']] for p in ETE])
o={'ETE':stats('ETE',ETE,np.array([G[p['id']] for p in ETE]),mlp(U,M),(nz(U)*nz(M)).sum(1))}
C=[json.loads(l) for l in open('../confirm_r4/B_confirm.jsonl')]; VQ=load('vconf_q'); VG=load('vconf_g')
U=np.stack([VQ['IS|'+p['task']] for p in C]);M=np.stack([VQ['IM|'+p['message']] for p in C])
g=(nz(np.stack([VG[p['task']] for p in C]))*nz(np.stack([VG[p['message']] for p in C]))).sum(1)
o['B_confirm']=stats('B_confirm',C,g,mlp(U,M),(nz(U)*nz(M)).sum(1))
json.dump(o,open('shift_b.json','w'),indent=1)
