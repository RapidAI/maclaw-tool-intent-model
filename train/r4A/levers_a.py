# r4 A levers on the GPU server (CPU, 20 cores): for recipes {base=r3W5F, hard=r4H} x seeds, compute LOFO OOF logits, old-train 5-fold CV logits
# and final-head weights (all fits in parallel, 1 BLAS thread each). Then evaluate single heads and logit-average ensembles with the LOCKED r4 rule
# (T on pooled OOF; tau/tau_s: OOF sel>=0.976 AND sens_prec>=0.997 -> max cov). Gates: OOF + old-train CV (+ indep839 report). No old-339 / confirm.
import os, sys, json, numpy as np, itertools, multiprocessing as mp, pickle
os.environ.setdefault('OMP_NUM_THREADS','1')
TAG='lev'; OLDW=5
src=open('scripts/xf_train.py').read(); exec(src.split("RES = dict(")[0])
HARD=[json.loads(l) for l in open('out/replace_audit/r4A/hard_train.jsonl')]
for l in open('out/replace_audit/r4A/emb_hard_qwen3go.jsonl'):
    r=json.loads(l); v=np.array(r['vec'],np.float32); V[r['id']]=v/np.linalg.norm(v)
HARD=[r for r in HARD if r['id'] in V]
for r in HARD: r.setdefault('fam','r4hard'); r.setdefault('split','train')
rng=np.random.RandomState(0); idx=rng.permutation(len(old_tr)); OF=np.array_split(idx,5)
RECIPES={'base':[], 'hard':HARD}
SEEDS=[int(s) for s in os.environ.get('LEV_SEEDS','0,1,2').split(',')]
def job(spec):
    rec,seed,kind,k=spec; extra=RECIPES[rec]
    if kind=='lofo':
        F=FAMS[k]; te=[r for r in XF if r['fam']==F]; tr=old_tr*OLDW+FIX+[r for r in XF if r['fam']!=F]+extra
    elif kind=='cv':
        te=[old_tr[i] for i in OF[k]]; tr=[old_tr[i] for i in np.setdiff1d(idx,OF[k])]*OLDW+FIX+XF+extra
    else:
        te=IND; tr=old_tr*OLDW+FIX+XF+extra
    h=Head(tr,seed=seed)
    return spec, h.labels, h.logits(te), ((h.w1,h.b1,h.w2,h.b2) if kind=='final' else None)
if __name__=='__main__':
    specs=[(rec,s,'lofo',k) for rec in RECIPES for s in SEEDS for k in range(len(FAMS))]+[(rec,s,'cv',k) for rec in RECIPES for s in SEEDS for k in range(5)]+[(rec,s,'final',0) for rec in RECIPES for s in SEEDS]
    print('jobs',len(specs),flush=True)
    res={}
    with mp.get_context('fork').Pool(int(os.environ.get('LEV_PROCS','18'))) as pool:
        for spec,lab,Z,W in pool.imap_unordered(job,specs):
            res[spec]=(lab,Z,W); print('done',spec,len(res),'/',len(specs),flush=True)
    pickle.dump(dict(res=res,FAMS=FAMS),open('out/replace_audit/r4A/levers_raw.pkl','wb'))
    print('saved',flush=True)
