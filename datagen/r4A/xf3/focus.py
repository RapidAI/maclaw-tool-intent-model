# TRAIN-only error analysis (OOF LOFO for xf rows + old-train CV) -> focus labels / confusable pairs for Qwen generation. Never touches old-339/indep.
import json, collections
R='/workspace/maclaw_reranker'; meta=json.load(open(f'{R}/data/intent_meta.json')); SENS=set(meta['sensitive']); SAFEU=set(meta['safe_unknown'])
ok=lambda g,p:p==g or (g=='unknown' and p in SAFEU)
out={}
for h in ('base0','r4LS9'):
    H=json.load(open(f'probs_{h}.json')); tau,ts=H['tau'],H['tau_s']
    err=collections.Counter(); tot=collections.Counter(); pairs=collections.Counter(); fe=collections.Counter(); dec=collections.Counter()
    for t,(g,p,P) in H['train'].items():
        tot[g]+=1; a=P>=(max(tau,ts) if p in SENS else tau)
        if not ok(g,p): pairs[(g,p)]+=1
        if a and not ok(g,p):
            err[g]+=1
            if p in SENS: fe[(g,p)]+=1
        if not a: dec[g]+=1
    out[h]=dict(err=err,dec=dec,tot=tot,pairs=pairs,fe=fe)
lab=set(out['base0']['tot'])
score={l:sum(out[h]['err'][l]*3+out[h]['dec'][l] for h in out)/max(1,sum(out[h]['tot'][l] for h in out)) for l in lab}
focus=sorted(lab,key=lambda l:-score[l])[:20]
fe=collections.Counter(); pr=collections.Counter()
for h in out: fe.update(out[h]['fe']); pr.update(out[h]['pairs'])
fepairs=[k for k,_ in fe.most_common(15)]; topairs=[k for k,_ in pr.most_common(40) if k[1]!='unknown' and k[0]!=k[1]][:25]
pairs=list(dict.fromkeys(fepairs+topairs))
print('focus labels',[(l,round(score[l],3)) for l in focus]); print('FE pairs',fe.most_common(15)); print('pairs',len(pairs),pairs[:12])
json.dump(dict(focus=focus,pairs=[list(p) for p in pairs],score=score),open('xf3/focus.json','w'),indent=1,ensure_ascii=False)
