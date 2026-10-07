# r11S5 old-339 miss analysis (old-339 may now be read repeatedly; Daniel 18:41).
import json, sys, collections
R='/workspace/maclaw_reranker'; F=sys.argv[1]
rows={json.loads(l)['id']:json.loads(l) for f in (f'{R}/data/intent_all.jsonl',f'{R}/data/indep_test.jsonl') for l in open(f)}
meta=json.load(open(f'{R}/data/intent_meta.json')); SAFEU=set(meta['safe_unknown']); SENS=set(meta['sensitive'])
ok=lambda g,p: p==g or (g=='unknown' and p in SAFEU)
X=[json.loads(l) for l in open(F)]
miss=[r for r in X if not ok(rows[r['id']]['intent'],r['cls_primary'])]
cat=collections.Counter(); bylab=collections.Counter(); path=collections.Counter(); D=[]
for r in miss:
    g=rows[r['id']]['intent']; top=r.get('head_top') or []; ha=top[0]['L'] if top else None; hp=top[0]['P'] if top else None
    declined=r['cls_primary'] in ('unknown','ambiguous')
    k=('declined' if declined else 'accepted-wrong')+'/'+('head-argmax-correct' if ha and ok(g,ha) else 'head-argmax-wrong')
    cat[k]+=1; bylab[(g,k)]+=1; path[(k,r['cls_layer'],r['cls_reason'].split(':')[0][:40])]+=1
    D.append(dict(id=r['id'],gold=g,pred=r['cls_primary'],head=ha,hp=round(hp,3) if hp else None,l2=r['l2_primary'],l2c=round(r['l2_conf'],3),layer=r['cls_layer'],reason=r['cls_reason'][:70],sens=g in SENS or (ha in SENS)))
print('n',len(X),'miss',len(miss)); print(dict(cat))
print('-- by path'); [print(' ',k,v) for k,v in path.most_common()]
print('-- declined but head-argmax correct (sorted by head p):')
for d in sorted([d for d in D if d['pred'] in ('unknown','ambiguous') and d['head'] and ok(d['gold'],d['head'])],key=lambda d:-d['hp']): print(' ',d)
print('-- head-argmax wrong:')
for d in D:
    if not(d['head'] and ok(d['gold'],d['head'])): print(' ',d)
