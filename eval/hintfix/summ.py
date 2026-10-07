import json,sys
rows={json.loads(l)['id']:json.loads(l) for f in ('data/intent_all.jsonl','data/indep_test.jsonl') for l in open(f)}
meta=json.load(open('data/intent_meta.json')); SAFEU=set(meta['safe_unknown']); SENS=set(meta['sensitive'])
ok=lambda g,p: p==g or (g=='unknown' and p in SAFEU)
def summ(f):
    R=[json.loads(l) for l in open(f)]; n=len(R)
    a=sum(ok(rows[r['id']]['intent'],r['cls_primary']) for r in R)
    acc=[r for r in R if r['cls_primary'] not in ('unknown','ambiguous')]
    aok=sum(ok(rows[r['id']]['intent'],r['cls_primary']) for r in acc)
    fe=sum(1 for r in acc if r['cls_primary'] in SENS and r['cls_primary']!=rows[r['id']]['intent'])
    veto=sum('local head disagrees' in (r.get('cls_reason') or '') for r in R); ov=sum('overrides L2 composite' in (r.get('cls_reason') or '') for r in R)
    return dict(n=n,primary=round(a/n,3),accepted=len(acc),acc_accepted=round(aok/max(1,len(acc)),3),sens_fe=fe,vetoes=veto,overrides=ov)
for f in sys.argv[1:]: print(f.split('/')[-1], summ(f))
