# A_confirm scoring: gold = A_confirm 'intent'; same metrics + paired bootstrap (2000, seed 7) as hintfix/cmp_gemma.py.
import json, random
R='/workspace/maclaw_reranker'; O=R+'/out/replace_audit/aconf'
rows={json.loads(l)['id']:json.loads(l) for l in open(R+'/out/replace_audit/confirm_r4/A_confirm.jsonl')}
meta=json.load(open(R+'/data/intent_meta.json')); SAFEU=set(meta['safe_unknown']); SENS=set(meta['sensitive'])
ok=lambda g,p: p==g or (g=='unknown' and p in SAFEU)
ld=lambda f:{json.loads(l)['id']:json.loads(l)['cls_primary'] for l in open(f)}
q,g=ld(O+'/qwen3.jsonl'),ld(O+'/gemma.jsonl'); ids=sorted(set(q)&set(g)&set(rows))
def st(d):
    a=sum(ok(rows[i]['intent'],d[i]) for i in ids); acc=[i for i in ids if d[i] not in ('unknown','ambiguous')]
    return dict(primary=a/len(ids),accepted=len(acc),acc_accepted=sum(ok(rows[i]['intent'],d[i]) for i in acc)/max(1,len(acc)),sensFE=sum(d[i] in SENS and d[i]!=rows[i]['intent'] for i in acc))
sq,sg=st(q),st(g); x=[ok(rows[i]['intent'],q[i])-ok(rows[i]['intent'],g[i]) for i in ids]; r=random.Random(7); v=[]
for _ in range(2000): s=[x[r.randrange(len(x))] for _ in x]; v.append(sum(s)/len(s))
v.sort(); out=dict(n=len(ids),qwen3=sq,gemma=sg,delta=sq['primary']-sg['primary'],ci=[v[50],v[1949]])
json.dump(out,open(O+'/aconf_result.json','w'),indent=1); print(out)
