# Offline threshold simulation from Go records (head_top p + decision): re-decide only items whose final decision came from the local head being below threshold or accepted by it.
import json, sys
R='/workspace/maclaw_reranker'
rows={json.loads(l)['id']:json.loads(l) for f in (f'{R}/data/intent_all.jsonl',f'{R}/data/indep_test.jsonl') for l in open(f)}
meta=json.load(open(f'{R}/data/intent_meta.json')); SAFEU=set(meta['safe_unknown']); SENS=set(meta['sensitive'])
ok=lambda g,p: p==g or (g=='unknown' and p in SAFEU)
K={json.loads(l)['id'] for l in open(f'{R}/data/go_in_GK_test_150.jsonl')}
def sim(F,tau,ts,ids=None):
    X=[json.loads(l) for l in open(F)]; X=[r for r in X if ids is None or r['id'] in ids]; c=fe=acc=0
    for r in X:
        g=rows[r['id']]['intent']; p=r['cls_primary']; top=r.get('head_top') or []
        headpath=r['cls_reason'].startswith('embedding ambiguous; tree classification') or r['cls_reason'].startswith('tree-after-embedding:')
        if headpath and top:
            L,P=top[0]['L'],top[0]['P']; th=max(tau,ts) if L in SENS else tau
            p=L if P>=th else 'unknown'
        c+=ok(g,p); a=p not in ('unknown','ambiguous'); acc+=a; fe+=a and p in SENS and p!=g
    return c,len(X),acc,fe
A='out/replace_audit/hintfix'
for tau in (0.80,0.75,0.70):
    for ts in (0.99,0.98,0.97):
        o=sim(f'{A}/go_r11S5_fix1_old339.jsonl',tau,ts); i=sim(f'{A}/go_r11S5_fix1_indep839.jsonl',tau,ts); k=sim(f'{A}/go_r11S5_fix1_indep839.jsonl',tau,ts,K)
        print(f'τ {tau:.2f} τs {ts:.2f} | old339 {o[0]}/{o[1]} FE {o[3]} | indep {i[0]}/{i[1]} FE {i[3]} | K150 {k[0]}/{k[1]} FE {k[3]}')
