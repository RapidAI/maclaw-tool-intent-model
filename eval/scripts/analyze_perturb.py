import json, numpy as np
from collections import defaultdict
meta = json.load(open('data/intent_meta.json')); SENS = set(meta['sensitive']); SAFEU = set(meta['safe_unknown'])
src = {json.loads(l)['id']: json.loads(l) for l in open('data/perturb.jsonl')}
base = {json.loads(l)['id']: json.loads(l) for l in open('out/go_eval_none.jsonl')}
tau = json.load(open('heads/head_mlp_gemma.json'))['tau']
def ok(g, p): return p == g or (g == 'unknown' and p in SAFEU)
by = defaultdict(list); odd = []
for l in open('out/go_eval_perturb.jsonl'):
    r = json.loads(l); s = src[r['id']]; top = r['head_top'][0]
    if s['kind'] == 'odd':
        odd.append((s['text'][:40].replace('\n', ' '), len(s['text']), top['L'], round(top['P'], 2), top['P'] >= tau, r['l2_primary'], r['l2_confident'], r['cls_primary'], round(r['head_embed_ms'], 1)))
        continue
    bid = r['id'].rsplit('-', 1)[0]; b = base[bid]
    by[s['kind']].append(dict(head_ok=ok(s['intent'], top['L']), same=top['L'] == b['head_top'][0]['L'],
                              l2_ok=ok(s['intent'], r['l2_primary']), acc=top['P'] >= tau,
                              sens_fp=top['P'] >= tau and top['L'] in SENS and top['L'] != s['intent']))
print("kind n head_acc head_label_unchanged_vs_clean L2_acc accepted sens_false_exposed")
for k, v in by.items():
    print(k, len(v), *(f"{np.mean([x[f] for x in v]):.3f}" for f in ('head_ok', 'same', 'l2_ok', 'acc')), sum(x['sens_fp'] for x in v))
print("\nodd inputs: text[:40], len, head_label, p, accepted, l2_primary, l2_confident, cls_primary(no L3), embed_ms")
for o in odd: print(" ", o)
