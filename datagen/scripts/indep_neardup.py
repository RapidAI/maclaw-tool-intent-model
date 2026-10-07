"""Step 2: near-duplicate removal vs train/anchor (Gemma cos > 0.92 or char-3gram Jaccard > 0.6)."""
import json, numpy as np, collections, re, unicodedata
rows = [json.loads(l) for l in open('data/intent_all.jsonl')]
V = {}
for f in ('out/emb_gemma.jsonl', 'out/emb_indep_gemma.jsonl'):
    for l in open(f):
        r = json.loads(l); V[r['id']] = np.array(r['vec'], np.float32)
C = [json.loads(l) for l in open('data/indep_candidates.jsonl')]
def nz(ids): X = np.stack([V[i] for i in ids]); return X / np.linalg.norm(X, axis=1, keepdims=True)
tr = [r for r in rows if r['split'] == 'train']; te = [r for r in rows if r['split'] == 'test']
Xc, Xtr, Xte = nz([c['id'] for c in C]), nz([r['id'] for r in tr]), nz([r['id'] for r in te])
Str, Ste = Xc @ Xtr.T, Xc @ Xte.T
def grams(s):
    s = re.sub(r'\s+', '', unicodedata.normalize('NFKC', s).lower()); return {s[i:i + 3] for i in range(max(1, len(s) - 2))}
Gtr = [grams(r['text']) for r in tr]
st = collections.Counter(); keep = []
for n, c in enumerate(C):
    g = grams(c['text']); j = max(len(g & h) / len(g | h) for h in Gtr)
    c['max_cos_train'] = float(Str[n].max()); c['nn_train'] = tr[int(Str[n].argmax())]['text']; c['max_jacc_train'] = round(j, 3)
    c['max_cos_oldtest'] = float(Ste[n].max())
    if c['max_cos_train'] > 0.92: st['rm_cos>0.92'] += 1; c['rm'] = 'cos'
    elif j > 0.6: st['rm_jacc>0.6'] += 1; c['rm'] = 'jacc'
    else: keep.append(c)
    st['oldtest_cos>0.92'] += c['max_cos_oldtest'] > 0.92
st['kept'] = len(keep); st['in'] = len(C)
json.dump([c for c in C if c.get('rm')], open('out/indep_neardup_removed.json', 'w'), ensure_ascii=False, indent=1)
with open('data/indep_filtered.jsonl', 'w') as f:
    for c in keep: f.write(json.dumps(c, ensure_ascii=False) + '\n')
print(dict(st)); print('cos to train: p50 %.3f p90 %.3f max %.3f' % tuple(np.percentile([c['max_cos_train'] for c in C], [50, 90, 100])))
json.dump(dict(st), open('out/indep_qc_step2.json', 'w'), indent=1)
