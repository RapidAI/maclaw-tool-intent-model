"""Cheap fix #2: same production Go EmbeddingGemma, but with the model-card classification prompt
('task: classification | query: ', maclaw RoleClassification) instead of the raw text L2 uses today.
Train on OLD train only (anchor+synthetic), T/tau via 5-fold OOF (same protocol as section 2), evaluate on old test + indep (never trained/calibrated on)."""
import json, sys, numpy as np
sys.path.insert(0, 'scripts')
import train_heads as T
from sklearn.model_selection import StratifiedKFold
meta = json.load(open('data/intent_meta.json')); SENS = set(meta['sensitive']); SAFEU = set(meta['safe_unknown'])
ok = lambda g, p: p == g or (g == 'unknown' and p in SAFEU)
def load(f):
    V = {}
    for l in open(f):
        r = json.loads(l); v = np.array(r['vec'], np.float32); V[r['id']] = v / np.linalg.norm(v)
    return V
VR = load('out/emb_gemma.jsonl'); VR.update(load('out/emb_indep_gemma.jsonl')); VC = load('out/emb_gemma_clsprompt.jsonl')
rows = T.rows; tr = [r for r in rows if r['split'] == 'train']; te = [r for r in rows if r['split'] == 'test']
NEW = [json.loads(l) for l in open('data/indep_test.jsonl')]
def ece(conf, corr, bins=10):
    e = 0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return e
def maxcos(V, items):
    A = [r for r in tr if r['src'] == 'anchor']; labs = sorted({r['intent'] for r in A}); XA = np.stack([V[r['id']] for r in A])
    S = np.stack([V[r['id']] for r in items]) @ XA.T
    sc = np.stack([S[:, [j for j, r in enumerate(A) if r['intent'] == l]].max(1) for l in labs], 1)
    return np.mean([ok(r['intent'], labs[j]) for r, j in zip(items, sc.argmax(1))])
for name, V in (('raw text (current L2 usage)', VR), ('classification prompt', VC)):
    labels = sorted({r['intent'] for r in tr}); lid = {l: k for k, l in enumerate(labels)}
    X = np.stack([V[r['id']] for r in tr]); y = np.array([lid[r['intent']] for r in tr]); oof = np.zeros((len(tr), len(labels)))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        c = T.make('mlp').fit(X[a], y[a]); z = np.full((len(b), len(labels)), -30.0); z[:, c.classes_] = T.logits_of(c, X[b]); oof[b] = z
    Tm = T.fit_temp(oof, y); P = T.softmax(oof / Tm); conf = P.max(1); corr = P.argmax(1) == y; tau = 0.99
    for t in np.linspace(0.05, 0.99, 95):
        m = conf >= t
        if m.sum() and corr[m].mean() >= 0.97: tau = t; break
    clf = T.make('mlp').fit(X, y)
    for sn, items in (('old_test', te), ('indep', NEW)):
        P = T.softmax(T.logits_of(clf, np.stack([V[r['id']] for r in items])) / Tm); pred = [labels[j] for j in P.argmax(1)]; cf = P.max(1)
        cr = np.array([ok(r['intent'], p) for r, p in zip(items, pred)]); need = np.array([max(tau, 0.90) if p in SENS else tau for p in pred]); acc = cf >= need
        fe = sum(1 for r, p, k in zip(items, pred, acc) if k and p in SENS and p != r['intent'])
        lang = ' '.join(f"{L}={cr[np.array([r.get('lang') == L for r in items])].mean():.3f}" for L in ('zh', 'mix', 'en'))
        print(f"{name:28s} {sn:8s} n={len(items)} L2maxcos={maxcos(V, items):.3f} MLP top1={cr.mean():.3f} {lang} ECE={ece(cf, cr):.3f} T={Tm:.2f} tau={tau:.2f} cov={acc.mean():.3f} sel={cr[acc].mean():.3f} sensFE={fe}", flush=True)
