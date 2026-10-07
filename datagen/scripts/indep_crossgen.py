"""Step 5: quick improvement with a proper cross-generator split.
Train = old train (anchor+synthetic) + independent items of generator A; test = independent items of generator B (never seen), and vice versa.
Temperature/tau refit on out-of-fold logits of the training set only."""
import json, sys, numpy as np, collections
sys.path.insert(0, 'scripts')
import train_heads as T
from sklearn.model_selection import StratifiedKFold
meta = json.load(open('data/intent_meta.json')); SENS = set(meta['sensitive']); SAFEU = set(meta['safe_unknown'])
rows = T.rows
V = {}
for f in ('out/emb_gemma.jsonl', 'out/emb_indep_gemma.jsonl'):
    for l in open(f):
        r = json.loads(l); v = np.array(r['vec'], np.float32); V[r['id']] = v / np.linalg.norm(v)
NEW = [json.loads(l) for l in open('data/indep_test.jsonl')]
old_tr = [r for r in rows if r['split'] == 'train']; old_te = [r for r in rows if r['split'] == 'test']
ok = lambda g, p: p == g or (g == 'unknown' and p in SAFEU)
def fit(train, kind='mlp'):
    labels = sorted({r['intent'] for r in train}); lid = {l: k for k, l in enumerate(labels)}
    X = np.stack([V[r['id']] for r in train]); y = np.array([lid[r['intent']] for r in train])
    oof = np.zeros((len(train), len(labels)))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        c = T.make(kind).fit(X[a], y[a]); z = np.full((len(b), len(labels)), -30.0); z[:, c.classes_] = T.logits_of(c, X[b]); oof[b] = z
    Tm = T.fit_temp(oof, y); P = T.softmax(oof / Tm); conf = P.max(1); corr = P.argmax(1) == y; tau = 0.99
    for t in np.linspace(0.05, 0.99, 95):
        m = conf >= t
        if m.sum() and corr[m].mean() >= 0.97: tau = t; break
    return T.make(kind).fit(X, y), labels, Tm, tau
def ev(model, items, tau_s=0.90):
    clf, labels, Tm, tau = model
    X = np.stack([V[r['id']] for r in items]); P = T.softmax(T.logits_of(clf, X) / Tm)
    pred = [labels[j] for j in P.argmax(1)]; conf = P.max(1); g = [r['intent'] for r in items]
    corr = np.array([ok(a, b) for a, b in zip(g, pred)])
    need = np.array([max(tau, tau_s) if p in SENS else tau for p in pred]); acc = conf >= need
    d = dict(n=len(items), top1=corr.mean(), coverage=acc.mean(), sel_acc=corr[acc].mean() if acc.any() else float('nan'),
             sens_fexp_acc=int(sum(1 for a, p, m in zip(g, pred, acc) if m and p in SENS and p != a)), ece=T.ece(P, np.array([labels.index(a) if a in labels else -1 for a in g])), T=Tm, tau=tau)
    for L in ('zh', 'mix', 'en'):
        m = np.array([r.get('lang') == L for r in items]); d[f'acc_{L}'] = corr[m].mean()
    return d
out = []
for A, B in (('gemini', 'kimi'), ('kimi', 'gemini')):
    tA = [r for r in NEW if r['gen'] == A]; tB = [r for r in NEW if r['gen'] == B]
    base = fit(old_tr); aug = fit(old_tr + tA)
    for name, m in ((f'baseline (old train) -> test {B}', base), (f'old train + {A} ({len(tA)}) -> test {B}', aug)):
        d = ev(m, tB); d.update(name=name, set=f'indep_{B}'); out.append(d)
        d2 = ev(m, old_te); d2.update(name=name, set='old_test'); out.append(d2)
json.dump(out, open('out/indep_crossgen.json', 'w'), indent=1, default=float)
for d in out:
    print(f"{d['name']:42s} on {d['set']:12s} n={d['n']} top1={d['top1']:.3f} zh={d['acc_zh']:.3f} mix={d['acc_mix']:.3f} en={d['acc_en']:.3f} cov={d['coverage']:.3f} sel={d['sel_acc']:.3f} sensFE={d['sens_fexp_acc']} ECE={d['ece']:.3f} T={d['T']:.2f} tau={d['tau']:.2f}")
