"""Train classification heads on frozen embeddings; calibrate; evaluate; export for Go."""
import json, sys, time, numpy as np, warnings
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize_scalar
warnings.filterwarnings("ignore")

rows = [json.loads(l) for l in open('data/intent_all.jsonl')]
meta = json.load(open('data/intent_meta.json')); SENS = set(meta['sensitive']); SAFEU = set(meta['safe_unknown'])

def load(backbone):
    if backbone == 'gemma':
        V = {}
        for l in open('out/emb_gemma.jsonl'):
            r = json.loads(l); V[r['id']] = np.array(r['vec'], dtype=np.float32)
        X = np.stack([V[r['id']] for r in rows])
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    else:
        X = np.load(f'out/emb_{backbone}.npy'); ids = json.load(open(f'out/emb_{backbone}.npy.ids.json'))
        assert ids == [r['id'] for r in rows]
    return X

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def fit_temp(logits, y):
    def nll(T):
        p = softmax(logits / T); return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    return minimize_scalar(nll, bounds=(0.05, 20), method='bounded').x

def ece(p, y, bins=10):
    conf = p.max(1); pred = p.argmax(1); e = 0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e

def make(kind, seed=0):
    if kind == 'logreg': return LogisticRegression(C=20, max_iter=500, tol=1e-3)
    if kind == 'mlp': return MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=200, random_state=seed, early_stopping=False)

def logits_of(clf, X):
    if isinstance(clf, LogisticRegression): return clf.decision_function(X)
    return np.log(clf.predict_proba(X) + 1e-12)

def evaluate(name, P, labels, te_idx, tau, extra=None):
    """P: calibrated probs on test rows; tau: acceptance threshold."""
    gold = [rows[i]['intent'] for i in te_idx]
    pred = [labels[j] for j in P.argmax(1)]; conf = P.max(1)
    top3 = [[labels[j] for j in np.argsort(-p)[:3]] for p in P]
    def ok(g, p): return p == g or (g == 'unknown' and p in SAFEU)
    acc = np.mean([ok(g, p) for g, p in zip(gold, pred)])
    acc3 = np.mean([bool(g in t or (g == 'unknown' and set(t) & SAFEU)) for g, t in zip(gold, top3)])
    acc_mask = conf >= tau
    sel = np.mean([ok(g, p) for g, p, a in zip(gold, pred, acc_mask) if a]) if acc_mask.any() else float('nan')
    # fail-closed: a sensitive label accepted while gold is something else
    fexp = sum(1 for g, p, a in zip(gold, pred, acc_mask) if a and p in SENS and p != g)
    fexp_any = sum(1 for g, p in zip(gold, pred) if p in SENS and p != g)
    # missed sensitive: gold sensitive but not accepted correctly (falls back -> safe, costs a turn)
    by = {}
    for key in ('heldout', 'hard'):
        m = [rows[i]['src'] == key for i in te_idx]
        by[key] = np.mean([ok(g, p) for g, p, mm in zip(gold, pred, m) if mm])
    for key in ('zh', 'mix', 'en'):
        m = [rows[i]['lang'] == key for i in te_idx]
        by[key] = np.mean([ok(g, p) for g, p, mm in zip(gold, pred, m) if mm])
    r = dict(method=name, n=len(gold), top1=acc, top3=acc3, coverage=acc_mask.mean(), sel_acc=sel,
             sens_false_exposed_accepted=fexp, sens_false_top1=fexp_any, **{f'acc_{k}': v for k, v in by.items()})
    if extra: r.update(extra)
    return r, pred, conf

def run(backbone, kind, train_src=('anchor', 'synthetic'), export=None, target_sel=0.97):
    X = load(backbone)
    tr = [i for i, r in enumerate(rows) if r['split'] == 'train' and r['src'] in train_src]
    te = [i for i, r in enumerate(rows) if r['split'] == 'test']
    labels = sorted({rows[i]['intent'] for i in tr})
    lid = {l: k for k, l in enumerate(labels)}
    ytr = np.array([lid[rows[i]['intent']] for i in tr])
    # out-of-fold logits for temperature + threshold selection (no test leakage)
    oof = np.zeros((len(tr), len(labels)))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for a, b in skf.split(X[tr], ytr):
        c = make(kind).fit(X[tr][a], ytr[a])
        z = np.full((len(b), len(labels)), -30.0); z[:, c.classes_] = logits_of(c, X[tr][b]); oof[b] = z
    T = fit_temp(oof, ytr)
    Poof = softmax(oof / T)
    # threshold: smallest tau whose OOF selective accuracy >= target
    conf = Poof.max(1); corr = Poof.argmax(1) == ytr; tau = 0.99
    for t in np.linspace(0.05, 0.99, 95):
        m = conf >= t
        if m.sum() > 0 and corr[m].mean() >= target_sel: tau = t; break
    t0 = time.time(); clf = make(kind).fit(X[tr], ytr); fit_s = time.time() - t0
    zte = logits_of(clf, X[te]); Praw = softmax(zte); P = softmax(zte / T)
    gold = np.array([lid.get(rows[i]['intent'], -1) for i in te])
    mask = gold >= 0
    name = f"head-{kind}/{backbone}" + ("" if set(train_src) == {'anchor', 'synthetic'} else "/anchors-only")
    res, pred, confs = evaluate(name, P, labels, te, tau, dict(T=round(T, 3), tau=round(tau, 2), fit_s=round(fit_s, 1),
                         ece_raw=ece(Praw[mask], gold[mask]), ece_cal=ece(P[mask], gold[mask]), dim=X.shape[1]))
    if export:
        h = dict(labels=labels, temperature=T, tau=tau, l2norm=True, sensitive=sorted(SENS))
        if kind == 'logreg':
            h.update(w2=clf.coef_.tolist(), b2=clf.intercept_.tolist())
        else:
            h.update(w1=clf.coefs_[0].T.tolist(), b1=clf.intercepts_[0].tolist(), w2=clf.coefs_[1].T.tolist(), b2=clf.intercepts_[1].tolist())
            h['note'] = 'MLP: probs = softmax(log(softmax(W2 relu(W1 x + b1) + b2)) / T) == softmax((W2 h + b2)/T)'
        json.dump(h, open(export, 'w'))
        np.save(export + '.test_probs.npy', P)
    print("done", name, f"{res['top1']:.3f}", flush=True)
    return res, dict(zip([rows[i]['id'] for i in te], zip(pred, confs.tolist())))

def knn_baseline(backbone, train_src=('anchor',)):
    """maclaw-L2-style: max cosine to per-label exemplar set (anchors [+ synthetic])."""
    X = load(backbone)
    tr = [i for i, r in enumerate(rows) if r['split'] == 'train' and r['src'] in train_src]
    te = [i for i, r in enumerate(rows) if r['split'] == 'test']
    labels = sorted({rows[i]['intent'] for i in tr})
    S = X[te] @ X[tr].T
    sc = np.full((len(te), len(labels)), -1.0)
    for k, l in enumerate(labels):
        cols = [j for j, i in enumerate(tr) if rows[i]['intent'] == l]
        sc[:, k] = S[:, cols].max(1)
    P = softmax(sc / 0.02)  # pseudo-probabilities just for argmax/top3
    name = f"maxcos/{backbone}/" + "+".join(train_src)
    res, _, _ = evaluate(name, P, labels, te, 2.0)
    res['coverage'] = float('nan'); res['sel_acc'] = float('nan'); res['sens_false_exposed_accepted'] = None
    return res

if __name__ == '__main__':
    backbones = sys.argv[1].split(',') if len(sys.argv) > 1 else ['gemma']
    out = []
    for bb in backbones:
        out.append(knn_baseline(bb, ('anchor',)))
        out.append(knn_baseline(bb, ('anchor', 'synthetic')))
        for kind in ('logreg', 'mlp'):
            r, _ = run(bb, kind, export=f'heads/head_{kind}_{bb}.json'); out.append(r)
            r, _ = run(bb, kind, train_src=('anchor',)); out.append(r)
    json.dump(out, open(f'out/heads_{"_".join(backbones)}.json', 'w'), indent=1, default=float)
    cols = ['method', 'top1', 'top3', 'acc_heldout', 'acc_hard', 'acc_zh', 'acc_mix', 'acc_en', 'coverage', 'sel_acc', 'sens_false_top1', 'sens_false_exposed_accepted', 'T', 'tau', 'ece_raw', 'ece_cal']
    print("\t".join(cols))
    for r in out:
        print("\t".join((f"{r.get(c):.3f}" if isinstance(r.get(c), float) else str(r.get(c, ''))) for c in cols))
