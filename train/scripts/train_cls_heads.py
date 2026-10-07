"""Section 9: export Go heads trained on classification-prompt Gemma vectors, cross-generator protocol.
GK: train = old train (anchor+synth) + 70% Gemini + fix anchors ; calibrate T/tau/tau_s on the other 30% Gemini ; test = all Kimi.
KG: mirror. tau = smallest t with cal selective acc >= 0.95; tau_s starts at max(tau, 0.95) and is raised until cal precision of
accepted sensitive predictions >= 0.97 (cap 0.99). Split identical to scripts/indep_improve.py (per_label_split seed 0)."""
import json, sys, os, numpy as np
os.environ.setdefault('BB', 'gemma_cls')  # BB=gemma -> raw-space ablation heads (no prompt)
TAG = {'gemma_cls': 'gemmacls', 'qwen3go': 'qwen3go'}.get(os.environ['BB'], 'gemmaraw')  # qwen3go: section 10 (pure-Go Qwen3, classification instruct)
sys.argv = ['x']
src = open('scripts/indep_improve.py').read().split('results = []')[0]
exec(src)  # V (cls vectors), rows, old_tr, old_te, NEW, FIX (leak-guarded), train/calibrate helpers, per_label_split
def export(name, tr_new, cal, test):
    ids_test = {r['id'] for r in test}; assert not ids_test & {r['id'] for r in tr_new + cal + FIX}
    items = old_tr + tr_new + FIX; labels = sorted({r['intent'] for r in items}); lid = {l: k for k, l in enumerate(labels)}
    clf = T.make('mlp', 0).fit(np.stack([V[r['id']] for r in items]), np.array([lid[r['intent']] for r in items]))
    assert list(clf.classes_) == list(range(len(labels)))
    def logits(its):
        X = np.stack([V[r['id']] for r in its]); H = np.maximum(0, X @ clf.coefs_[0] + clf.intercepts_[0]); return H @ clf.coefs_[1] + clf.intercepts_[1]
    zc = logits(cal); yc = np.array([lid[r['intent']] for r in cal]); Tm = T.fit_temp(zc, yc)
    P = T.softmax(zc / Tm); conf = P.max(1); pred = [labels[j] for j in P.argmax(1)]; corr = np.array([ok(r['intent'], p) for r, p in zip(cal, pred)])
    tau = 0.99
    for t in np.linspace(0.30, 0.99, 70):
        k = conf >= t
        if k.sum() >= 5 and corr[k].mean() >= 0.95: tau = float(t); break
    isS = np.array([p in SENS for p in pred]); tau_s = max(tau, 0.95)
    for t in np.arange(tau_s, 0.991, 0.01):
        k = (conf >= t) & isS
        tau_s = float(t)
        if k.sum() == 0 or corr[k].mean() >= 0.97: break
    need = np.where(isS, tau_s, tau); acc = conf >= need
    print(f"{name}: n_train={len(items)} (new {len(tr_new)}, fix {len(FIX)}) n_cal={len(cal)} T={Tm:.3f} tau={tau:.2f} tau_s={tau_s:.2f} | cal cov={acc.mean():.3f} sel={corr[acc].mean():.3f} sensFE={int(sum(1 for r,p,a in zip(cal,pred,acc) if a and p in SENS and p!=r['intent']))}")
    # python-side preview on test (Go e2e is the real number)
    zt = logits(test); Pt = T.softmax(zt / Tm); ct = Pt.max(1); pt = [labels[j] for j in Pt.argmax(1)]; cr = np.array([ok(r['intent'], p) for r, p in zip(test, pt)])
    nd = np.array([tau_s if p in SENS else tau for p in pt]); at = ct >= nd
    print(f"   python preview on test n={len(test)}: top1={cr.mean():.3f} cov={at.mean():.3f} sel={cr[at].mean():.3f} sensFE={int(sum(1 for r,p,a in zip(test,pt,at) if a and p in SENS and p!=r['intent']))}")
    h = dict(labels=labels, temperature=float(Tm), tau=tau, tau_s=tau_s, l2norm=True, sensitive=sorted(SENS), embed_role='classification' if TAG in ('gemmacls', 'qwen3go') else '',
             note=f'MLP on {TAG} vectors; {name}; T/tau/tau_s calibrated on held-out new-distribution slice',
             w1=clf.coefs_[0].T.tolist(), b1=clf.intercepts_[0].tolist(), w2=clf.coefs_[1].T.tolist(), b2=clf.intercepts_[1].tolist())
    json.dump(h, open(f'heads/head_mlp_{TAG}_{name}.json', 'w'))
    with open(f'data/go_in_{name}_test.jsonl', 'w') as f:
        for r in test: f.write(json.dumps({'id': r['id'], 'text': r['text']}, ensure_ascii=False) + '\n')
G = [r for r in NEW if r['gen'] == 'gemini']; K = [r for r in NEW if r['gen'] == 'kimi']
trG, calG = per_label_split(G, [0.7, 0.3], 0); export('GK', trG, calG, K)
trK, calK = per_label_split(K, [0.7, 0.3], 0); export('KG', trK, calK, G)
