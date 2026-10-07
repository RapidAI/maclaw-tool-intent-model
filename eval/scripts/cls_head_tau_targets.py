"""tau on the held-out calibration slice for several precision targets, using the exported Go heads (no retraining)."""
import json, os, sys, numpy as np
os.environ['BB'] = 'gemma_cls'; sys.argv = ['x']
exec(open('scripts/indep_improve.py').read().split('results = []')[0])
G = [r for r in NEW if r['gen'] == 'gemini']; K = [r for r in NEW if r['gen'] == 'kimi']
for name, A in (('GK', G), ('KG', K)):
    tr, cal = per_label_split(A, [0.7, 0.3], 0); h = json.load(open(f'heads/head_mlp_gemmacls_{name}.json'))
    X = np.stack([V[r['id']] for r in cal]); H = np.maximum(0, X @ np.array(h['w1']).T + np.array(h['b1'])); z = H @ np.array(h['w2']).T + np.array(h['b2'])
    P = T.softmax(z / h['temperature']); conf = P.max(1); pred = [h['labels'][j] for j in P.argmax(1)]; corr = np.array([ok(r['intent'], p) for r, p in zip(cal, pred)])
    for tgt in (0.97, 0.95, 0.90, 0.85):
        tau = 0.99
        for t in np.linspace(0.30, 0.99, 70):
            k = conf >= t
            if k.sum() >= 5 and corr[k].mean() >= tgt: tau = float(t); break
        k = conf >= tau; print(f"{name} target {tgt:.2f}: tau={tau:.2f} cal cov={k.mean():.3f} sel={corr[k].mean():.3f} n_cal={len(cal)}")
