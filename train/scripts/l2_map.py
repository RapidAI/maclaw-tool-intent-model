"""Fit raw-equivalent score map s_raw ~= a*s_cls + b on UNLABELED old-synthetic texts only (no new-distribution data),
by matching quantiles of per-label max-cos (top-5 labels per text). Then evaluate the unchanged Go grant rule (0.78/0.10, lookup 0.70/0.05)
in mapped space per source, and the per-source gap/score sweep."""
import json, numpy as np, sys
sys.argv = ['x']; exec(open('scripts/l2_calib.py').read().split("SV = {}")[0])
def allsc(V, items):
    XA = np.stack([V[r['id']] for r in A]); S = np.stack([V[r['id']] for r in items]) @ XA.T
    return np.stack([S[:, [j for j, r in enumerate(A) if r['intent'] == l]].max(1) for l in labs], 1)
R = allsc(VR, SYN); C = allsc(VC, SYN)
k = np.argsort(-R, 1)[:, :5]; r5 = np.take_along_axis(R, k, 1).ravel(); c5 = np.take_along_axis(C, np.argsort(-C, 1)[:, :5], 1).ravel()
qs = np.linspace(0.05, 0.95, 19); qr = np.quantile(r5, qs); qc = np.quantile(c5, qs)
a, b = np.polyfit(qc, qr, 1); print(f"quantile map on SYN top-5 scores: raw ~= {a:.4f}*cls + {b:.4f}")
# gaps
gr = np.sort(R, 1)[:, ::-1]; gc = np.sort(C, 1)[:, ::-1]
print("median gap raw/cls:", np.median(gr[:, 0] - gr[:, 1]).round(4), np.median(gc[:, 0] - gc[:, 1]).round(4))
json.dump(dict(scale=float(a), shift=float(b)), open('out/l2_map_quantile.json', 'w'))
