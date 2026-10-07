"""§23 old-distribution guard: 5-fold CV on the OLD TRAIN set (925 anchor+synth; never old-339 test) as an extra selection signal
next to LOFO. For each config the head is trained on (old train minus fold) x w + fix + xf and scored on the held-out old fold.
Also LOFO top1 for the up-weighted variants (same protocol as xf_final OOF: old x w + fix + all other families).
usage: XF_TAG=r3cv python scripts/xf_oldcv.py"""
import os, sys, json, numpy as np, collections, multiprocessing as mp
src = open('scripts/xf_train.py').read(); exec(src.split("RES = dict(")[0])
SNAP = {k: [json.loads(l) for l in open(f'data/train_xfamily/train_xf_{k}.jsonl')] for k in ('r1b', 'r2a', 'r2b', 'r3')}
for k, xs in SNAP.items(): assert all(r['id'] in V for r in xs) and not ({r['id'] for r in xs} & TEST_IDS), k
rng = np.random.RandomState(0); fold = {}
by = collections.defaultdict(list)
for r in old_tr: by[r['intent']].append(r['id'])
for lab in sorted(by):
    ids = by[lab][:]; rng.shuffle(ids)
    for j, i in enumerate(ids): fold[i] = j % 5
CFGS = [('base(no xf)', None, 1.0, 1), ('r1b', 'r1b', 1.0, 1), ('r2a', 'r2a', 1.0, 1), ('r2b', 'r2b', 1.0, 1),
        ('r3 x0.25', 'r3', 0.25, 1), ('r3 x0.5', 'r3', 0.5, 1), ('r3', 'r3', 1.0, 1), ('r3 old_w3', 'r3', 1.0, 3), ('r3 old_w5', 'r3', 1.0, 5),
        ('r2b old_w3', 'r2b', 1.0, 3)]
def xf_of(snap, frac): return [] if snap is None else (SNAP[snap] if frac >= 1 else per_label_split(SNAP[snap], frac, 1)[0])
def job(a):
    kind, ci, k = a; name, snap, frac, w = CFGS[ci]
    if kind == 'cv':
        tr = [r for r in old_tr if fold[r['id']] != k]; te = [r for r in old_tr if fold[r['id']] == k]
        h = Head(tr * w + FIX + xf_of(snap, frac)); P = T.softmax(h.logits(te)); pred = [h.labels[j] for j in P.argmax(1)]
        return kind, ci, k, [(r['id'], r['intent'], p, float(c), r.get('lang')) for r, p, c in zip(te, pred, P.max(1))]
    F = FAMS[k]; X = SNAP[snap]; te = [r for r in X if r['fam'] == F]
    h = Head(old_tr * w + FIX + [r for r in X if r['fam'] != F]); P = T.softmax(h.logits(te)); pred = [h.labels[j] for j in P.argmax(1)]
    return kind, ci, k, [(r['id'], r['intent'], p, float(c), r.get('lang')) for r, p, c in zip(te, pred, P.max(1))]
LOFO_CFGS = [i for i, c in enumerate(CFGS) if c[1] in ('r3', 'r2b') and c[2] >= 1]
jobs = [('cv', ci, k) for ci in range(len(CFGS)) for k in range(5)] + [('lofo', ci, k) for ci in LOFO_CFGS for k in range(len(FAMS))]
out = collections.defaultdict(list)
with mp.Pool(int(os.environ.get('XF_PROCS', '4'))) as pool:
    for kind, ci, k, rows_ in pool.imap_unordered(job, jobs):
        out[(kind, ci)] += rows_; print('done', kind, CFGS[ci][0], k, flush=True)
RES = dict(tag=TAG, folds=5, n_old_train=len(old_tr), configs=[])
for ci, (name, snap, frac, w) in enumerate(CFGS):
    rr = out[('cv', ci)]; corr = np.array([ok(g, p) for _, g, p, _, _ in rr])
    d = dict(name=name, snap=snap, frac=frac, old_w=w, n_xf=len(xf_of(snap, frac)), oldcv_n=len(rr), oldcv_top1=float(corr.mean()))
    for L in ('zh', 'mix', 'en'):
        m = np.array([l == L for *_, l in rr]); d[f'oldcv_{L}'] = float(corr[m].mean()) if m.any() else None
    d['oldcv_conf'] = collections.Counter((g, p) for _, g, p, _, _ in rr if not ok(g, p)).most_common(6)
    if ('lofo', ci) in out:
        lr = out[('lofo', ci)]; lc = np.array([ok(g, p) for _, g, p, _, _ in lr]); d['lofo_n'] = len(lr); d['lofo_top1'] = float(lc.mean())
    RES['configs'].append(d)
    print(f"{name:14s} n_xf={d['n_xf']:5d} oldCV top1={d['oldcv_top1']:.3f} (zh/mix/en {d['oldcv_zh']}/{d['oldcv_mix']}/{d['oldcv_en']})" + (f"  LOFO top1={d['lofo_top1']:.3f} (n={d['lofo_n']})" if 'lofo_top1' in d else ''), flush=True)
json.dump(RES, open(f'out/xf/xf_oldcv_{TAG}.json', 'w'), indent=1)
