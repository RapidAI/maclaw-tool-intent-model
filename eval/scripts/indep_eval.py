"""Evaluate existing heads (NO retraining on new data) on the independent set, side by side with the old test set."""
import json, sys, os, numpy as np, collections
sys.path.insert(0, 'scripts')
from tool_gold import GOLD
meta = json.load(open('data/intent_meta.json')); SENS = set(meta['sensitive']); SAFEU = set(meta['safe_unknown'])
TAU_S = 0.90
rows = [json.loads(l) for l in open('data/intent_all.jsonl')]
old = [r for r in rows if r['split'] == 'test']
def loadset(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
NEW = loadset('data/indep_test.jsonl'); AMB = loadset('data/indep_ambiguous.jsonl')
def gemma(files):
    V = {}
    for f in files:
        for l in open(f):
            r = json.loads(l); v = np.array(r['vec'], np.float32); V[r['id']] = v / np.linalg.norm(v)
    return V
VG = gemma(['out/emb_gemma.jsonl', 'out/emb_indep_gemma.jsonl'])
VQ = {}
for f in ('out/emb_qwen3e.npy', 'out/emb_indep_qwen3e.npy'):
    if os.path.exists(f):
        X = np.load(f); ids = json.load(open(f + '.ids.json')); VQ.update(dict(zip(ids, X)))
def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def head_probs(h, X):
    if 'w1' in h:
        H = np.maximum(0, X @ np.array(h['w1']).T + np.array(h['b1'])); z = H @ np.array(h['w2']).T + np.array(h['b2'])
    else:
        z = X @ np.array(h['w2']).T + np.array(h['b2'])
    return softmax(z / h['temperature'])
def ok(g, p): return p == g or (g == 'unknown' and p in SAFEU)
def ece(conf, corr, bins=10):
    e = 0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return e
def metrics(items, preds, top3, conf, tau):
    g = [r['intent'] for r in items]
    corr = np.array([ok(a, b) for a, b in zip(g, preds)])
    need = np.array([max(tau, TAU_S) if p in SENS else tau for p in preds])
    acc_m = conf >= need
    d = dict(n=len(items), top1=corr.mean(), top3=np.mean([bool(a in t or (a == 'unknown' and set(t) & SAFEU)) for a, t in zip(g, top3)]),
             coverage=acc_m.mean(), sel_acc=corr[acc_m].mean() if acc_m.any() else float('nan'),
             sens_fexp_acc=int(sum(1 for a, p, m in zip(g, preds, acc_m) if m and p in SENS and p != a)),
             sens_fexp_top1=int(sum(1 for a, p in zip(g, preds) if p in SENS and p != a)),
             ece=ece(conf, corr), mean_conf=float(conf.mean()))
    for L in ('zh', 'mix', 'en'):
        m = np.array([r.get('lang') == L for r in items]); d[f'acc_{L}'] = corr[m].mean() if m.any() else float('nan')
    for G in ('gemini', 'kimi'):
        m = np.array([r.get('gen') == G for r in items]); d[f'acc_{G}'] = corr[m].mean() if m.any() else float('nan')
    for K in ('label', 'hard'):
        m = np.array([r.get('kind') == K for r in items]); d[f'acc_{K}'] = corr[m].mean() if m.any() else float('nan')
    return d, corr
HEADS = [('MLP anchor+synth (Gemma)', 'heads/head_mlp_gemma.json', VG), ('MLP anchor-only (Gemma)', 'heads/head_mlp_gemma_anchoronly.json', VG),
         ('logreg anchor+synth (Gemma)', 'heads/head_logreg_gemma.json', VG), ('logreg anchor-only (Gemma)', 'heads/head_logreg_gemma_anchoronly.json', VG),
         ('MLP anchor+synth (Qwen3-Emb-0.6B)', 'heads/head_mlp_qwen3e.json', VQ), ('logreg anchor+synth (Qwen3-Emb-0.6B)', 'heads/head_logreg_qwen3e.json', VQ)]
def maxcos(items, src):
    tr = [r for r in rows if r['split'] == 'train' and r['src'] in src]; labs = sorted({r['intent'] for r in tr})
    X = np.stack([VG[r['id']] for r in items]); A = np.stack([VG[r['id']] for r in tr]); S = X @ A.T
    sc = np.stack([S[:, [j for j, r in enumerate(tr) if r['intent'] == l]].max(1) for l in labs], 1)
    o = np.argsort(-sc, 1); return [labs[i[0]] for i in o], [[labs[k] for k in i[:3]] for i in o], sc.max(1)
res = []; perrow = collections.defaultdict(dict); confs = {}
for setname, items in (('old_test', old), ('indep', NEW), ('indep_ambiguous', AMB)):
    if not items: continue
    for nm, src in (('L2 max-cos anchors (python replica)', ('anchor',)),):
        p, t3, c = maxcos(items, src); d, _ = metrics(items, p, t3, np.zeros(len(items)) - 1, 2.0)
        d.update(set=setname, method=nm, coverage=float('nan'), sel_acc=float('nan'), ece=float('nan')); res.append(d)
    for nm, path, V in HEADS:
        if not os.path.exists(path) or any(r['id'] not in V for r in items): continue
        h = json.load(open(path)); X = np.stack([V[r['id']] for r in items]); P = head_probs(h, X); L = h['labels']
        o = np.argsort(-P, 1); preds = [L[i[0]] for i in o]; t3 = [[L[k] for k in i[:3]] for i in o]; conf = P.max(1)
        d, corr = metrics(items, preds, t3, conf, h['tau']); d.update(set=setname, method=nm, tau=h['tau']); res.append(d)
        for r, p, c in zip(items, preds, conf): perrow[nm][r['id']] = (p, float(c))
        if nm.startswith('MLP anchor+synth (Gemma)'): confs[setname] = (conf, corr)
json.dump(res, open('out/indep_eval.json', 'w'), indent=1, default=float)
json.dump(perrow, open('out/indep_eval_preds.json', 'w'))
cols = ['set', 'method', 'n', 'top1', 'top3', 'acc_zh', 'acc_mix', 'acc_en', 'acc_gemini', 'acc_kimi', 'acc_label', 'acc_hard', 'coverage', 'sel_acc', 'sens_fexp_acc', 'sens_fexp_top1', 'ece', 'mean_conf']
print('\t'.join(cols))
for d in res: print('\t'.join(f"{d.get(c):.3f}" if isinstance(d.get(c), float) else str(d.get(c)) for c in cols))
# reliability bins for calibration (MLP Gemma)
for s, (conf, corr) in confs.items():
    print(f'reliability {s}:', ' '.join(f"[{lo:.1f},{lo+.1:.1f}] n={int(m.sum())} acc={corr[m].mean():.2f} conf={conf[m].mean():.2f}" for lo in np.arange(0.3, 1.0, 0.1) for m in [(conf > lo) & (conf <= lo + .1)] if m.sum() >= 5))
# confusions (MLP Gemma) on indep
if NEW:
    cc = collections.Counter((r['intent'], perrow['MLP anchor+synth (Gemma)'][r['id']][0]) for r in NEW if not ok(r['intent'], perrow['MLP anchor+synth (Gemma)'][r['id']][0]))
    print('top confusions gold->pred:', cc.most_common(25))
    pl = collections.Counter(r['intent'] for r in NEW); wl = collections.Counter(r['intent'] for r in NEW if not ok(r['intent'], perrow['MLP anchor+synth (Gemma)'][r['id']][0]))
    print('worst labels:', sorted(((wl[l] / pl[l], l, pl[l]) for l in pl), reverse=True)[:15])
# tool head
th = json.load(open('heads/tool_head_ovr_gemma.json')); W = np.array(th['W']); b = np.array(th['b']); T = th['tools']
for setname, items in (('old_test', old), ('indep', NEW)):
    it = [r for r in items if r['intent'] in GOLD]
    if not it: continue
    X = np.stack([VG[r['id']] for r in it]); Pt = X @ W.T + b; o = np.argsort(-Pt, 1)
    for L in (None, 'zh', 'mix', 'en'):
        sel = [k for k, r in enumerate(it) if L is None or r.get('lang') == L]
        r1 = np.mean([T[o[k][0]] in GOLD[it[k]['intent']] for k in sel]); r5 = np.mean([any(T[j] in GOLD[it[k]['intent']] for j in o[k][:5]) for k in sel])
        print(f'tool head {setname} lang={L or "all"} n={len(sel)} R@1={r1:.3f} R@5={r5:.3f}')
