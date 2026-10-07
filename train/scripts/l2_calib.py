"""Section 9: calibrate Layer-2 confident thresholds for the classification-prompt Gemma space.
Simplified replica of classifyByEmbedding's grant rule: confident if (top1>=t & gap>=g) or (top1 in lookup & top1>=tl & gap>=gl);
document_generate never confident. Go maps prompt-space cosine s -> a*s+b so that existing raw-space constants
(0.78/0.10 confident, 0.70/0.05 lookup, and every other floor) keep their meaning: a=0.10/g_p, b=0.78-a*t_p
=> lookup rule in prompt space is (t_p-0.8*g_p, g_p/2).
Calibration sets never include the evaluated test items: G-side = old synthetic (non-anchor train) + all Gemini indep; K-side mirror; ALL = old synth + all indep (production default)."""
import json, numpy as np, collections, sys
meta = json.load(open('data/intent_meta.json')); SAFEU = set(meta['safe_unknown'])
LOOK = {'search', 'live_data', 'web_fetch'}
def load(f):
    V = {}
    for l in open(f):
        r = json.loads(l); v = np.array(r['vec'], np.float32); V[r['id']] = v / np.linalg.norm(v)
    return V
VR = load('out/emb_gemma.jsonl'); VR.update(load('out/emb_indep_gemma.jsonl')); VC = load('out/emb_gemma_clsprompt.jsonl')
rows = [json.loads(l) for l in open('data/intent_all.jsonl')]
A = [r for r in rows if r['split'] == 'train' and r['src'] == 'anchor']; SYN = [r for r in rows if r['split'] == 'train' and r['src'] != 'anchor']
OLD = [r for r in rows if r['split'] == 'test']; NEW = [json.loads(l) for l in open('data/indep_test.jsonl')]
G = [r for r in NEW if r['gen'] == 'gemini']; K = [r for r in NEW if r['gen'] == 'kimi']
labs = sorted({r['intent'] for r in A})
def scores(V, items):
    XA = np.stack([V[r['id']] for r in A]); S = np.stack([V[r['id']] for r in items]) @ XA.T
    sc = np.stack([S[:, [j for j, r in enumerate(A) if r['intent'] == l]].max(1) for l in labs], 1)
    o = np.argsort(-sc, 1); top = sc[np.arange(len(items)), o[:, 0]]; gap = top - sc[np.arange(len(items)), o[:, 1]]
    pred = [labs[i] for i in o[:, 0]]; corr = np.array([p == r['intent'] or (r['intent'] == 'unknown' and p in SAFEU) for p, r in zip(pred, items)])
    return top, gap, pred, corr
def rule(top, gap, pred, t, g, tl, gl):
    look = np.array([p in LOOK for p in pred]); dg = np.array([p == 'document_generate' for p in pred])
    return (((top >= t) & (gap >= g)) | (look & (top >= tl) & (gap >= gl))) & ~dg
def stats(s, t, g, tl, gl):
    top, gap, pred, corr = s; m = rule(top, gap, pred, t, g, tl, gl)
    return m.mean(), (corr[m].mean() if m.any() else float('nan')), int(m.sum())
def calibrate(s, target=0.97, min_n=10, parts=None):
    """max coverage on s subject to sel>=target on s AND on every part in `parts` (each part: >=target or no grants)."""
    best = None
    for t in np.arange(0.60, 0.99, 0.005):
        for g in np.arange(0.01, 0.20, 0.0025):
            cov, sel, n = stats(s, t, g, t - 0.8 * g, g / 2)
            if n < min_n or sel < target: continue
            if parts and any((lambda c: c[2] > 0 and c[1] < target)(stats(p, t, g, t - 0.8 * g, g / 2)) for p in parts): continue
            if best is None or cov > best[0]: best = (cov, sel, n, t, g)
    return best
SV = {}
for nm, V in (('raw', VR), ('cls', VC)):
    for sn, it in (('old', OLD), ('G', G), ('K', K), ('SYN', SYN)):
        SV[nm, sn] = scores(V, it)
    print(f"[{nm}] top1 score p10/p50/p90 on indep:", np.percentile(np.r_[SV[nm, 'G'][0], SV[nm, 'K'][0]], [10, 50, 90]).round(3),
          "gap p50:", np.percentile(np.r_[SV[nm, 'G'][1], SV[nm, 'K'][1]], 50).round(3), "L2 top1 old/G/K:", [round(SV[nm, x][3].mean(), 3) for x in ('old', 'G', 'K')])
def cat(*xs): return tuple(np.concatenate([x[i] for x in xs]) if i != 2 else sum([list(x[2]) for x in xs], []) for i in range(4))
print('== raw space, current constants 0.78/0.10 + lookup 0.70/0.05 (replica)')
for sn in ('old', 'G', 'K', 'SYN'):
    print(f"  {sn}: cov/sel/n =", stats(SV['raw', sn], 0.78, 0.10, 0.70, 0.05))
print('== raw space, recalibrated (for reference)')
for cn, cs in (('G-side', cat(SV['raw', 'SYN'], SV['raw', 'G'])), ('K-side', cat(SV['raw', 'SYN'], SV['raw', 'K']))):
    print(' ', cn, calibrate(cs))
out = {}
print('== cls-prompt space calibration (target sel>=0.97 on the pooled cal set AND on each source: old synth / each generator)')
for cn, cs, tests, parts in (('G-side', cat(SV['cls', 'SYN'], SV['cls', 'G']), ('K', 'old'), [SV['cls', 'SYN'], SV['cls', 'G']]),
                             ('K-side', cat(SV['cls', 'SYN'], SV['cls', 'K']), ('G', 'old'), [SV['cls', 'SYN'], SV['cls', 'K']]),
                             ('ALL', cat(SV['cls', 'SYN'], SV['cls', 'G'], SV['cls', 'K']), ('old',), [SV['cls', 'SYN'], SV['cls', 'G'], SV['cls', 'K']])):
    b = calibrate(cs, parts=parts); cov, sel, n, t, g = b; a = 0.10 / g; bb = 0.78 - a * t
    print(f"  {cn}: cal cov={cov:.3f} sel={sel:.3f} n={n} -> t_p={t:.3f} g_p={g:.4f} lookup=({t-0.8*g:.3f},{g/2:.4f}) map a={a:.3f} b={bb:.3f}")
    for ts in tests:
        c2, s2, n2 = stats(SV['cls', ts], t, g, t - 0.8 * g, g / 2); print(f"     test {ts}: cov={c2:.3f} sel={s2:.3f} n={n2}")
    out[cn] = dict(t=float(t), g=float(g), scale=float(a), shift=float(bb), cal_cov=float(cov), cal_sel=float(sel), cal_n=int(n))
# naive: cls space with the old raw constants (what happens if you only flip the prefix)
print('== cls space with UNCHANGED raw constants (naive flip)')
for sn in ('old', 'G', 'K'): print(f"  {sn}: cov/sel/n =", stats(SV['cls', sn], 0.78, 0.10, 0.70, 0.05))
json.dump(out, open('out/l2_calib.json', 'w'), indent=1)
