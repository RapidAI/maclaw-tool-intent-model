"""Extra stats for REPORT 8.3 from out/indep_eval_preds.json (no retraining): coverage@tau curve, sensitive mis-exposures, per-generator x lang."""
import json, numpy as np, collections
meta = json.load(open('data/intent_meta.json')); SENS = set(meta['sensitive']); SAFEU = set(meta['safe_unknown'])
P = json.load(open('out/indep_eval_preds.json'))
rows = [json.loads(l) for l in open('data/intent_all.jsonl')]; old = [r for r in rows if r['split'] == 'test']
NEW = [json.loads(l) for l in open('data/indep_test.jsonl')]
ok = lambda g, p: p == g or (g == 'unknown' and p in SAFEU)
for nm in ('MLP anchor+synth (Gemma)', 'MLP anchor-only (Gemma)'):
    print('==', nm)
    for sn, items in (('old', old), ('indep', NEW)):
        pr = [P[nm][r['id']] for r in items]; g = [r['intent'] for r in items]
        corr = np.array([ok(a, p) for a, (p, c) in zip(g, pr)]); conf = np.array([c for p, c in pr]); isS = np.array([p in SENS for p, c in pr])
        for tau, ts in ((0.5, 0.9), (0.7, 0.9), (0.79, 0.79), (0.79, 0.9), (0.79, 0.95), (0.9, 0.95), (0.95, 0.95)):
            need = np.where(isS, max(tau, ts), tau); m = conf >= need
            fe = sum(1 for a, (p, c), k in zip(g, pr, m) if k and p in SENS and p != a)
            print(f' {sn:5s} tau={tau:.2f} tau_s={ts:.2f} cov={m.mean():.3f} sel_acc={corr[m].mean():.3f} sensFE={fe} ({fe/len(items)*100:.1f}%)')
        # smallest tau reaching sel_acc>=0.97 on this set (oracle, for information only)
        for t in np.linspace(0.3, 0.999, 300):
            m = conf >= t
            if m.sum() and corr[m].mean() >= 0.97: print(f' {sn} oracle tau for sel_acc>=0.97: {t:.3f} cov={m.mean():.3f}'); break
        else: print(f' {sn} oracle: no tau reaches 0.97')
nm = 'MLP anchor+synth (Gemma)'
g = [r['intent'] for r in NEW]; pr = [P[nm][r['id']] for r in NEW]
print('sensitive mis-exposures (indep, tau=0.79, tau_s=0.90): gold->pred counts')
fe = [(r['intent'], p, round(c, 3), r['gen'], r['kind'], r['text'][:60]) for r, (p, c) in zip(NEW, pr) if p in SENS and p != r['intent'] and c >= 0.90]
print(len(fe), collections.Counter((a, b) for a, b, *_ in fe).most_common(15))
for x in fe[:12]: print('  ', x)
print('gen x lang acc:')
for G in ('gemini', 'kimi'):
    for L in ('zh', 'mix', 'en'):
        s = [ok(r['intent'], P[nm][r['id']][0]) for r in NEW if r['gen'] == G and r['lang'] == L]; print(f'  {G} {L} n={len(s)} acc={np.mean(s):.3f}')
print('acc vs max_cos_train bins (style distance):')
for lo, hi in ((0, .7), (.7, .75), (.75, .8), (.8, .85), (.85, 1)):
    s = [ok(r['intent'], P[nm][r['id']][0]) for r in NEW if lo <= r['max_cos_train'] < hi]; print(f'  [{lo},{hi}) n={len(s)} acc={np.mean(s) if s else float("nan"):.3f}')
print('by style:')
for st, n in collections.Counter(r.get('style') for r in NEW).most_common():
    s = [ok(r['intent'], P[nm][r['id']][0]) for r in NEW if r.get('style') == st]; print(f'  {st} n={n} acc={np.mean(s):.3f}')
print('unknown gold: acc', np.mean([ok('unknown', P[nm][r['id']][0]) for r in NEW if r['intent'] == 'unknown']), sum(r['intent'] == 'unknown' for r in NEW))
