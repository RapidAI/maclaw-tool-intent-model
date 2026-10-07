"""§16 Qwen3 tool head: reproduce the Gemma tool-head protocol (scripts/tool_section.py) on Qwen3 (pure-Go) vectors.
Vectors: RoleClassification instruct = the SAME query vector intent L2/L3 uses (single embed per turn).
  VEC=final -> out/qwen3_toolhead/qvec_q3cls_final.jsonl (dumped by local/qwen3-toolhead harness = final-branch numerics)
  VEC=q8p2  -> out/q8k/all_q8p2.jsonl (Q8P two-pass, same numerics; fallback)
Protocols (tool gold = scripts/tool_gold.py, OvR logistic C=10 balanced, exactly as tool_section.py):
  OLD  : train tool_eval train (668) -> test old 262 / indep 644 / Kimi half  (no indep data in training)
  XGEN : GK = old train + 70% Gemini + fix (leak-guarded) -> test Kimi ; KG mirror -> test Gemini (split = per_label_split seed 0,
         identical to scripts/train_cls_heads.py / indep_improve.py). 'xgen644' = GK on Kimi ∪ KG on Gemini.
Gemma reference rows use raw EmbeddingGemma vectors (out/emb_*gemma*.jsonl), same code."""
import json, os, sys, time, random, collections, numpy as np
sys.path.insert(0, 'scripts')
from tool_gold import GOLD
from sklearn.linear_model import LogisticRegression
OUT = 'out/qwen3_toolhead'; VEC = os.environ.get('VEC', 'final')
tools = sorted({d.get('function', d)['name'] for d in json.load(open('data/core_tool_defs.json'))})
ev = [json.loads(l) for l in open('data/tool_eval.jsonl')]
lang = {json.loads(l)['id']: json.loads(l).get('lang') for l in open('data/intent_all.jsonl')}
for r in ev: r['lang'] = lang.get(r['id'])
NEW = [json.loads(l) for l in open('data/indep_test.jsonl')]
FIXALL = [json.loads(l) for l in open('data/fix_anchors.jsonl')]
for r in NEW + FIXALL: r['gold'] = GOLD.get(r['intent'])
def loadv(files):
    V = {}
    for f in files:
        for l in open(f):
            r = json.loads(l); v = np.array(r['vec'], np.float32); V[r['id']] = v / np.linalg.norm(v)
    return V
VQ = loadv([f'{OUT}/qvec_q3cls_final.jsonl'] if VEC == 'final' else ['out/q8k/all_q8p2.jsonl'])
VG = loadv(['out/emb_gemma.jsonl', 'out/emb_indep_gemma.jsonl', 'out/emb_fix_gemma.jsonl'])
def per_label_split(items, fracs, seed):  # verbatim from scripts/indep_improve.py
    rng = random.Random(seed); by = collections.defaultdict(list); out = [[] for _ in fracs]
    for r in items: by[r['intent']].append(r)
    for lab in sorted(by):
        xs = by[lab][:]; rng.shuffle(xs); n = len(xs); cuts = []; acc = 0
        for f in fracs[:-1]: acc += f; cuts.append(int(round(acc * n)))
        parts = np.split(np.array(xs, dtype=object), cuts)
        for o, p in zip(out, parts): o.extend(list(p))
    return out
G = [r for r in NEW if r['gen'] == 'gemini']; K = [r for r in NEW if r['gen'] == 'kimi']
trG, calG = per_label_split(G, [0.7, 0.3], 0); trK, calK = per_label_split(K, [0.7, 0.3], 0)
old_tr = [r for r in ev if r['split'] == 'train']; old_te = [r for r in ev if r['split'] == 'test']
TG = lambda xs: [r for r in xs if r['gold']]  # only items whose intent maps to a first-party tool
def guard_fix(V):  # same leakage guard as indep_improve.py: drop fix sentences with cos > 0.92 to any indep item
    XN = np.stack([V[r['id']] for r in NEW]); return [r for r in FIXALL if float((XN @ V[r['id']]).max()) <= 0.92]
def train(V, items, kind='ovr', seed=0):
    X = np.stack([V[r['id']] for r in items]); d = X.shape[1]
    W = np.zeros((len(tools), d), np.float32); b = np.full(len(tools), -20.0, np.float32)
    if kind == 'ovr':
        for k, t in enumerate(tools):
            y = np.array([t in r['gold'] for r in items])
            if y.sum() == 0: continue
            c = LogisticRegression(C=10, max_iter=300, class_weight='balanced').fit(X, y); W[k] = c.coef_[0]; b[k] = c.intercept_[0]
        return ('ovr', W, b)
    from sklearn.neural_network import MLPClassifier
    Y = np.array([[t in r['gold'] for t in tools] for r in items], int)
    m = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=400, random_state=seed).fit(X, Y)
    return ('mlp', m, None)
def probs(h, V, items):
    X = np.stack([V[r['id']] for r in items])
    if h[0] == 'ovr': return 1 / (1 + np.exp(-(X @ h[1].T + h[2])))
    return h[1].predict_proba(X)
def ranks_of(P): return [[tools[k] for k in np.argsort(-p, kind='stable')] for p in P]
def rk(top, gold, k): return any(t in gold for t in top[:k])
def metrics(R, items):
    d = {'n': len(items)}
    for g, sel in [('all', items)] + [(L, [r for r in items if r.get('lang') == L]) for L in ('zh', 'mix', 'en')]:
        for k in (1, 3, 5, 20):
            d[f'{g}@{k}'] = float(np.mean([rk(R[r['id']], r['gold'], k) for r in sel])) if sel else None
        d[f'n_{g}'] = len(sel)
    d['mrr'] = float(np.mean([next((1 / (j + 1) for j, t in enumerate(R[r['id']]) if t in r['gold']), 0) for r in items]))
    return d
ROWS = []
def rep(method, setname, R, items):
    items = [r for r in items if r['id'] in R]; m = metrics(R, items); m.update(method=method, set=setname); ROWS.append(m)
    f = lambda x: '  –  ' if x is None else f'{x:.3f}'
    print(f"{method:44s} {setname:8s} n={m['n']:3d} R@1 {f(m['all@1'])} R@3 {f(m['all@3'])} R@5 {f(m['all@5'])} R@20 {f(m['all@20'])} | "
          f"zh {f(m['zh@1'])}/{f(m['zh@5'])} mix {f(m['mix@1'])}/{f(m['mix@5'])} en {f(m['en@1'])}/{f(m['en@5'])} MRR {m['mrr']:.3f}")
    return m
SETS = {'old262': old_te, 'indep644': TG(NEW), 'kimi': TG(K), 'gemini': TG(G)}
assert len(SETS['old262']) == 262, len(SETS['old262'])
print('indep tool-mappable', len(SETS['indep644']), 'kimi', len(SETS['kimi']), 'gemini', len(SETS['gemini']), 'VEC', VEC)
import pickle
CACHE = f'{OUT}/cache_heads_{VEC}.pkl'
heads = {}; PROB = {}
if os.path.exists(CACHE) and not os.environ.get('RETRAIN'):
    heads, PROB = pickle.load(open(CACHE, 'rb')); print('loaded cached heads', CACHE)
for bb, V in (() if heads else (('qwen3', VQ), ('gemma', VG))):
    FIX = TG(guard_fix(V)); print(bb, 'fix kept (tool-mappable)', len(FIX), '/', len(TG(FIXALL)))
    t0 = time.time(); hO = train(V, old_tr); print(bb, 'OLD train s %.1f, tools with positives %d/49' % (time.time() - t0, int((hO[2] > -20).sum())))
    trGK = old_tr + TG(trG) + FIX; trKG = old_tr + TG(trK) + FIX
    assert not {r['id'] for r in trGK} & {r['id'] for r in K} and not {r['id'] for r in trKG} & {r['id'] for r in G}
    hGK = train(V, trGK); hKG = train(V, trKG)
    print(bb, f'GK n_train={len(trGK)} pos tools {int((hGK[2] > -20).sum())}; KG n_train={len(trKG)} pos tools {int((hKG[2] > -20).sum())}')
    heads[bb] = dict(OLD=hO, GK=hGK, KG=hKG, FIX=FIX, trGK=trGK, trKG=trKG)
    for name, h in (('OLD', hO), ('GK', hGK), ('KG', hKG)):
        for s, items in SETS.items(): PROB[(bb, name, s)] = (items, probs(h, V, items))
if not os.path.exists(CACHE) or os.environ.get('RETRAIN'): pickle.dump((heads, PROB), open(CACHE, 'wb'))
# sanity: Gemma OLD head must reproduce heads/tool_head_ovr_gemma.json numbers (0.920 / 0.691)
def R_of(bb, name, s):
    items, P = PROB[(bb, name, s)]; return {r['id']: x for r, x in zip(items, ranks_of(P))}
def R_xgen(bb):
    R = R_of(bb, 'GK', 'kimi'); R.update(R_of(bb, 'KG', 'gemini')); return R
def loadA(f):
    A = {}; S = {}
    if not os.path.exists(f): return None, None
    for l in open(f):
        try: r = json.loads(l)
        except ValueError: continue  # partial last line while the harness is still writing
        A[r['id']] = [x['n'] for x in r['top']]; S[r['id']] = {x['n']: x['s'] for x in r['top']}
    return A, S
TA = {'TrackA Qwen3 (q=cls vec, reuse L2; doc=no instr)': loadA(f'{OUT}/trackA_q3cls.jsonl')[0],
      'TrackA Qwen3 (q=Embed no instr, Router default)': loadA(f'{OUT}/trackA_q3none.jsonl')[0],
      'TrackA Qwen3 (q=retrieval instr)': loadA(f'{OUT}/trackA_q3query.jsonl')[0]}
ga = loadA('out/finalize/tool_trackA_test.jsonl')[0] or {}; gi = loadA(f'{OUT}/trackA_gemma_indep.jsonl')[0] or {}
ga.update(gi); TA['[ref] TrackA Gemma (Router default)'] = ga
if __name__ == '__main__':
    print('== primary: Qwen3-only')
    for s in ('old262', 'indep644', 'kimi'): rep('Qwen3 tool head OvR [OLD train]', s, R_of('qwen3', 'OLD', s), SETS[s])
    rep('Qwen3 tool head OvR [xgen GK→Kimi∪KG→Gemini]', 'xgen644', R_xgen('qwen3'), SETS['indep644'])
    rep('Qwen3 tool head OvR [GK]', 'kimi', R_of('qwen3', 'GK', 'kimi'), SETS['kimi'])
    for name, A in TA.items():
        if not A: continue
        for s in ('old262', 'indep644', 'kimi'): rep(name, s, A, SETS[s])
    print('== reference: Gemma')
    for s in ('old262', 'indep644', 'kimi'): rep('[ref] Gemma tool head OvR [OLD train]', s, R_of('gemma', 'OLD', s), SETS[s])
    rep('[ref] Gemma tool head OvR [xgen]', 'xgen644', R_xgen('gemma'), SETS['indep644'])
    rep('[ref] Gemma tool head OvR [GK]', 'kimi', R_of('gemma', 'GK', 'kimi'), SETS['kimi'])
    th = json.load(open('heads/tool_head_ovr_gemma.json')); W = np.array(th['W']); b = np.array(th['b'])
    for s in ('old262', 'indep644'):
        items = SETS[s]; P = 1 / (1 + np.exp(-(np.stack([VG[r['id']] for r in items]) @ W.T + b)))
        rep('[ref] Gemma tool head (shipped json, sanity)', s, {r['id']: x for r, x in zip(items, ranks_of(P))}, items)
    if os.environ.get('MLP'):
        print('== MLP variant (Qwen3, multi-label MLP 256)')
        V = VQ; hm = train(V, old_tr, 'mlp')
        for s in ('old262', 'indep644'):
            items = SETS[s]; rep('Qwen3 tool head MLP [OLD train]', s, {r['id']: x for r, x in zip(items, ranks_of(probs(hm, V, items)))}, items)
        hgk = train(V, heads['qwen3']['trGK'], 'mlp'); hkg = train(V, heads['qwen3']['trKG'], 'mlp')
        R = {r['id']: x for r, x in zip(SETS['kimi'], ranks_of(probs(hgk, V, SETS['kimi'])))}
        R.update({r['id']: x for r, x in zip(SETS['gemini'], ranks_of(probs(hkg, V, SETS['gemini'])))})
        rep('Qwen3 tool head MLP [xgen]', 'xgen644', R, SETS['indep644'])
    json.dump(ROWS, open(f'{OUT}/toolhead_eval_{VEC}.json', 'w'), indent=1)

# ---------------- union coverage + export (run after Track A dumps exist) ----------------
def scope_of():  # production intent scope: intent_defs.json tools of the Go-predicted primary (+secondary) label
    defs = {x['label']: x.get('tools') or [] for x in json.load(open('data/intent_defs.json'))['defs']}
    S = {}
    for f in ('out/finalize/acc_qwen3_old339.jsonl', 'out/finalize/acc_qwen3_indep839_crossgen.jsonl'):
        for l in open(f):
            r = json.loads(l); labs = [r.get('cls_primary')] + ([r['cls_secondary']] if isinstance(r.get('cls_secondary'), str) else [])
            S[r['id']] = set(t for L in labs if L for t in defs.get(L, []))
    return S
def union_table(bb, Afile_key):
    S = scope_of(); A = TA[Afile_key]; out = []
    sets = {'old262': (SETS['old262'], PROB[(bb, 'OLD', 'old262')]),
            'xgen644': (SETS['kimi'] + SETS['gemini'], None), 'kimi(GK)': (SETS['kimi'], PROB[(bb, 'GK', 'kimi')])}
    Pk = dict(zip([r['id'] for r in SETS['kimi']], PROB[(bb, 'GK', 'kimi')][1])); Pk.update(zip([r['id'] for r in SETS['gemini']], PROB[(bb, 'KG', 'gemini')][1]))
    Po = dict(zip([r['id'] for r in SETS['old262']], PROB[(bb, 'OLD', 'old262')][1]))
    def cand(i, P, k, th, N):
        c = set(S.get(i, set()))
        if k: c |= {tools[j] for j in np.argsort(-P[i], kind='stable')[:k] if P[i][j] >= th}
        if N: c |= set(A[i][:N])
        return c
    grid = [(0, 0, 0), (0, 0, 5), (0, 0, 10), (0, 0, 20)] + [(k, th, N) for k in (3, 5) for th in (0.3, 0.5, 0.7, 0.9) for N in (0, 5, 10)] + [(5, 0.0, 0), (5, 0.0, 10)]
    for name, items, P in (('old262', SETS['old262'], Po), ('xgen644', SETS['kimi'] + SETS['gemini'], Pk)):
        items = [r for r in items if r['id'] in A]
        base = np.mean([bool(S.get(r['id'], set()) & set(r['gold'])) for r in items])
        for k, th, N in grid:
            C = [cand(r['id'], P, k, th, N) for r in items]
            rec = np.mean([bool(c & set(r['gold'])) for c, r in zip(C, items)])
            hk = [len({tools[j] for j in np.argsort(-P[r['id']], kind='stable')[:k] if P[r['id']][j] >= th}) if k else 0 for r in items]
            out.append(dict(set=name, n=len(items), k=k, theta=th, N=N, recall=float(rec), mean_cands=float(np.mean([len(c) for c in C])),
                            mean_head_cands=float(np.mean(hk)), scope_only=float(base)))
            print(f"{bb} {name:8s} n={len(items)} scope∪head(k={k},θ={th})∪A(N={N}): recall={rec:.3f} |C|={np.mean([len(c) for c in C]):.1f} head|={np.mean(hk):.2f}  (scope alone {base:.3f})")
    return out
def export(h, path, note, theta=None, k=None, train_desc=''):
    import hashlib
    W, b = h[1], h[2]; trained = [bool(x > -20) for x in b]
    ltv = 'core%d@%s' % (len(tools), hashlib.sha256('\n'.join(tools).encode()).hexdigest()[:12])
    mid = json.loads(open(f'{OUT}/model_id.json').read())['model_id']
    js = dict(format='maclaw-tool-head-ovr-v1', embedder_model_id=mid, embed_role='classification', dims=int(W.shape[1]), l2norm=True,
              label_table_version=ltv, tools=tools, trained=trained, W=[[round(float(x), 6) for x in row] for row in W],
              b=[round(float(x), 6) for x in b], theta=theta, top_k=k, train=train_desc, note=note)
    json.dump(js, open(path, 'w'), separators=(',', ':'))
    np.savez(path.replace('.json', '.npz'), W=W, b=b, tools=np.array(tools))
    return js
if __name__ == '__main__' and os.environ.get('UNION'):
    U = []
    for key in [k for k, v in TA.items() if v and 'Qwen3' in k]:
        print('== union with', key); U += [dict(u, trackA=key) for u in union_table('qwen3', key)]
    print('== [ref] union Gemma head + Gemma Track A'); U += [dict(u, trackA='[ref] gemma') for u in union_table('gemma', '[ref] TrackA Gemma (Router default)')]
    json.dump(U, open(f'{OUT}/union_{VEC}.json', 'w'), indent=1)
if __name__ == '__main__' and os.environ.get('EXPORT'):
    V = VQ; FIX = heads['qwen3']['FIX']; allitems = old_tr + old_te + TG(NEW) + FIX
    th, k = float(os.environ.get('THETA', '0.5')), int(os.environ.get('TOPK', '5'))
    hA = train(V, allitems)
    export(hA, 'heads/tool_head_ovr_qwen3go.json', f'OvR logistic (C=10, balanced) on L2-normalised Qwen3-Embedding-0.6B Q8_0 pure-Go vectors, RoleClassification instruct (= intent L2/L3 query vector); trained on ALL labelled data (old train+test, indep Gemini+Kimi tool-mappable, fix); held-out estimate = §16 xgen protocol', th, k,
           f'old_train {len(old_tr)} + old_test {len(old_te)} + indep {len(TG(NEW))} + fix {len(FIX)} = {len(allitems)}')
    export(heads['qwen3']['GK'], 'heads/tool_head_ovr_qwen3go_GK.json', 'eval head: old train + 70% Gemini + fix; test on Kimi', th, k, 'GK')
    export(heads['qwen3']['KG'], 'heads/tool_head_ovr_qwen3go_KG.json', 'eval head: old train + 70% Kimi + fix; test on Gemini', th, k, 'KG')
    # parity fixture for Go: 50 Kimi vectors with Python probabilities from the GK head
    items = SETS['kimi'][:50]; js = json.load(open('heads/tool_head_ovr_qwen3go_GK.json'))
    Wj, bj = np.array(js['W']), np.array(js['b']); P = 1 / (1 + np.exp(-(np.stack([V[r['id']] for r in items]) @ Wj.T + bj)))
    P[:, ~np.array(js['trained'])] = 0  # Go returns 0 for untrained tools
    with open(f'{OUT}/parity_GK_kimi50.jsonl', 'w') as f:
        for r, p in zip(items, P): f.write(json.dumps({'id': r['id'], 'vec': [float(x) for x in V[r['id']]], 'p': [float(x) for x in p]}) + '\n')
    print('exported', len(allitems))
