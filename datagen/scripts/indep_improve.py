"""Section 8.5: retrain MLP head on new-distribution data with a proper held-out calibration split.
Splits (test items are NEVER used for training or calibration):
  G->K : train = old train (+fix) + 70% Gemini indep ; cal = other 30% Gemini ; test = all Kimi
  K->G : mirror
  MIX s: pooled indep, per-label 50% train / 20% cal / 30% test, seeds 0..2
Variants: old head as-is (old T, tau=0.79) | old head recalibrated on cal | retrain old+fix | retrain old+indep | retrain old+indep+fix.
T fitted on cal logits (NLL), tau = smallest t with cal selective acc >= target (0.97, else report 0.90 target too). tau_s = max(tau, 0.90)."""
import json, sys, os, numpy as np, collections, random
sys.path.insert(0, 'scripts')
import train_heads as T
meta = json.load(open('data/intent_meta.json')); SENS = set(meta['sensitive']); SAFEU = set(meta['safe_unknown'])
BB = os.environ.get('BB', 'gemma')  # gemma (production Go embedder) | qwen3e (Qwen3-Embedding-0.6B, no fix-anchor embeddings)
V = {}
if BB == 'gemma':
    for f in ('out/emb_gemma.jsonl', 'out/emb_indep_gemma.jsonl', 'out/emb_fix_gemma.jsonl'):
        for l in open(f):
            r = json.loads(l); v = np.array(r['vec'], np.float32); V[r['id']] = v / np.linalg.norm(v)
elif BB == 'gemma_cls':  # same Go embedder, EmbeddingGemma 'task: classification | query: ' prompt (RoleClassification)
    for l in open('out/emb_gemma_clsprompt.jsonl'):
        r = json.loads(l); v = np.array(r['vec'], np.float32); V[r['id']] = v / np.linalg.norm(v)
elif BB == 'qwen3go':  # pure-Go maclaw Qwen3-Embedding runner, Q8_0 GGUF, RoleClassification instruct (section 10)
    for f in ('out/emb_qwen3go.npy', 'out/emb_indep_qwen3go.npy'):
        X = np.load(f); ids = json.load(open(f + '.ids.json')); V.update({i: x / np.linalg.norm(x) for i, x in zip(ids, X)})
    for l in open('out/emb_fix_qwen3go.jsonl'):
        r = json.loads(l); v = np.array(r['vec'], np.float32); V[r['id']] = v / np.linalg.norm(v)
else:
    for f in ('out/emb_qwen3e.npy', 'out/emb_indep_qwen3e.npy'):
        X = np.load(f); ids = json.load(open(f + '.ids.json')); V.update({i: x / np.linalg.norm(x) for i, x in zip(ids, X)})
rows = T.rows; old_tr = [r for r in rows if r['split'] == 'train']; old_te = [r for r in rows if r['split'] == 'test']
NEW = [json.loads(l) for l in open('data/indep_test.jsonl')]
FIX = [json.loads(l) for l in open('data/fix_anchors.jsonl')] if BB in ('gemma', 'gemma_cls', 'qwen3go') else []
# leakage guard: drop fix sentences too close to any indep test item
XN = np.stack([V[r['id']] for r in NEW]); keep = []
for r in FIX:
    m = float((XN @ V[r['id']]).max())
    if m <= 0.92: keep.append(r)
    else: print('drop fix (near-dup of indep item)', r['text'], round(m, 3))
if FIX: print(f'fix kept {len(keep)}/{len(FIX)}; max cos fix->indep p50={np.median([(XN @ V[r["id"]]).max() for r in FIX]):.3f}')
FIX = keep
OLDH = None if BB == 'gemma_cls' else json.load(open({'gemma': 'heads/head_mlp_gemma.json', 'qwen3go': 'heads/head_mlp_qwen3go.json'}.get(BB, 'heads/head_mlp_qwen3e.json')))
ok = lambda g, p: p == g or (g == 'unknown' and p in SAFEU)
def ece(conf, corr, bins=10):
    e = 0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return e
class Model:
    def __init__(s, labels, logit_fn, Tm=1.0): s.labels, s.logit_fn, s.T = labels, logit_fn, Tm
    def logits(s, items): return s.logit_fn(np.stack([V[r['id']] for r in items]))
def old_model():
    w1, b1, w2, b2 = (np.array(OLDH[k]) for k in ('w1', 'b1', 'w2', 'b2'))
    return Model(OLDH['labels'], lambda X: np.maximum(0, X @ w1.T + b1) @ w2.T + b2, OLDH['temperature'])
def train(items, seed=0):
    labels = sorted({r['intent'] for r in items}); lid = {l: k for k, l in enumerate(labels)}
    clf = T.make('mlp', seed).fit(np.stack([V[r['id']] for r in items]), np.array([lid[r['intent']] for r in items]))
    def lf(X):
        z = np.full((len(X), len(labels)), -30.0); z[:, clf.classes_] = T.logits_of(clf, X); return z
    return Model(labels, lf)
def calibrate(m, cal):
    z = m.logits(cal); y = np.array([m.labels.index(r['intent']) for r in cal]); m.T = T.fit_temp(z, y)
    P = T.softmax(z / m.T); conf = P.max(1); corr = np.array([ok(r['intent'], m.labels[j]) for r, j in zip(cal, P.argmax(1))])
    taus = {}
    for tgt in (0.97, 0.90):
        taus[tgt] = 0.99
        for t in np.linspace(0.30, 0.99, 70):
            k = conf >= t
            if k.sum() >= 5 and corr[k].mean() >= tgt: taus[tgt] = float(t); break
    return taus
def evaluate(m, items, tau, tau_s=0.90):
    P = T.softmax(m.logits(items) / m.T); pred = [m.labels[j] for j in P.argmax(1)]; conf = P.max(1); g = [r['intent'] for r in items]
    corr = np.array([ok(a, b) for a, b in zip(g, pred)]); need = np.array([max(tau, tau_s) if p in SENS else tau for p in pred]); acc = conf >= need
    fe = sum(1 for a, p, k in zip(g, pred, acc) if k and p in SENS and p != a)
    d = dict(n=len(items), top1=corr.mean(), ece=ece(conf, corr), cov=acc.mean(), sel=corr[acc].mean() if acc.any() else float('nan'), fe=fe, fe_pct=fe / len(items), tau=tau, T=m.T)
    for L in ('zh', 'mix', 'en'):
        k = np.array([r.get('lang') == L for r in items]); d[L] = corr[k].mean() if k.any() else float('nan')
    d['conf'] = collections.Counter((a, p) for a, p in zip(g, pred) if not ok(a, p)).most_common(6)
    return d
def per_label_split(items, fracs, seed):
    rng = random.Random(seed); by = collections.defaultdict(list); out = [[] for _ in fracs]
    for r in items: by[r['intent']].append(r)
    for lab in sorted(by):
        xs = by[lab][:]; rng.shuffle(xs); n = len(xs); cuts = []; acc = 0
        for f in fracs[:-1]: acc += f; cuts.append(int(round(acc * n)))
        parts = np.split(np.array(xs, dtype=object), cuts)
        for o, p in zip(out, parts): o.extend(list(p))
    return out
results = []
def run_split(name, tr_new, cal, test, seed=0):
    assert not ({r['id'] for r in test} & ({r['id'] for r in tr_new} | {r['id'] for r in cal}))
    assert not ({r['id'] for r in tr_new} & {r['id'] for r in cal})
    variants = []
    if OLDH:
        m0 = old_model(); variants.append(('旧头原样 (旧 T, τ=0.79)', m0, {0.97: OLDH['tau'], 0.90: OLDH['tau']}))
        m1 = old_model(); variants.append(('旧头 + 新分布 cal 重标定', m1, calibrate(m1, cal)))
    else:
        mb = train(old_tr, seed); variants.append(('重训 仅旧训练 + 新分布 cal 标定', mb, calibrate(mb, cal)))
    if FIX: m2 = train(old_tr + FIX, seed); variants.append(('重训 旧训练+fix', m2, calibrate(m2, cal)))
    m3 = train(old_tr + tr_new, seed); variants.append((f'重训 旧训练+新分布({len(tr_new)})', m3, calibrate(m3, cal)))
    if FIX: m4 = train(old_tr + tr_new + FIX, seed); variants.append((f'重训 旧训练+新分布+fix', m4, calibrate(m4, cal)))
    for vn, m, taus in variants:
        for tgt in (0.97, 0.90):
            if vn.startswith('旧头原样') and tgt == 0.90: continue
            d = evaluate(m, test, taus[tgt]); d.update(split=name, variant=vn, tau_target=tgt, n_cal=len(cal), n_trnew=len(tr_new))
            d['old_test_top1'] = evaluate(m, old_te, taus[tgt])['top1']
            results.append(d)
            print(f"{name:6s} | {vn:28s} | tgt={tgt:.2f} τ={d['tau']:.2f} T={d['T']:.2f} | n={d['n']} top1={d['top1']:.3f} zh={d['zh']:.3f} mix={d['mix']:.3f} en={d['en']:.3f} ECE={d['ece']:.3f} cov={d['cov']:.3f} sel={d['sel']:.3f} sensFE={d['fe']} ({d['fe_pct']*100:.1f}%) | old_test={d['old_test_top1']:.3f}", flush=True)
    print(f"{name} top confusions (last variant):", results[-1]['conf'], flush=True)
G = [r for r in NEW if r['gen'] == 'gemini']; K = [r for r in NEW if r['gen'] == 'kimi']
for A, B, nm in ((G, K, 'G->K'), (K, G, 'K->G')):
    trA, calA = per_label_split(A, [0.7, 0.3], 0); run_split(nm, trA, calA, B)
for s in (0, 1, 2):
    tr, cal, te = per_label_split(NEW, [0.5, 0.2, 0.3], s); run_split(f'MIX{s}', tr, cal, te, seed=s)
json.dump(results, open('out/indep_improve.json' if BB == 'gemma' else f'out/indep_improve_{BB}.json', 'w'), indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())
# mixed summary mean/sd
print('== MIX mean ± sd over 3 seeds')
for vn in dict.fromkeys(d['variant'] for d in results):
    for tgt in (0.97, 0.90):
        ds = [d for d in results if d['split'].startswith('MIX') and d['variant'] == vn and d['tau_target'] == tgt]
        if not ds: continue
        f = lambda k: f"{np.mean([d[k] for d in ds]):.3f}±{np.std([d[k] for d in ds]):.3f}"
        print(f"{vn:28s} tgt={tgt:.2f} top1={f('top1')} ECE={f('ece')} cov={f('cov')} sel={f('sel')} sensFE%={f('fe_pct')} τ={f('tau')} old_test={f('old_test_top1')}")
