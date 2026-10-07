"""Section 17: multi-family Qwen3 (pure-Go, RoleClassification) intent MLP head + tool OvR head, leave-one-family-out.
Head recipe = head_mlp_qwen3go_GK.json (sklearn MLP 1024->256, seed 0; T by NLL on a held-out calibration slice; tau = smallest t with
cal selective acc >= 0.95; tau_s from max(tau,0.95) raised until cal sensitive precision >= 0.97, cap 0.99).
Test items (indep 839 / ambiguous 95 / old test 339) are never used for training or calibration (asserted).
usage: XF_TAG=r1 python scripts/xf_train.py [export]"""
import json, sys, os, collections, numpy as np, time
sys.path.insert(0, 'scripts'); sys.argv_ = sys.argv; sys.argv = ['x']
import train_heads as T
from tool_gold import GOLD
from sklearn.linear_model import LogisticRegression
TAG = os.environ.get('XF_TAG', 'r1'); EXPORT = 'export' in sys.argv_
meta = json.load(open('data/intent_meta.json')); SENS = set(meta['sensitive']); SAFEU = set(meta['safe_unknown'])
ok = lambda g, p: p == g or (g == 'unknown' and p in SAFEU)
V = {}
for fn in ('out/emb_qwen3go.npy', 'out/emb_indep_qwen3go.npy'):
    X = np.load(fn); ids = json.load(open(fn + '.ids.json')); V.update({i: x / np.linalg.norm(x) for i, x in zip(ids, X)})
for fn in ('out/emb_fix_qwen3go.jsonl', 'out/xf/emb_xf_qwen3go.jsonl'):
    for l in open(fn): r = json.loads(l); v = np.array(r['vec'], np.float32); V[r['id']] = v / np.linalg.norm(v)
rows = T.rows; old_tr = [r for r in rows if r['split'] == 'train']; old_te = [r for r in rows if r['split'] == 'test']
IND = [json.loads(l) for l in open('data/indep_test.jsonl')]; AMB = [json.loads(l) for l in open('data/indep_ambiguous.jsonl')]
XN = np.stack([V[r['id']] for r in IND])
FIX = [r for r in (json.loads(l) for l in open('data/fix_anchors.jsonl')) if float((XN @ V[r['id']]).max()) <= 0.92]
XF = [json.loads(l) for l in open('data/train_xfamily/train_xf.jsonl')]
TEST_IDS = {r['id'] for r in IND + AMB + old_te}; TEST_TXT = {r['text'].strip() for r in IND + AMB + old_te}
assert not ({r['id'] for r in XF + FIX + old_tr} & TEST_IDS) and not ({r['text'].strip() for r in XF} & TEST_TXT)
FAMS = sorted(f for f, n in collections.Counter(r['fam'] for r in XF).items() if n >= 20)
print(f'[{TAG}] base old_tr={len(old_tr)} fix={len(FIX)} xf={len(XF)} per fam', dict(collections.Counter(r['fam'] for r in XF)), flush=True)
def per_label_split(items, frac, seed):
    rng = np.random.RandomState(seed); by = collections.defaultdict(list); a, b = [], []
    for r in items: by[r['intent']].append(r)
    for lab in sorted(by):
        xs = by[lab][:]; rng.shuffle(xs); k = int(round(frac * len(xs))); a += xs[:k]; b += xs[k:]
    return a, b
def ece(conf, corr, bins=10):
    e = 0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)
class Head:
    def __init__(s, items, seed=0):
        s.labels = sorted({r['intent'] for r in items}); lid = {l: k for k, l in enumerate(s.labels)}
        s.clf = T.make('mlp', seed).fit(np.stack([V[r['id']] for r in items]), np.array([lid[r['intent']] for r in items]))
        s.w1, s.b1, s.w2, s.b2 = s.clf.coefs_[0], s.clf.intercepts_[0], s.clf.coefs_[1], s.clf.intercepts_[1]; s.T = 1.0
    def logits(s, items): X = np.stack([V[r['id']] for r in items]); return np.maximum(0, X @ s.w1 + s.b1) @ s.w2 + s.b2
    def calibrate(s, cal):
        z = s.logits(cal); y = np.array([s.labels.index(r['intent']) for r in cal]); s.T = T.fit_temp(z, y)
        P = T.softmax(z / s.T); conf = P.max(1); pred = [s.labels[j] for j in P.argmax(1)]; corr = np.array([ok(r['intent'], p) for r, p in zip(cal, pred)])
        s.tau = 0.99
        for t in np.linspace(0.30, 0.99, 70):
            k = conf >= t
            if k.sum() >= 5 and corr[k].mean() >= 0.95: s.tau = float(t); break
        isS = np.array([p in SENS for p in pred]); s.tau_s = max(s.tau, 0.95)
        for t in np.arange(s.tau_s, 0.991, 0.01):
            k = (conf >= t) & isS; s.tau_s = float(t)
            if k.sum() == 0 or corr[k].mean() >= 0.97: break
        return s
class Loaded(Head):
    def __init__(s, path):
        h = json.load(open(path)); s.labels = h['labels']; s.T, s.tau, s.tau_s = h['temperature'], h['tau'], h['tau_s']
        s.w1, s.b1, s.w2, s.b2 = np.array(h['w1']).T, np.array(h['b1']), np.array(h['w2']).T, np.array(h['b2'])
def evaluate(h, items):
    P = T.softmax(h.logits(items) / h.T); pred = [h.labels[j] for j in P.argmax(1)]; conf = P.max(1)
    corr = np.array([ok(r['intent'], p) for r, p in zip(items, pred)]); top3 = np.argsort(-P, 1)[:, :3]
    t3 = np.mean([ok(r['intent'], r['intent']) and (r['intent'] in [h.labels[j] for j in tt] or (r['intent'] == 'unknown' and any(h.labels[j] in SAFEU for j in tt))) for r, tt in zip(items, top3)])
    need = np.array([h.tau_s if p in SENS else h.tau for p in pred]); acc = conf >= need
    fe = int(sum(1 for r, p, a in zip(items, pred, acc) if a and p in SENS and p != r['intent']))
    d = dict(n=len(items), top1=float(corr.mean()), top3=float(t3), ece=ece(conf, corr), cov=float(acc.mean()),
             sel=float(corr[acc].mean()) if acc.any() else float('nan'), sensFE=fe, sensFE_pct=fe / len(items), T=float(h.T), tau=h.tau, tau_s=h.tau_s)
    for L in ('zh', 'mix', 'en'):
        k = np.array([r.get('lang') == L for r in items]); d[L] = float(corr[k].mean()) if k.any() else float('nan')
    d['confusions'] = collections.Counter((r['intent'], p) for r, p in zip(items, pred) if not ok(r['intent'], p)).most_common(5)
    return d
fmt = lambda d: f"n={d['n']} top1={d['top1']:.3f} top3={d['top3']:.3f} zh/mix/en={d['zh']:.3f}/{d['mix']:.3f}/{d['en']:.3f} ECE={d['ece']:.3f} cov={d['cov']:.3f} sel={d['sel']:.3f} sensFE={d['sensFE']}({100*d['sensFE_pct']:.1f}%) T={d['T']:.2f} tau={d['tau']:.2f} tau_s={d['tau_s']:.2f}"
RES = dict(tag=TAG, counts=dict(collections.Counter(r['fam'] for r in XF)), lofo={}, curve=[], tool={})
GK, KG = Loaded('heads/head_mlp_qwen3go_GK.json'), Loaded('heads/head_mlp_qwen3go_KG.json')
FRACS = [float(x) for x in os.environ.get('XF_FRACS', '0,0.5,1').split(',')]
for F in FAMS:
    test = [r for r in XF if r['fam'] == F]; others = [r for r in XF if r['fam'] != F]
    tr_o, cal = per_label_split(others, 0.7, 0)
    assert not ({r['id'] for r in test} & {r['id'] for r in tr_o + cal})
    row = dict(n=len(test), GK=evaluate(GK, test), KG=evaluate(KG, test))
    print(f'== LOFO hold-out {F} (n={len(test)}; train others={len(tr_o)}, cal={len(cal)})', flush=True)
    print(f'   GK head (current)      | {fmt(row["GK"])}', flush=True)
    for fr in FRACS:
        sub = per_label_split(tr_o, fr, 1)[0] if fr < 1 else tr_o
        h = Head(old_tr + FIX + sub).calibrate(cal); d = evaluate(h, test); di = evaluate(h, IND)
        row[f'xf_{fr}'] = d; row[f'xf_{fr}_indep839'] = di
        RES['curve'].append(dict(heldout=F, frac=fr, n_xf_train=len(sub), lofo_top1=d['top1'], indep_top1=di['top1'], lofo_sel=d['sel'], lofo_cov=d['cov']))
        print(f'   xf frac={fr:.2f} (+{len(sub):4d}) | {fmt(d)} || indep839 top1={di["top1"]:.3f} cov={di["cov"]:.3f} sel={di["sel"]:.3f} sensFE={di["sensFE"]}', flush=True)
    RES['lofo'][F] = row
print('== GK head on indep839 (Gemini half is in-family for GK):', fmt(evaluate(GK, IND)))
# pooled LOFO summary
for key in ['GK'] + [f'xf_{fr}' for fr in FRACS]:
    n = sum(RES['lofo'][F]['n'] for F in FAMS); g = lambda m: sum(RES['lofo'][F][key][m] * RES['lofo'][F]['n'] for F in FAMS) / n
    fe = sum(RES['lofo'][F][key]['sensFE'] for F in FAMS)
    print(f'POOLED LOFO {key:10s} n={n} top1={g("top1"):.3f} ECE(avg)={g("ece"):.3f} cov={g("cov"):.3f} sensFE={fe} ({100*fe/n:.1f}%)', flush=True)
    RES.setdefault('pooled', {})[key] = dict(n=n, top1=g('top1'), ece=g('ece'), cov=g('cov'), sensFE=fe)
# ---- tool head (OvR logistic, C=10 balanced, gold from intent via tool_gold.GOLD); LOFO
tools = sorted({d.get('function', d)['name'] for d in json.load(open('data/core_tool_defs.json'))})
EV = {json.loads(l)['id']: json.loads(l) for l in open('data/tool_eval.jsonl')}
def gold(r): return EV[r['id']]['gold'] if r['id'] in EV else GOLD.get(r['intent'], [])
def tool_fit(items):
    items = [r for r in items if gold(r)]; X = np.stack([V[r['id']] for r in items]); W = np.zeros((len(tools), X.shape[1])); b = np.full(len(tools), -20.0)
    for k, t in enumerate(tools):
        y = np.array([t in gold(r) for r in items])
        if y.sum() == 0: continue
        c = LogisticRegression(C=10, max_iter=300, class_weight='balanced').fit(X, y); W[k], b[k] = c.coef_[0], c.intercept_[0]
    return W, b
def tool_eval(Wb, items):
    items = [r for r in items if gold(r)]; X = np.stack([V[r['id']] for r in items]); S = X @ Wb[0].T + Wb[1]; o = np.argsort(-S, 1)
    return dict(n=len(items), **{f'R@{k}': float(np.mean([any(tools[j] in gold(r) for j in o[i, :k]) for i, r in enumerate(items)])) for k in (1, 3, 5)})
old_tool_tr = [r for r in old_tr]
cur = tool_fit(old_tool_tr)  # "current data" Qwen3 tool head (old train only), my own reproduction for comparison
RES['tool']['current_on_indep839'] = tool_eval(cur, IND); print('TOOL current-data head on indep839', RES['tool']['current_on_indep839'], flush=True)
RES['tool']['curve'] = []
for F in FAMS:
    test = [r for r in XF if r['fam'] == F]; others = [r for r in XF if r['fam'] != F]
    a = tool_eval(cur, test); row = dict(current=a)
    for fr in FRACS:
        sub = per_label_split(others, fr, 1)[0] if fr < 1 else others
        bq = tool_eval(tool_fit(old_tool_tr + sub), test); row[f'xf_{fr}'] = bq
        RES['tool']['curve'].append(dict(heldout=F, frac=fr, n_xf_train=len(sub), **{f'R@{k}': bq[f'R@{k}'] for k in (1,3,5)}, n=bq['n']))
        print(f'TOOL LOFO {F} frac={fr:.2f} (+{len(sub):4d}): {bq}', flush=True)
    row['xf'] = row[f'xf_{FRACS[-1]}']; RES['tool'][F] = row; print(f'TOOL LOFO {F}: current {a} | xf {row["xf"]}', flush=True)
n = sum(RES['tool'][F]['xf']['n'] for F in FAMS)
for key in ['current'] + [f'xf_{fr}' for fr in FRACS]:
    kk = 'current' if key == 'current' else key
    RES['tool'][f'pooled_{key}'] = {m: sum(RES['tool'][F][kk if key=='current' else key][m] * RES['tool'][F][kk if key=='current' else key]['n'] for F in FAMS) / n for m in ('R@1', 'R@3', 'R@5')}
    print(f'TOOL POOLED LOFO {key}', RES['tool'][f'pooled_{key}'], flush=True)
ext = 'heads/tool_head_ovr_qwen3go.json'
if os.path.exists(ext):
    h = json.load(open(ext)); Wx = (np.array(h['W']), np.array(h['b']))
    if h['tools'] == tools:
        RES['tool']['other_worker'] = {F: tool_eval(Wx, [r for r in XF if r['fam'] == F]) for F in FAMS}; RES['tool']['other_worker_indep839'] = tool_eval(Wx, IND)
        print('TOOL other worker head', RES['tool']['other_worker'], RES['tool']['other_worker_indep839'], flush=True)
if EXPORT:
    tr_all, cal_all = per_label_split(XF, 0.7, 0)
    h = Head(old_tr + FIX + tr_all).calibrate(cal_all); di = evaluate(h, IND); do = evaluate(h, old_te)
    print('EXPORT head: train', len(old_tr + FIX + tr_all), 'cal', len(cal_all), '| indep839', fmt(di), '| old339 top1', round(do['top1'], 3), flush=True)
    RES['export'] = dict(indep839=di, old339=do, n_train=len(old_tr + FIX + tr_all), n_cal=len(cal_all))
    json.dump(dict(labels=h.labels, temperature=float(h.T), tau=h.tau, tau_s=h.tau_s, l2norm=True, sensitive=sorted(SENS), embed_role='classification',
                   embedder_model_id='Qwen3 Embedding 0.6b:1024:prompt-v1:qwen3-instruct-v1', label_table_version='intent-defs-50-v1 (data/intent_defs.json, 49 labels incl. unknown)',
                   note=f'MLP on pure-Go Qwen3 vectors (RoleClassification); train = anchor+synth+fix+multi-family xf ({TAG}, 70%); T/tau/tau_s on held-out 30% xf slice; no indep test items',
                   w1=h.w1.T.tolist(), b1=h.b1.tolist(), w2=h.w2.T.tolist(), b2=h.b2.tolist()), open('heads/head_mlp_qwen3go_xf.json', 'w'))
    W, b = tool_fit(old_tool_tr + XF)
    json.dump(dict(tools=tools, W=W.tolist(), b=b.tolist(), embedder_model_id='Qwen3 Embedding 0.6b:1024:prompt-v1:qwen3-instruct-v1', embed_role='classification',
                   note=f'OvR logistic (C=10, balanced) on L2-normalized pure-Go Qwen3 vectors; train = old train + all xf ({TAG}); gold via scripts/tool_gold.py; sigmoid(W x + b)'),
              open('heads/tool_head_ovr_qwen3go_xf.json', 'w'))
json.dump(RES, open(f'out/xf/xf_train_{TAG}.json', 'w'), indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
