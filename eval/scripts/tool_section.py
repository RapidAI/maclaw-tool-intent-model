"""Section 5 summary: tool routing candidates, per-language Recall@k on held-out test split."""
import json, sys, time, os, numpy as np
sys.path.insert(0, 'scripts')
from analyze_tools import ev
from sklearn.linear_model import LogisticRegression
lang = {json.loads(l)['id']: json.loads(l).get('lang') for l in open('data/intent_all.jsonl')}
te = [i for i in ev if ev[i]['split'] == 'test']; tr = [i for i in ev if ev[i]['split'] == 'train']
sub = te[::3]
def rk(top, gold, k): return any(t in gold for t in top[:k])
rows = []
def rep(name, R, ids):
    ids = [i for i in ids if i in R]
    r = {'name': name, 'n': len(ids)}
    for g, sel in [('all', ids)] + [(L, [i for i in ids if lang.get(i) == L]) for L in ('zh', 'mix', 'en')]:
        for k in (1, 5, 20):
            r[f'{g}@{k}'] = float(np.mean([rk(R[i], ev[i]['gold'], k) for i in sel])) if sel else None
        r[f'n_{g}'] = len(sel)
    r['mrr'] = float(np.mean([next((1/(j+1) for j, t in enumerate(R[i]) if t in ev[i]['gold']), 0) for i in ids]))
    rows.append(r)
    print(f"{name:48s} n={r['n']:3d} R@1={r['all@1']:.3f} R@5={r['all@5']:.3f} R@20={r['all@20']:.3f} | zh@1={r['zh@1']:.3f} mix@1={r['mix@1']:.3f} en@1={r['en@1']:.3f} | zh@5={r['zh@5']:.3f} mix@5={r['mix@5']:.3f} en@5={r['en@5']:.3f} MRR={r['mrr']:.3f}")
A = {}; As = {}
for l in open('out/tool_trackA.jsonl'):
    r = json.loads(l); A[r['id']] = [x['n'] for x in r['top']]; As[r['id']] = {x['n']: x['s'] for x in r['top']}
V = {}
for l in open('out/emb_gemma.jsonl'):
    r = json.loads(l); V[r['id']] = np.array(r['vec'], dtype=np.float32)
tools = sorted({d.get('function', d)['name'] for d in json.load(open('data/core_tool_defs.json'))})
X = lambda ids: np.stack([V[i] / np.linalg.norm(V[i]) for i in ids])
Xtr, Xte = X(tr), X(te)
W = np.zeros((len(tools), Xtr.shape[1]), np.float32); b = np.full(len(tools), -20.0, np.float32)
t0 = time.time()
for k, t in enumerate(tools):
    y = np.array([t in ev[i]['gold'] for i in tr])
    if y.sum() == 0: continue
    c = LogisticRegression(C=10, max_iter=300, class_weight='balanced').fit(Xtr, y)
    W[k] = c.coef_[0]; b[k] = c.intercept_[0]
print('tool head train s %.1f, tools with positives %d/%d' % (time.time() - t0, int((b > -20).sum()), len(tools)))
lat = []
for n in range(len(te)):
    t = time.perf_counter(); z = W @ Xte[n] + b; lat.append((time.perf_counter() - t) * 1000)
P = 1 / (1 + np.exp(-(Xte @ W.T + b)))
H = {i: [tools[k] for k in np.argsort(-P[n])] for n, i in enumerate(te)}
F = {}
for n, i in enumerate(te):
    cand = A[i][:20]; F[i] = sorted(cand, key=lambda t: -(0.3 * As[i][t] + 0.7 * P[n][tools.index(t)]))
np.savez('heads/tool_head_ovr_gemma.npz', W=W, b=b, tools=np.array(tools))
json.dump({'tools': tools, 'W': W.tolist(), 'b': b.tolist(), 'note': 'OvR logistic on L2-normalized EmbeddingGemma vectors; sigmoid(W x + b)'}, open('heads/tool_head_ovr_gemma.json', 'w'))
print('== full held-out test')
rep('A: BM25+Gemma (Router.Route fusion)', A, te)
rep('A top20 -> bge-reranker-base', json.load(open('out/rerank_bge-reranker-base.json')), te)
rep('tool head (Gemma OvR logreg) alone', H, te)
rep('A top20 -> tool head (0.3A+0.7H)', F, te)
print('== 1/3 subset (same as jina / bge-m3 runs)')
rep('[sub] A', A, sub)
rep('[sub] A top20 -> bge-reranker-base', json.load(open('out/rerank_bge-reranker-base.json')), sub)
rep('[sub] A top20 -> jina-reranker-v2-base-multilingual', json.load(open('out/rerank_jina-reranker-v2-base-multilingual.json')), sub)
if os.path.exists('out/rerank_bge-reranker-v2-m3.json'):
    rep('[sub] A top20 -> bge-reranker-v2-m3', json.load(open('out/rerank_bge-reranker-v2-m3.json')), sub)
rep('[sub] tool head alone', H, sub)
rep('[sub] A top20 -> tool head', F, sub)
print('tool head matvec ms (numpy) p50 %.4f p95 %.4f p99 %.4f' % tuple(np.percentile(lat, [50, 95, 99])))
# OOD-ish check: tool head on clean-anchor-only training
anc = {json.loads(l)['id'] for l in open('data/intent_all.jsonl') if json.loads(l).get('src') == 'anchor'}
tra = [i for i in tr if i in anc]; Xa = X(tra); P2 = np.zeros((len(te), len(tools)))
for k, t in enumerate(tools):
    y = np.array([t in ev[i]['gold'] for i in tra])
    if y.sum() == 0 or y.all(): P2[:, k] = 0; continue
    P2[:, k] = LogisticRegression(C=10, max_iter=300, class_weight='balanced').fit(Xa, y).predict_proba(Xte)[:, 1]
H2 = {i: [tools[k] for k in np.argsort(-P2[n])] for n, i in enumerate(te)}
rep('tool head, anchor-only train (clean control)', H2, te)
json.dump(rows, open('out/tool_section.json', 'w'), indent=1)
