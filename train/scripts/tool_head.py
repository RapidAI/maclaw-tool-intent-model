"""Tool-level multi-label head on frozen EmbeddingGemma (one-vs-rest logistic), and fusion with Track A.
Caveat: gold tools are derived from intent labels (scripts/tool_gold.py), so this partly re-learns intent->tool."""
import json, sys, numpy as np
sys.path.insert(0, 'scripts')
from analyze_tools import report, ev
from sklearn.linear_model import LogisticRegression
V = {}
for l in open('out/emb_gemma.jsonl'):
    r = json.loads(l); V[r['id']] = np.array(r['vec'], dtype=np.float32)
tools = sorted({t for d in json.load(open('data/core_tool_defs.json')) for t in [d.get('function', d)['name']]})
tr = [i for i in ev if ev[i]['split'] == 'train']; te = [i for i in ev if ev[i]['split'] == 'test']
X = lambda ids: np.stack([V[i] / np.linalg.norm(V[i]) for i in ids])
Xtr, Xte = X(tr), X(te)
P = np.zeros((len(te), len(tools)))
for k, t in enumerate(tools):
    y = np.array([t in ev[i]['gold'] for i in tr])
    if y.sum() == 0: continue
    c = LogisticRegression(C=10, max_iter=300, class_weight='balanced').fit(Xtr, y); P[:, k] = c.predict_proba(Xte)[:, 1]
H = {i: [tools[k] for k in np.argsort(-P[n])] for n, i in enumerate(te)}
report('tool head (Gemma, OvR logreg) alone', H)
A = {}
for l in open('out/tool_trackA.jsonl'):
    r = json.loads(l); A[r['id']] = {x['n']: x['s'] for x in r['top']}
F = {}
for n, i in enumerate(te):
    cand = list(A[i].keys())  # Track A top-20 only (head reranks the candidate set)
    sc = {t: 0.3 * A[i][t] + 0.7 * P[n][tools.index(t)] for t in cand}
    F[i] = sorted(cand, key=lambda t: -sc[t])
report('TrackA top20 -> tool head rerank (0.3A+0.7H)', F)
# same 1/3 subset as rerank_eval.py <model> 3
sub = te[::3]
report('[sub/3] TrackA', {i: list(A[i].keys()) for i in sub})
report('[sub/3] tool head alone', {i: H[i] for i in sub})
report('[sub/3] TrackA top20 -> tool head', {i: F[i] for i in sub})
