import json, sys, numpy as np
ev = {json.loads(l)['id']: json.loads(l) for l in open('data/tool_eval.jsonl')}
def rk(top, gold, k): return any(t in gold for t in top[:k])
def report(name, ranks, split='test'):
    ids = [i for i in ranks if ev[i]['split'] == split]
    out = {k: np.mean([rk(ranks[i], ev[i]['gold'], k) for i in ids]) for k in (1, 3, 5, 10, 20)}
    mrr = np.mean([next((1 / (j + 1) for j, t in enumerate(ranks[i]) if t in ev[i]['gold']), 0) for i in ids])
    print(f"{name:42s} n={len(ids)} " + " ".join(f"R@{k}={v:.3f}" for k, v in out.items()) + f" MRR={mrr:.3f}")
    return out
if __name__ == '__main__':
    A = {json.loads(l)['id']: [x['n'] for x in json.loads(l)['top']] for l in open('out/tool_trackA.jsonl')}
    report('TrackA BM25+Gemma (train split)', A, 'train'); report('TrackA BM25+Gemma (test split)', A)
    ms = [json.loads(l)['ms'] for l in open('out/tool_trackA.jsonl')]
    print('trackA ms p50 %.1f p95 %.1f p99 %.1f' % tuple(np.percentile(ms, [50, 95, 99])))
