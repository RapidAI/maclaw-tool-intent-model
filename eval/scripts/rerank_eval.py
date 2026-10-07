import json, sys, time, numpy as np, torch
sys.path.insert(0, 'scripts')
from analyze_tools import report, ev
torch.set_num_threads(4)
defs = json.load(open('data/core_tool_defs.json'))
desc = {}
for d in defs:
    f = d.get('function', d); desc[f['name']] = f['name'] + ": " + (f.get('description') or '')[:600]
A = {json.loads(l)['id']: [x['n'] for x in json.loads(l)['top']] for l in open('out/tool_trackA.jsonl')}
test_ids = [i for i in A if ev[i]["split"] == "test"][::int(sys.argv[2]) if len(sys.argv) > 2 else 1]
name = sys.argv[1]
from sentence_transformers import CrossEncoder
kw = dict(trust_remote_code=True) if 'jina' in name else {}
m = CrossEncoder(name, device='cpu', max_length=384, **kw)
R, lat = {}, []
for i in test_ids:
    cands = A[i][:20]
    t = time.time()
    s = m.predict([(ev[i]['text'], desc[c]) for c in cands], batch_size=20, show_progress_bar=False)
    lat.append((time.time() - t) * 1000)
    R[i] = [c for _, c in sorted(zip(-np.asarray(s), cands))]
json.dump(R, open(f"out/rerank_{name.split('/')[-1]}.json", 'w'))
report(f"TrackA top20 -> {name.split('/')[-1]}", R)
print("rerank latency/query(20 pairs) ms p50 %.0f p95 %.0f p99 %.0f" % tuple(np.percentile(lat, [50, 95, 99])))
