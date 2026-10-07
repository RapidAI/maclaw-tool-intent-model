import json, sys, time, numpy as np, torch
from sentence_transformers import SentenceTransformer
torch.set_num_threads(4)
name, out = sys.argv[1], sys.argv[2]
ids, texts = [], []
for l in open('data/intent_texts.jsonl'):
    r = json.loads(l); ids.append(r['id']); texts.append(r['text'])
m = SentenceTransformer(name, device='cpu')
kw = {}
if 'Qwen3-Embedding' in name:
    kw['prompt'] = "Instruct: Classify the intent of this assistant user request\nQuery: "
t = time.time()
V = m.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False, **kw)
el = time.time() - t
# single-query latency
lat = []
for s in texts[:50]:
    t0 = time.time(); m.encode([s], normalize_embeddings=True, **kw); lat.append((time.time() - t0) * 1000)
np.save(out, V.astype(np.float32)); json.dump(ids, open(out + '.ids.json', 'w'))
print(name, V.shape, f"batch total {el:.1f}s", "single p50 %.1fms p95 %.1fms" % (np.percentile(lat, 50), np.percentile(lat, 95)))
