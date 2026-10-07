import json, sys, numpy as np, torch
from sentence_transformers import SentenceTransformer
torch.set_num_threads(4)
rows = [json.loads(l) for l in open(sys.argv[1])]
m = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', device='cpu')
V = m.encode([r['text'] for r in rows], batch_size=32, normalize_embeddings=True, show_progress_bar=False,
             prompt="Instruct: Classify the intent of this assistant user request\nQuery: ")
np.save('out/emb_indep_qwen3e.npy', V.astype(np.float32)); json.dump([r['id'] for r in rows], open('out/emb_indep_qwen3e.npy.ids.json', 'w'))
print(V.shape)
