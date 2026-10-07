# Box side: train_xf3.jsonl (judge-agreed EXT+QW+DS) -> aug.jsonl (aug=True) + aug.npy/aug.ids.json (pure-Go Qwen3 classification vectors from emb_xf3.jsonl)
import json, numpy as np, collections
X='/workspace/maclaw_reranker/out/replace_audit/r4A/xf3'
K=[json.loads(l) for l in open(f'{X}/train_xf3.jsonl')]; V={}
for l in open(f'{X}/emb_xf3.jsonl'): r=json.loads(l); V[r['id']]=r['vec']
K=[dict(r,aug=True,split='train') for r in K if r['id'] in V]
open(f'{X}/aug.jsonl','w').writelines(json.dumps(r,ensure_ascii=False)+'\n' for r in K)
np.save(f'{X}/aug.npy',np.array([V[r['id']] for r in K],np.float32)); json.dump([r['id'] for r in K],open(f'{X}/aug.ids.json','w'))
print(len(K),collections.Counter((r['fam'],r['src']) for r in K))
