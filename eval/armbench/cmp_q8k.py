# Section 12: Q8P kernel output vs current-branch (local/qwen3-speed, prefix KV on) output.
import json, sys, numpy as np
a=[json.loads(l) for l in open(sys.argv[1])]; b=[json.loads(l) for l in open(sys.argv[2])]
assert len(a)==len(b)
cos=[]; tok=0; worst=[]
for x,y in zip(a,b):
    assert x['id']==y['id']
    tok += x.get('tokens')==y.get('tokens')
    u=np.array(x['vec']); v=np.array(y['vec'])
    c=float(u@v/np.linalg.norm(u)/np.linalg.norm(v)); cos.append(c); worst.append((c,x['id'],len(x.get('tokens') or [])))
cos=np.array(cos); worst.sort()
print(f"n={len(cos)} tok_equal={tok}/{len(cos)} cos mean={cos.mean():.6f} min={cos.min():.6f} p01={np.percentile(cos,1):.6f} p05={np.percentile(cos,5):.6f} median={np.median(cos):.6f} n<0.999={int((cos<0.999).sum())} n<0.9995={int((cos<0.9995).sum())}")
print("worst5", [(round(c,6),i,n) for c,i,n in worst[:5]])
