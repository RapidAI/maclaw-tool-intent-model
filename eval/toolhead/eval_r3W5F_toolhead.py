# Tool head compatibility + clean evaluation (Qwen3 RoleClassification query vectors, same math as corelib/tool/toolhead.go:
# sigmoid(W·x̂+b), rank by prob). Sets: old262 (tool_eval test) and indep644 (tool-mappable independent set).
# xf heads are trained on old_train + xf only (no old262 / indep) -> clean on both. The currently shipped tool_head_ovr_qwen3go.json
# was trained on old_train + old_test 262 + indep 644 + fix (its own 'train' field) -> NOT clean on either; shown for reference only.
import json, sys, numpy as np
sys.path.insert(0,'scripts'); from tool_gold import GOLD
ev=[json.loads(l) for l in open('data/tool_eval.jsonl')]; old_te=[r for r in ev if r['split']=='test']
NEW=[json.loads(l) for l in open('data/indep_test.jsonl')]
for r in NEW: r['gold']=GOLD.get(r['intent'])
ind=[r for r in NEW if r['gold']]
V={}
for l in open('out/qwen3_toolhead/qvec_q3cls_final.jsonl'):
    r=json.loads(l); v=np.array(r['vec'],np.float32); V[r['id']]=v/np.linalg.norm(v)
def ev_head(path, items):
    h=json.load(open(path)); W=np.array(h['W'],np.float32); b=np.array(h['b'],np.float32); tools=h['tools']
    X=np.stack([V[r['id']] for r in items]); P=1/(1+np.exp(-(X@W.T+b)))
    out={}
    for k in (1,3,5):
        out[f'R@{k}']=float(np.mean([any(tools[j] in r['gold'] for j in np.argsort(-p,kind='stable')[:k]) for p,r in zip(P,items)]))
    out['MRR']=float(np.mean([next(1/(i+1) for i,j in enumerate(np.argsort(-p,kind='stable')) if tools[j] in r['gold']) for p,r in zip(P,items)]))
    return out, P, tools
res={}
for name in ['tool_head_ovr_qwen3go_xf_r2bF.json','tool_head_ovr_qwen3go_xf_r3W5F.json']:
    for sname,items in (('old262',old_te),('indep644',ind)):
        items=[r for r in items if r['id'] in V]
        m,_,_=ev_head('heads/'+name,items); res[(name,sname)]=m
        print(f"{name:38s} {sname:8s} n={len(items)} "+' '.join(f"{k}={v:.3f}" for k,v in m.items()))
json.dump({f'{a}|{b}':v for (a,b),v in res.items()},open('out/replace_audit/toolhead/eval_r3W5F_toolhead.json','w'),indent=1)
