# Merge judgments: local Qwen judge (EXT only) first, then NIM (judge_raw3.jsonl) overrides. Agreement rule as xf_judge.py. -> train_xf3_final.jsonl
import json, collections
X='/workspace/maclaw_reranker/out/replace_audit/r4A/xf3'
SAFEU=set(json.load(open('/workspace/maclaw_reranker/data/intent_meta.json'))['safe_unknown']); J={}; src={}
import os
if os.path.exists(f'{X}/judge_raw_qwenjudge.jsonl'):
    for l in open(f'{X}/judge_raw_qwenjudge.jsonl'):
        d=json.loads(l)
        for r in d['results']: J[r['id']]=r; src[r['id']]='qwen'
for l in open(f'{X}/judge_raw3.jsonl'):
    d=json.loads(l)
    for r in d['results']: J[r['id']]=r; src[r['id']]=d['judge'].split('/')[-1]
C=[json.loads(l) for l in open(f'{X}/candidates3.jsonl')]; st=collections.Counter(); keep=[]
for c in C:
    if c['id'] not in J: st[f'unjudged_{c["fam"]}']+=1; continue
    if c['fam']=='qwen' and src[c['id']]=='qwen': raise SystemExit('qwen item judged by qwen')
    j=J[c['id']]; c['judge']=j['label']; c['judge_second']=j['second']; c['judge_model']=src[c['id']]
    if j['label']==c['intent'] or (c['intent']=='unknown' and j['label'] in SAFEU): keep.append(c); st[f'agree_{c["fam"]}']+=1
    else: st[f'disagree_{c["fam"]}']+=1
open(f'{X}/train_xf3.jsonl','w').writelines(json.dumps(c,ensure_ascii=False)+'\n' for c in keep)
print(dict(sorted(st.items())),'kept',len(keep)); print(collections.Counter((c['fam'],c['judge_model']) for c in keep))
json.dump(dict(st),open(f'{X}/merge3_stats.json','w'),indent=1)
