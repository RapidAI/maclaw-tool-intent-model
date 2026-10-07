import sys,os,json,re,concurrent.futures as cf,time
from collections import Counter
R='/workspace/maclaw_reranker'; sys.path.insert(0,R+'/scripts'); os.chdir(R); import llmc
D=R+'/out/replace_audit/confirm_r4'
JUDGE='nvidia:openai/gpt-oss-20b'
SYS=("You label data for an AI agent interrupt handler. The agent is executing TASK; the user sends MESSAGE mid-task. "
     "'related' = the message supplements, corrects, constrains or modifies TASK; 'unrelated' = it is a separate new request or question. "
     "Return ONLY JSON {\"labels\":[\"related\"|\"unrelated\", ...]} in item order.")
def parse_labels(c,n):
    m=re.search(r'(\{.*\}|\[.*\])',c,re.S)
    if not m: raise ValueError('no json')
    o=json.loads(m.group(1))
    L=(o.get('labels') if isinstance(o,dict) else o) or []
    L=[str(x).lower().strip() for x in L[:n]]
    if len(L)!=n: raise ValueError(f'len {len(L)}')
    return L
N=lambda s: re.sub(r'\W+','',str(s).lower())
rows=[json.loads(l) for l in open(f'{D}/B_raw.jsonl')]
exist=set()
for f in [R+'/out/replace_audit/cap3/interrupt.jsonl',R+'/out/replace_audit/cap3/interrupt_raw.jsonl',R+'/out/replace_audit/int3/train_raw.jsonl']:
    for l in open(f): exist.add(N(json.loads(l)['message']))
seen=set(); clean=[]
for r in rows:
    x=N(r['message'])
    if x and x not in exist and x not in seen: seen.add(x); clean.append(r)
print(f'dedup {len(rows)}->{len(clean)}',flush=True)
# resume: skip already-judged if a partial file exists
done=set()
if os.path.exists(f'{D}/B_partial.jsonl'):
    for l in open(f'{D}/B_partial.jsonl'): done.add(json.loads(l)['id'])
todo=[r for r in clean if r['id'] not in done]
print(f'resume todo {len(todo)} done {len(done)}',flush=True)
B=[todo[i:i+5] for i in range(0,len(todo),5)]
lk=__import__('threading').Lock()
def do(b):
    u='\n\n'.join(f"Item {k+1}\nTASK: {r['task']}\nMESSAGE: {r['message']}" for k,r in enumerate(b))
    for a in range(6):
        try:
            c,_,_=llmc.chat(JUDGE,[{'role':'system','content':SYS},{'role':'user','content':u}],temperature=0,max_tokens=800,timeout=120,retries=1)
            L=parse_labels(c,len(b))
            out=[dict(r,judge=JUDGE,judge_label=l) for r,l in zip(b,L) if l in ('related','unrelated')]
            with lk, open(f'{D}/B_partial.jsonl','a') as f:
                for x in out: f.write(json.dumps(x,ensure_ascii=False)+'\n')
            return out
        except Exception as e:
            print('ERR',str(e)[:100],flush=True); time.sleep(15*(a+1))
    return []
allr=[]
with cf.ThreadPoolExecutor(4) as ex:
    for o in ex.map(do,B): allr+=o
# merge
seen=set(); keep=[]
for l in open(f'{D}/B_partial.jsonl'):
    r=json.loads(l)
    if r['id'] in seen: continue
    seen.add(r['id'])
    if r['judge_label']==r['label']: keep.append(r)
with open(f'{D}/B_confirm.jsonl','w') as f:
    for r in keep: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(f'B agreed {len(keep)} {Counter(r["label"] for r in keep)}',flush=True)
