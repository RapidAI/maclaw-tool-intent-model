import sys,os,json,re,concurrent.futures as cf,time
from collections import Counter
R='/workspace/maclaw_reranker'; sys.path.insert(0,R+'/scripts'); os.chdir(R); import llmc
D=R+'/out/replace_audit/confirm_r4'
JUDGE='nvidia:meta/llama-3.2-90b-vision-instruct'
SYS=("You label data for an AI agent interrupt handler. The agent is executing TASK; the user sends MESSAGE mid-task. "
     "'related' = the message supplements, corrects, constrains or modifies TASK; 'unrelated' = it is a separate new request or question. "
     "Return ONLY JSON {\"labels\":[\"related\"|\"unrelated\", ...]} in item order.")
def parse_labels(c,n):
    m=re.search(r'(\{.*\}|\[.*\])',c,re.S)
    if not m: raise ValueError('no json')
    o=json.loads(m.group(1))
    if isinstance(o,dict): L=o.get('labels') or o.get('label')
    else: L=o
    if not isinstance(L,list): raise ValueError('not list')
    L=L[:n]
    if len(L)!=n: raise ValueError(f'bad labels len={len(L)}')
    return [str(x).lower().strip() for x in L]
N=lambda s: re.sub(r'\W+','',str(s).lower())
rows=[json.loads(l) for l in open(f'{D}/B_raw.jsonl')]
exist=set()
for f in [R+'/out/replace_audit/cap3/interrupt.jsonl',R+'/out/replace_audit/cap3/interrupt_raw.jsonl',R+'/out/replace_audit/int3/train_raw.jsonl']:
    for l in open(f): exist.add(N(json.loads(l)['message']))
seen=set(); clean=[]
for r in rows:
    x=N(r['message'])
    if not x or x in exist or x in seen: continue
    seen.add(x); clean.append(r)
print(f'dedup {len(rows)}->{len(clean)}',flush=True)
B=[clean[i:i+5] for i in range(0,len(clean),5)]
def do(b):
    u='\n\n'.join(f"Item {k+1}\nTASK: {r['task']}\nMESSAGE: {r['message']}" for k,r in enumerate(b))
    for a in range(5):
        try:
            c,_,_=llmc.chat(JUDGE,[{'role':'system','content':SYS},{'role':'user','content':u}],temperature=0,max_tokens=1200,timeout=180,retries=1)
            L=parse_labels(c,len(b))
            return [dict(r,judge=JUDGE,judge_label=l) for r,l in zip(b,L) if l in ('related','unrelated')]
        except Exception as e:
            print('ERR',str(e)[:120],flush=True); time.sleep(8*(a+1))
    return []
allr=[]
with cf.ThreadPoolExecutor(3) as ex:
    for o in ex.map(do,B): allr+=o
keep=[r for r in allr if r['judge_label']==r['label']]
with open(f'{D}/B_confirm.jsonl','w') as f:
    for r in keep: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(f'B raw {len(clean)} judged {len(allr)} agreed {len(keep)} {Counter(r["label"] for r in keep)}',flush=True)
