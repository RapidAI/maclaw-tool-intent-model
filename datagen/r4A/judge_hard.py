# Cross-family judge for r4 hard-pair TRAINING data. Judge=Nemotron (gen was Llama).
# Keep only label agreement. Output: hard_train.jsonl (for head retrain; NOT confirm_r4).
import sys,os,json,re,concurrent.futures as cf,time
from collections import Counter
R='/workspace/maclaw_reranker'; sys.path.insert(0,R+'/scripts'); os.chdir(R)
import llmc
D=R+'/out/replace_audit/r4A'
JUDGE='nvidia:nvidia/nemotron-3-super-120b-a12b'
src=open(R+'/scripts/indep_judge.py').read(); g={'json':json}
exec(src[src.index('D = json.load'):src.index('C = [json.loads')],g); SYS=g['SYS']
rows=[json.loads(l) for l in open(f'{D}/hard_raw.jsonl')]
# dedupe vs all intent texts
N=lambda s: re.sub(r'\W+','',str(s).lower())
exist=set()
for f in ['data/intent_all.jsonl','data/indep_test.jsonl','data/fix_anchors.jsonl','data/train_xfamily/train_xf.jsonl']:
    if not os.path.exists(f): continue
    for l in open(f):
        try: r=json.loads(l)
        except Exception: continue
        if 'text' in r: exist.add(N(r['text']))
        for it in r.get('items',[]) if isinstance(r.get('items'),list) else []:
            if isinstance(it,dict) and 'text' in it: exist.add(N(it['text']))
seen=set(); clean=[]
for r in rows:
    x=N(r['text'])
    if x and x not in exist and x not in seen: seen.add(x); clean.append(r)
print(f'dedup {len(rows)}->{len(clean)}',flush=True)
B=[clean[i:i+12] for i in range(0,len(clean),12)]
def do(b):
    u="Messages:\n"+"\n".join(f"{k}. {c['text']}" for k,c in enumerate(b))
    for a in range(4):
        try:
            c,_,_=llmc.chat(JUDGE,[{'role':'system','content':SYS},{'role':'user','content':u}],temperature=0,max_tokens=3000,timeout=240,retries=1)
            m=re.search(r'(\{.*\})',c,re.S); res=json.loads(m.group(1))['results']
            out=[]
            for x in res:
                try: k=int(x['i'])
                except Exception: continue
                if 0<=k<len(b): out.append(dict(b[k],judge=JUDGE,judge_label=x.get('label')))
            return out
        except Exception as e:
            print('ERR',str(e)[:100],flush=True); time.sleep(15*(a+1))
    return []
allr=[]
with cf.ThreadPoolExecutor(4) as ex:
    for o in ex.map(do,B): allr+=o
keep=[r for r in allr if r['judge_label']==r['intent']]
with open(f'{D}/hard_train.jsonl','w') as f:
    for r in keep: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(f'judged {len(allr)} agreed {len(keep)} {Counter(r["intent"] for r in keep).most_common(10)}',flush=True)
