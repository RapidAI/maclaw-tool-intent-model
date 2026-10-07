import sys, json, random, re, concurrent.futures as cf
sys.path.insert(0,'/workspace/maclaw_reranker/scripts'); import llmc
D='/workspace/maclaw_reranker/out/replace_audit/rag'
corpus=json.load(open(f'{D}/corpus.json'))
N=int(sys.argv[1]) if len(sys.argv)>1 else 220
SPECS=sys.argv[2].split(',') if len(sys.argv)>2 else ['nvidia:meta/llama-3.3-70b-instruct','cohere:command-a-03-2025']
random.seed(11); targets=random.sample(corpus,N)
SYS=("You write realistic search queries for evaluating a retrieval system over a software product's documentation. "
"Given ONE passage, write three different queries that a user of the product would type into the knowledge-base search box and for which THIS passage is the best answer: "
"(1) 'zh': natural Chinese; (2) 'en': natural English; (3) 'mix': Chinese sentence mixing in English technical terms (code-switching, like Chinese developers type). "
"Rules: paraphrase — do NOT copy any run of more than 4 consecutive words / 6 consecutive Chinese characters from the passage; ask about the specific content (not generic); 6-30 words each; no quotes around queries. "
"Return ONLY JSON: {\"zh\":\"...\",\"en\":\"...\",\"mix\":\"...\"}")
def one(p):
    for spec in SPECS:
        try:
            c,m,dt=llmc.chat(spec,[{'role':'system','content':SYS},{'role':'user','content':'Passage:\n'+p['text']}],temperature=0.7,max_tokens=600,timeout=90,retries=1)
            j=json.loads(re.search(r'\{.*\}',c,re.S).group(0))
            if all(isinstance(j.get(k),str) and len(j[k])>3 for k in ('zh','en','mix')):
                return {'pid':p['id'],'gen':m,**{k:j[k].strip() for k in ('zh','en','mix')}}
        except Exception as e:
            print('ERR',spec,p['id'],str(e)[:120],file=sys.stderr)
    return None
res=[]
import threading; lk=threading.Lock(); fo=open(f'{D}/queries_raw.jsonl','w')
def one2(p):
    r=one(p)
    if r:
        with lk: fo.write(json.dumps(r,ensure_ascii=False)+'\n'); fo.flush(); res.append(r)
with cf.ThreadPoolExecutor(10) as ex: list(ex.map(one2,targets))
from collections import Counter
print(len(res),Counter(r['gen'] for r in res))
