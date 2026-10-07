# Round-4 FRESH confirmation sets (frozen before any r4 tuning; evaluated once at the end).
#  A intent (old-339 style: ~7/label over 49 labels + ~13% boundary 'hard' items, zh>mix>en):
#      generator NVIDIA Nemotron-3-Super (never used for intent data), blind judge gpt-oss-20b (indep_judge.py prompt), keep label agreement.
#  B interrupt: generator Nemotron-3-Super (never used for interrupt), judge gpt-oss-20b; same prompt as cap3/gen.py.
#  C coding: generator Nemotron-3-Super (never used for coding), judge gpt-oss-20b; same catalogue + prompts as cap3; gold = gen ∩ judge.
#  (A judge also gpt-oss-20b.) Gemini (quota) / Cohere (trial quota) / DeepSeek, Yi, Jamba, Phi, Granite (NIM 404/timeouts) unavailable 11:53.
# All deduped (normalised text) against every existing intent/interrupt/coding dataset on the box.
import sys, os, json, re, random, threading, concurrent.futures as cf, time
R='/workspace/maclaw_reranker'; sys.path.insert(0,R+'/scripts'); os.chdir(R); import llmc
import indep_gen_defs as G
D=R+'/out/replace_audit/confirm_r4'; CAP=R+'/out/replace_audit/cap3'
NEMO='nvidia:nvidia/nemotron-3-super-120b-a12b'; GEM='nvidia:openai/gpt-oss-20b'  # judge (Gemini free quota exhausted 11:53; Cohere trial quota exhausted)
lk=threading.Lock()
def J(c): m=re.search(r'(\{.*\}|\[.*\])',c,re.S); return json.loads(m.group(1))
def chat(spec,sys_,user,temp,mt=4000):
    for a in range(4):
        try:
            c,_,_=llmc.chat(spec,[{'role':'system','content':sys_},{'role':'user','content':user}],temperature=temp,max_tokens=mt,timeout=240,retries=2); return J(c)
        except Exception as e:
            print('ERR',spec,str(e)[:100],file=sys.stderr,flush=True); time.sleep(20*(a+1))
    return None
def app(f,rows):
    with lk, open(f'{D}/{f}','a') as fo:
        for r in rows: fo.write(json.dumps(r,ensure_ascii=False)+'\n')
# ---------------- A ----------------
LABELS=sorted(l for l in G.D if l!='ambiguous')+(['unknown'] if 'unknown' not in G.D else [])
def genA(t):
    kind,key,i=t
    if kind=='label': items=(chat(NEMO,G._SYS,G.task_label(key,8),1.0) or {}).get('items',[]); lab=key
    else: a,b=key.split('|'); items=(chat(NEMO,G._SYS,G.task_pair(a,b,3),1.0) or {}).get('items',[]); lab=a
    app('A_raw.jsonl',[{'id':f'cfA-{i:03d}-{k}','intent':lab,'src':'hard' if kind=='hard' else 'heldout','lang':it.get('lang'),'text':it['text'].strip(),'gen':NEMO}
                       for k,it in enumerate(items) if isinstance(it,dict) and isinstance(it.get('text'),str) and it['text'].strip()])
def judgeA():
    src=open(R+'/scripts/indep_judge.py').read(); g={'json':json}
    exec(src[src.index('D = json.load'):src.index('C = [json.loads')],g); SYS=g['SYS']
    rows=dedup([json.loads(l) for l in open(f'{D}/A_raw.jsonl')],'text',existing_texts())
    B=[rows[i:i+15] for i in range(0,len(rows),15)]
    def do(b):
        r=chat(GEM,SYS,"Messages:\n"+"\n".join(f"{k}. {c['text']}" for k,c in enumerate(b)),0)
        out=[]
        for x in (r or {}).get('results',[]):
            try: k=int(x['i'])
            except Exception: continue
            if 0<=k<len(b): out.append(dict(b[k],judge=GEM,judge_label=x.get('label')))
        return out
    allr=[]
    with cf.ThreadPoolExecutor(3) as ex:
        for o in ex.map(do,B): allr+=o
    keep=[r for r in allr if r['judge_label']==r['intent']]
    write_frozen('A',keep); print('A raw',len(rows),'judged',len(allr),'agreed',len(keep))
# ---------------- B / C (cap3 prompts) ----------------
sys.path.insert(0,CAP); import gen as cap
PERS=cap.PERSONAS; LANGS=cap.LANGS
def genB(i):
    lang=LANGS[i%4]; p=PERS[i%len(PERS)]
    r=chat(NEMO,cap.INT_SYS,f"Persona: {p}. Language: {lang} (zh=Chinese, en=English, mix=Chinese with English technical terms). Write 8 items with 8 different tasks (coding, documents, data, devops, scheduling, research, messaging, files).",0.9)
    app('B_raw.jsonl',[{'id':f'cfB-{i:03d}-{k}','gen':NEMO,'lang':lang,'task':it['task'],'message':it['message'],'label':it['label']} for k,it in enumerate((r or {}).get('items',[])) if isinstance(it,dict) and it.get('label') in ('related','unrelated') and it.get('task') and it.get('message')])
CANDS=json.load(open(f'{CAP}/coding_cands.json'))
def genC(i):
    lang=LANGS[i%4]; cat='\n'.join(f"- {c['name']}: {c['description']}" for c in CANDS)
    SYS=("You create test data for selecting skills for a coding sub-agent. Given the catalogue, write coding task descriptions a developer would hand to the sub-agent, and for each list the catalogue names that are genuinely useful for it (1-3 names; never more). Return ONLY JSON {\"items\":[{\"task\":\"...\",\"relevant\":[\"name\",...]}]}. "
         "Do not copy catalogue wording into the task; describe the work naturally. Include some tasks where only one niche skill applies.")
    r=chat(NEMO,SYS,f"Catalogue:\n{cat}\n\nLanguage: {lang} (zh=Chinese, en=English, mix=Chinese with English technical terms). Write 8 tasks.",0.9)
    names={c['name'] for c in CANDS}
    app('C_raw.jsonl',[{'id':f'cfC-{i:03d}-{k}','gen':NEMO,'lang':lang,'task':it['task'],'relevant':[n for n in it['relevant'] if n in names]} for k,it in enumerate((r or {}).get('items',[])) if isinstance(it,dict) and it.get('task') and isinstance(it.get('relevant'),list)])
def judgeBC(kind):
    rows=[json.loads(l) for l in open(f'{D}/{kind}_raw.jsonl')]
    if kind=='B': rows=dedup(rows,'message',existing_int('message')); judge=GEM
    else: rows=[r for r in rows if r['relevant']]; rows=dedup(rows,'task',existing_cod()); judge=GEM
    B=[rows[i:i+10] for i in range(0,len(rows),10)]
    cat='\n'.join(f"- {c['name']}: {c['description']}" for c in CANDS)
    def do(b):
        if kind=='B':
            s=("You label data for an AI agent interrupt handler. The agent is executing TASK; the user sends MESSAGE mid-task. 'related' = the message supplements, corrects, constrains or modifies TASK; 'unrelated' = it is a separate new request or question. Return ONLY JSON {\"labels\":[\"related\"|\"unrelated\", ...]} in item order.")
            u='\n\n'.join(f"Item {k+1}\nTASK: {r['task']}\nMESSAGE: {r['message']}" for k,r in enumerate(b))
        else:
            s=("You select skills for a coding sub-agent. For each task, list the catalogue names that are genuinely useful (1-3 names, most useful first). Return ONLY JSON {\"labels\":[[\"name\",...], ...]} in task order.")
            u=f"Catalogue:\n{cat}\n\n"+'\n'.join(f"Task {k+1}: {r['task']}" for k,r in enumerate(b))
        res=chat(judge,s,u,0,3000)
        L=(res or {}).get('labels',[])
        return [dict(r,judge=judge,judge_label=l) for r,l in zip(b,L)] if len(L)==len(b) else []
    allr=[]
    with cf.ThreadPoolExecutor(3) as ex:
        for o in ex.map(do,B): allr+=o
    keep=[]
    for r in allr:
        if kind=='C':
            jl=[n for n in r['judge_label'] if isinstance(n,str)] if isinstance(r['judge_label'],list) else []
            g=[n for n in r['relevant'] if n in jl]
            if g: r['gold']=g; keep.append(r)
        elif r['judge_label']==r['label']: keep.append(r)
    write_frozen(kind,keep); print(kind,'raw',len(rows),'judged',len(allr),'agreed',len(keep))
# ---------------- dedup / freeze ----------------
N=lambda s: re.sub(r'\W+','',str(s).lower())
def existing_texts():
    S=set()
    for f in ['data/intent_all.jsonl','data/indep_test.jsonl','data/indep_ambiguous.jsonl','data/indep_candidates.jsonl']:
        if os.path.exists(f): S|={N(json.loads(l).get('text','')) for l in open(f)}
    import glob
    for f in glob.glob('data/train_xfamily/*.jsonl'):
        for l in open(f):
            try: r=json.loads(l)
            except Exception: continue
            if 'text' in r: S.add(N(r['text']))
            for it in r.get('items',[]) if isinstance(r.get('items'),list) else []:
                if isinstance(it,dict) and 'text' in it: S.add(N(it['text']))
    return S
def existing_int(k):
    S=set()
    for f in [f'{CAP}/interrupt.jsonl',f'{CAP}/interrupt_raw.jsonl',R+'/out/replace_audit/int3/train_raw.jsonl']:
        S|={N(json.loads(l)[k]) for l in open(f)}
    return S
def existing_cod():
    return {N(json.loads(l)['task']) for f in (f'{CAP}/coding.jsonl',f'{CAP}/coding_raw.jsonl') for l in open(f)}
def dedup(rows,k,S):
    seen=set(); out=[]
    for r in rows:
        x=N(r[k])
        if x in S or x in seen or not x: continue
        seen.add(x); out.append(r)
    return out
def write_frozen(kind,rows):
    p=f'{D}/{kind}_confirm.jsonl'
    with open(p,'w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
if __name__=='__main__':
    m=sys.argv[1]
    if m=='genA':
        T=[('label',l,i) for i,l in enumerate(LABELS)]+[('hard',f'{a}|{b}',100+i) for i,(a,b) in enumerate(G.PAIRS)]
        with cf.ThreadPoolExecutor(6) as ex: list(ex.map(genA,T))
    elif m=='genB':
        with cf.ThreadPoolExecutor(int(os.environ.get("CONC","3"))) as ex: list(ex.map(genB,range(int(sys.argv[3]) if len(sys.argv)>3 else 0,int(sys.argv[2]))))
    elif m=='genC':
        with cf.ThreadPoolExecutor(6) as ex: list(ex.map(genC,range(int(sys.argv[2]))))
    elif m=='judgeA': judgeA()
    else: judgeBC(m[-1])
