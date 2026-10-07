# FRESH interrupt-relevance TRAINING pairs (round 3). Generators: Llama (llama-3.2-90b) and GLM (glm-5.3-flash) —
# deliberately different families from the eval set's generators (Mistral / gpt-oss), never Gemma / Qwen.
# Cross-check: each item blind-labelled by the OTHER training family (fallback gpt-oss-20b); keep only agreement.
# The 262 eval items (cap3/interrupt.jsonl) are never shown to any model here and near-duplicates are removed.
import sys, os, json, re, random, threading, concurrent.futures as cf
sys.path.insert(0,'/workspace/maclaw_reranker/scripts'); import llmc
D='/workspace/maclaw_reranker/out/replace_audit/int3'
GENS=['nvidia:meta/llama-3.2-90b-vision-instruct','nvidia:z-ai/glm-5.3-flash','mistral:mistral-small-latest']
JUDGE={GENS[0]:['nvidia:openai/gpt-oss-20b',GENS[1]],GENS[1]:[GENS[0],'nvidia:openai/gpt-oss-20b'],GENS[2]:[GENS[0],GENS[1]]}
PERSONAS=['后端工程师','运维','产品经理','学生','财务','设计师','数据分析师','普通家庭用户','创业者','测试工程师','律师助理','老师','销售','HR','研究生','自媒体运营']
LANGS=['zh','zh','en','mix','mix']
DOMAINS=['coding','documents/office','data analysis','devops/servers','scheduling/reminders','web research','messaging/IM','files','design/images','finance/reports','learning/notes','home/life']
STY=['very short IM-style replies (3-8 words)','longer messages with context','typos / abbreviations','polite indirect phrasing','pronoun references ("that one", "它") without repeating the task words']
SYS=("You create TRAINING data for an AI agent's interrupt handler. While the agent is busy executing a task, the user sends another message. Return ONLY JSON {\"items\":[...]}. "
"Each item: {\"task\":\"the task as the user originally asked it (one or two sentences)\",\"message\":\"the new user message sent mid-task\",\"label\":\"related\"|\"unrelated\"}. "
"'related' = it supplements, corrects, constrains, asks about the progress/result of, or modifies the running task; 'unrelated' = a separate new request or a question about something else. "
"About half related and half unrelated. Include hard cases: related with no shared words; unrelated but in the same domain or reusing words from the task. Message 2-30 words.")
PFX=re.compile(r'^\s*(zh|en|mix)\s*[:：]\s*',re.I)
def clean(t): return PFX.sub('',t.strip()).strip()
def J(c): m=re.search(r'(\{.*\}|\[.*\])',c,re.S); return json.loads(m.group(1))
lk=threading.Lock()
def gen(i):
    G=[GENS[int(x)] for x in os.environ.get('INT_GENS','0,1,2').split(',')]
    g=G[i%len(G)]; r=random.Random(1000+i); lang=r.choice(LANGS)
    u=(f"Persona: {r.choice(PERSONAS)}. Language: {lang} (zh=Chinese, en=English, mix=Chinese with English technical terms). "
       f"Write 8 items with 8 different tasks, mostly from: {', '.join(r.sample(DOMAINS,4))}. Style emphasis: {r.choice(STY)}. "
       "Some tasks may share a task with 2 different messages (one related, one unrelated).")
    try:
        c,_,_=llmc.chat(g,[{'role':'system','content':SYS},{'role':'user','content':u}],temperature=0.95,max_tokens=3000,timeout=300,retries=1)
        items=J(c)['items']
    except Exception as e: print('ERR',g,str(e)[:100],file=sys.stderr); return
    rows=[{'id':f'intx-{i:04d}-{k}','gen':g,'lang':lang,'task':clean(it['task']),'message':clean(it['message']),'label':it['label']}
          for k,it in enumerate(items) if isinstance(it,dict) and it.get('label') in ('related','unrelated') and it.get('task') and it.get('message')]
    with lk:
        with open(f'{D}/train_raw.jsonl','a') as f:
            for x in rows: f.write(json.dumps(x,ensure_ascii=False)+'\n')
JSYS=("You label data for an AI agent interrupt handler. The agent is executing TASK; the user sends MESSAGE mid-task. 'related' = the message supplements, corrects, constrains, asks about the progress/result of, or modifies TASK; 'unrelated' = it is a separate new request or question. Return ONLY JSON {\"labels\":[\"related\"|\"unrelated\", ...]} in item order.")
def judge():
    rows=[json.loads(l) for l in open(f'{D}/train_raw.jsonl')]
    ev=[json.loads(l) for l in open('/workspace/maclaw_reranker/out/replace_audit/cap3/interrupt.jsonl')]
    norm=lambda s: re.sub(r'\W+','',s.lower())
    evm={norm(e['message']) for e in ev}; evt={norm(e['task']) for e in ev}
    seen=set(); rr=[]
    for r in rows:
        k=(norm(r['task']),norm(r['message']))
        if k in seen or norm(r['message']) in evm or norm(r['task']) in evt: continue
        seen.add(k); rr.append(r)
    B=[]
    for g in GENS:
        rs=[r for r in rr if r['gen']==g]; B+=[rs[i:i+10] for i in range(0,len(rs),10)]
    def do(b):
        u='\n\n'.join(f"Item {k+1}\nTASK: {r['task']}\nMESSAGE: {r['message']}" for k,r in enumerate(b))
        for spec in JUDGE[b[0]['gen']]:
            try:
                c,_,_=llmc.chat(spec,[{'role':'system','content':JSYS},{'role':'user','content':u}],temperature=0,max_tokens=1500,timeout=300,retries=1)
                L=J(c)['labels']
                if len(L)==len(b): return [dict(r,judge=spec,judge_label=l) for r,l in zip(b,L)]
            except Exception as e: print('judge err',spec,str(e)[:80],file=sys.stderr)
        return []
    out=[]
    with cf.ThreadPoolExecutor(12) as ex:
        for o in ex.map(do,B): out+=o
    keep=[r for r in out if r['judge_label']==r['label']]
    with open(f'{D}/train.jsonl','w') as f:
        for r in keep: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    from collections import Counter
    print('raw',len(rows),'dedup',len(rr),'judged',len(out),'agreed',len(keep),Counter(r['label'] for r in keep),Counter(r['gen'] for r in keep))
if __name__=='__main__':
    if sys.argv[1]=='gen':
        with cf.ThreadPoolExecutor(int(os.environ.get("INT_CONC","8"))) as ex: list(ex.map(gen, range(int(sys.argv[3]) if len(sys.argv)>3 else 0, int(sys.argv[2]))))
    else: judge()
