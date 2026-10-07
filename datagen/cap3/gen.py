# Labelled eval sets for topic-switch / IM-interrupt relevance / coding-subagent scoring.
# Generators: Mistral (ministral-8b) and OpenAI (gpt-oss-20b on NIM); labels cross-checked by the OTHER family
# (fallback judge: Cohere Command-A). No Gemma / Qwen family anywhere.
import sys, json, random, re, threading, concurrent.futures as cf
sys.path.insert(0,'/workspace/maclaw_reranker/scripts'); import llmc
D='/workspace/maclaw_reranker/out/replace_audit/cap3'
GENS=['mistral:ministral-8b-latest','nvidia:openai/gpt-oss-20b']
JUDGE={'mistral:ministral-8b-latest':['nvidia:openai/gpt-oss-20b','cohere:command-a-03-2025'],
       'nvidia:openai/gpt-oss-20b':['mistral:ministral-8b-latest','cohere:command-a-03-2025']}
PERSONAS=['后端工程师','运维','产品经理','学生','财务','设计师','数据分析师','普通家庭用户','创业者','测试工程师','律师助理','老师']
LANGS=['zh','zh','en','mix']
def J(c):
    m=re.search(r'(\{.*\}|\[.*\])',c,re.S); return json.loads(m.group(1))
def call(spec,sys_,user,temp=0.9):
    c,m,dt=llmc.chat(spec,[{'role':'system','content':sys_},{'role':'user','content':user}],temperature=temp,max_tokens=3000,timeout=120,retries=1)
    return J(c)
lk=threading.Lock()
def out(f,rows):
    with lk:
        with open(f'{D}/{f}','a') as fo:
            for r in rows: fo.write(json.dumps(r,ensure_ascii=False)+'\n')
# ---------- topic ----------
TOP_SYS=("You create test data for a chat assistant's topic-switch detector. Return ONLY JSON {\"items\":[...]}. Each item: "
"{\"history\":[4 short user messages on ONE topic, in order],\"assistant_last\":\"one-sentence assistant reply to the last user message\",\"new\":\"the next user message\",\"label\":\"same\"|\"new\"}. "
"'same' = the new message continues/follows up the same topic (may use different words, pronouns, short refinements); 'new' = the user switches to an unrelated task/topic (may still share some generic words). "
"Make about half same and half new, include hard cases (same topic with no shared keywords; new topic that reuses a word from the history). New message at least 5 words.")
TOPIC_EXTRA=''
TOPIC_TAG='topic'
def topic(i):
    g=GENS[i%2]; lang=LANGS[i%4]; p=PERSONAS[(i*5)%len(PERSONAS)]
    try: items=call(g,TOP_SYS,f"Persona: {p}. Language: {lang} (zh=Chinese, en=English, mix=Chinese with English technical terms). Write 6 items on 6 different everyday or work topics.{TOPIC_EXTRA}")['items']
    except Exception as e: print('ERR topic',g,str(e)[:100],file=sys.stderr); return
    rows=[]
    for k,it in enumerate(items):
        if it.get('label') in ('same','new') and isinstance(it.get('history'),list) and len(it['history'])>=3 and it.get('new'):
            rows.append({'id':f'{TOPIC_TAG}-{i:03d}-{k}','gen':g,'lang':lang,**{x:it[x] for x in ('history','assistant_last','new','label')}})
    out('topic_raw.jsonl',rows)
# ---------- interrupt ----------
INT_SYS=("You create test data for an AI agent's interrupt handler. While the agent is busy executing a task, the user sends another message. Return ONLY JSON {\"items\":[...]}. "
"Each item: {\"task\":\"the task description the agent is executing (one or two sentences, as the user originally asked)\",\"message\":\"the new user message sent mid-task\",\"label\":\"related\"|\"unrelated\"}. "
"'related' = supplements, corrects, constrains or modifies the running task (e.g. 'use red instead', 'also add tests', 'don't touch the config'); 'unrelated' = a separate new request or question about something else. "
"Half related, half unrelated; include hard cases (related but no shared words; unrelated but same domain words). Message 3-25 words.")
def interrupt(i):
    g=GENS[i%2]; lang=LANGS[i%4]; p=PERSONAS[i%len(PERSONAS)]
    try: items=call(g,INT_SYS,f"Persona: {p}. Language: {lang} (zh=Chinese, en=English, mix=Chinese with English technical terms). Write 8 items with 8 different tasks (coding, documents, data, devops, scheduling, research, messaging, files).")['items']
    except Exception as e: print('ERR int',g,str(e)[:100],file=sys.stderr); return
    rows=[{'id':f'int-{i:03d}-{k}','gen':g,'lang':lang,'task':it['task'],'message':it['message'],'label':it['label']} for k,it in enumerate(items) if it.get('label') in ('related','unrelated') and it.get('task') and it.get('message')]
    out('interrupt_raw.jsonl',rows)
# ---------- coding subagent ----------
CAND_SYS=("You write a catalogue of installable skills / MCP tools for a coding sub-agent (like Claude Code skills). Return ONLY JSON {\"cands\":[{\"name\":\"kebab-case-name\",\"description\":\"one or two sentence description of what it does\"}]}. "
"Write 45 distinct candidates covering: testing frameworks, linters/formatters, git/GitHub/PR, CI, docker/k8s, databases/migrations, cloud deploy, frontend build, mobile, data/notebooks, docs generation, security scanning, profiling, API clients, i18n, logging, package publishing, plus a few non-coding ones (spreadsheet export, image editing, email). Some descriptions in Chinese.")
def cands():
    c=call('mistral:ministral-8b-latest',CAND_SYS,'Write the catalogue.',temp=0.7)['cands']
    seen=set(); res=[]
    for x in c:
        if x.get('name') and x['name'] not in seen: seen.add(x['name']); res.append({'name':x['name'],'description':x['description']})
    json.dump(res,open(f'{D}/coding_cands.json','w'),ensure_ascii=False,indent=1); return res
def coding(i,C):
    g=GENS[i%2]; lang=LANGS[i%4]
    cat='\n'.join(f"- {c['name']}: {c['description']}" for c in C)
    SYS=("You create test data for selecting skills for a coding sub-agent. Given the catalogue, write coding task descriptions a developer would hand to the sub-agent, and for each list the catalogue names that are genuinely useful for it (1-3 names; never more). Return ONLY JSON {\"items\":[{\"task\":\"...\",\"relevant\":[\"name\",...]}]}. "
         "Do not copy catalogue wording into the task; describe the work naturally. Include some tasks where only one niche skill applies.")
    try: items=call(g,SYS,f"Catalogue:\n{cat}\n\nLanguage: {lang} (zh=Chinese, en=English, mix=Chinese with English technical terms). Write 8 tasks.")['items']
    except Exception as e: print('ERR cod',g,str(e)[:100],file=sys.stderr); return
    names={c['name'] for c in C}
    rows=[{'id':f'cod-{i:03d}-{k}','gen':g,'lang':lang,'task':it['task'],'relevant':[n for n in it['relevant'] if n in names]} for k,it in enumerate(items) if it.get('task') and isinstance(it.get('relevant'),list)]
    out('coding_raw.jsonl',[r for r in rows if r['relevant']])
if __name__=='__main__':
    what=sys.argv[1]; n=int(sys.argv[2])
    if what=='coding':
        C=cands() if len(sys.argv)<4 else json.load(open(f'{D}/coding_cands.json'))
        with cf.ThreadPoolExecutor(8) as ex: list(ex.map(lambda i: coding(i,C), range(n)))
    elif what=='topic2':
        TOPIC_EXTRA=' This batch: exactly 4 items labelled same (natural follow-ups: refinements, next steps, pronoun references, questions about the assistant reply) and 2 labelled new.'
        TOPIC_TAG='topic2'
        with cf.ThreadPoolExecutor(8) as ex: list(ex.map(topic, range(n)))
    else:
        fn={'topic':topic,'interrupt':interrupt}[what]
        with cf.ThreadPoolExecutor(8) as ex: list(ex.map(fn, range(n)))
