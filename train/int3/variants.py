# Instruction variants (Qwen3 "Instruct: {task}\nQuery: {text}" format; raw = production plain Embed, no prompt).
import json
INT={
 'raw': None,
 'IT':  "Given a task an AI agent is currently executing, retrieve follow-up user messages that supplement, correct, constrain or modify this task",
 'IM':  "Given a message a user sent while an AI agent was executing a task, retrieve the task this message refers to",
 'IS':  "Represent the user request by the task it concerns, so that a follow-up message and the task it modifies are close",
 'SIM': "Retrieve semantically similar text",
 'CLS': "Classify the intent of this assistant user request",
}
COD={
 'raw': None,
 'CQ':  "Given a coding task, retrieve skills or tools that would help a coding agent complete it",
 'WQ':  "Given a web search query, retrieve relevant passages that answer the query",
 'SIM': "Retrieve semantically similar text",
}
def fmt(inst,t): return t if inst is None else f"Instruct: {inst}\nQuery: {t}"
def ctx(task,msg): return f"Current task: {task}\nFollow-up message: {msg}"
def build(pairs, out, coding=None, cands=None):
    seen=set(); rows=[]
    def add(k,t):
        if k not in seen: seen.add(k); rows.append({'k':k,'t':t})
    for p in pairs:
        for v,inst in INT.items():
            add(f"{v}|{p['task']}",fmt(inst,p['task'])); add(f"{v}|{p['message']}",fmt(inst,p['message']))
        add(f"CTX|{p['task']}|{p['message']}",ctx(p['task'],p['message']))
    for r in coding or []:
        for v,inst in COD.items(): add(f"cod:{v}|{r['task']}",fmt(inst,r['task']))
    for c in cands or []:
        d=c['name']+' '+c['description']
        for v in ('raw','SIM'): add(f"doc:{v}|{d}",fmt(COD[v],d))
    with open(out,'w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    return len(rows)
if __name__=='__main__':
    import sys
    C='/workspace/maclaw_reranker/out/replace_audit/cap3'
    if sys.argv[1]=='eval':
        P=[json.loads(l) for l in open(f'{C}/interrupt.jsonl')]
        cod=[json.loads(l) for l in open(f'{C}/coding.jsonl')]; cands=json.load(open(f'{C}/coding_cands.json'))
        print(build(P,'texts_eval.jsonl',cod,cands))
    else:
        P=[json.loads(l) for l in open('train.jsonl')]; print(build(P,'texts_train.jsonl'))
