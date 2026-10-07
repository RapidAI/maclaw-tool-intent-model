# r4 C: extra task-side and doc-side variants for coding-subagent retrieval (Qwen3 pure-Go vecdump).
import json
C='/workspace/maclaw_reranker/out/replace_audit/cap3'
TASK={
 'raw':None,
 'WQ':"Given a web search query, retrieve relevant passages that answer the query",
 'CQ':"Given a coding task, retrieve skills or tools that would help a coding agent complete it",
 'CT2':"Given a software development task, retrieve the developer tools, libraries or frameworks needed to complete it",
 'CT3':"Given a programming task description, retrieve descriptions of the tools a developer would use for it",
 'CT4':"Given a coding request, retrieve the most relevant developer tool or library documentation",
}
DOCI={'DT':"Represent this developer tool description for retrieving the coding tasks it helps with"}
def fmt(inst,t): return t if inst is None else f"Instruct: {inst}\nQuery: {t}"
def docs(c):
    n,d=c['name'],c['description']
    return {'raw':n+' '+d,'colon':f"{n}: {d}",'desc':d,'tool':f"Tool: {n}\nDescription: {d}",
            'name2':f"{n} {n} {d}", 'DT':fmt(DOCI['DT'],n+' '+d)}
if __name__=='__main__':
    cod=[json.loads(l) for l in open(f'{C}/coding.jsonl')]; cands=json.load(open(f'{C}/coding_cands.json'))
    rows=[];seen=set()
    def add(k,t):
        if k not in seen: seen.add(k); rows.append({'k':k,'t':t})
    for r in cod:
        for v,i in TASK.items(): add(f"cod:{v}|{r['id']}",fmt(i,r['task']))
    for c in cands:
        for v,t in docs(c).items(): add(f"doc:{v}|{c['name']}",t)
    with open('texts_c.jsonl','w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(len(rows))
