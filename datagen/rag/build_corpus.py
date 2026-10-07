# Build a RAG passage corpus from maclaw repo docs (read-only), split by markdown headings.
import os, re, json, glob, hashlib, random
R='/workspace/maclaw-replace'
files=[f'{R}/{x}' for x in ['UserManual_CN.md','UserManual_EN.md','faq.md','faq_en.md','README.md','README_EN.md','maclaw_memory_and_tool_scheduling.md','PRODUCT.md','hub_manual.md']]
files+=sorted(f for f in glob.glob(f'{R}/docs/*.md') if 'DEAD-CODE' not in f and 'LEAN-CODE' not in f)
out=[]
def emit(src, title, body):
    body=re.sub(r'\n{3,}','\n\n',body).strip()
    if len(body)<150: return
    # drop code-heavy chunks
    if body.count('```')>=2 and len(re.sub(r'```.*?```','',body,flags=re.S))<150: return
    txt=(title+'\n'+body)[:900]
    out.append({'src':os.path.relpath(src,R),'title':title,'text':txt})
for f in files:
    s=open(f,encoding='utf-8',errors='replace').read()
    parts=re.split(r'(?m)^(#{1,4} .*)$',s)
    cur_title=os.path.basename(f); buf=''
    for p in parts:
        if re.match(r'^#{1,4} ',p):
            if buf.strip(): emit(f,cur_title,buf)
            cur_title=p.strip('# ').strip(); buf=''
        else: buf+=p
    if buf.strip(): emit(f,cur_title,buf)
# dedup exact
seen=set(); ded=[]
for x in out:
    h=hashlib.md5(x['text'].encode()).hexdigest()
    if h in seen: continue
    seen.add(h); ded.append(x)
random.seed(7); random.shuffle(ded)
ded=ded[:900]
for i,x in enumerate(ded): x['id']=f'p{i:04d}'; x['zh']=len(re.findall(r'[\u4e00-\u9fff]',x['text']))/max(1,len(x['text']))
json.dump(ded,open('/workspace/maclaw_reranker/out/replace_audit/rag/corpus.json','w'),ensure_ascii=False)
print(len(out),len(ded),'files',len(files),'zh-dominant',sum(x['zh']>0.2 for x in ded))
