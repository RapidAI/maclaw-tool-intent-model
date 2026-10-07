# Extra hard-pair TRAINING data for r4 intent head. Generator=Llama (xf already has Llama but we
# keep labels cross-checked by Nemotron). Target pairs taken from indep839 sensitive FE (out-of-family).
# NOT part of confirm_r4. Never touches old-339.
import sys,os,json,re,random,threading,concurrent.futures as cf,time
R='/workspace/maclaw_reranker'; sys.path.insert(0,R+'/scripts'); os.chdir(R)
import llmc, indep_gen_defs as G
D=R+'/out/replace_audit/r4A'
GEN='nvidia:meta/llama-3.2-90b-vision-instruct'; JUD='nvidia:nvidia/nemotron-3-super-120b-a12b'
# Confused pairs (gold → wrong): from indep FE + PAIRS that involve sensitive labels
PAIRS=[('memory_manage','ssh'),('ssh','memory_manage'),
       ('computer_use','screenshot'),('screenshot','computer_use'),
       ('document_delivery','screenshot'),('screenshot','document_delivery'),
       ('browser','business_data'),('business_data','browser'),
       ('file_write','config_manage'),('config_manage','file_write'),
       ('browser','app_launch'),('ssh','shell_command'),('shell_command','ssh'),
       ('delegate_task','audit_read'),('database','business_data'),
       ('git_mutate','git_inspect'),('schedule_dispatch','schedule_manage'),
       ('knowledge_write','memory_manage'),('file_delete','ssh')]
lk=threading.Lock()
def J(c):
    m=re.search(r'(\{.*\}|\[.*\])',c,re.S); return json.loads(m.group(1))
def chat(spec,sys_,user,temp=1.0):
    for a in range(4):
        try:
            c,_,_=llmc.chat(spec,[{'role':'system','content':sys_},{'role':'user','content':user}],temperature=temp,max_tokens=3000,timeout=180,retries=1)
            return J(c)
        except Exception as e:
            print('ERR',spec,str(e)[:100],flush=True); time.sleep(12*(a+1))
    return None
def gen(i):
    a,b=PAIRS[i%len(PAIRS)]; n=4
    r=chat(GEN,G._SYS,G.task_pair(a,b,n),1.0)
    items=(r.get('items') if isinstance(r,dict) else r) or []
    rows=[{'id':f'r4h-{i:03d}-{k}','intent':a,'confuse_with':b,'lang':it.get('lang'),'text':it['text'].strip(),'gen':GEN,'src':'r4hard'}
          for k,it in enumerate(items) if isinstance(it,dict) and it.get('text')]
    with lk, open(f'{D}/hard_raw.jsonl','a') as f:
        for x in rows: f.write(json.dumps(x,ensure_ascii=False)+'\n')
if __name__=='__main__':
    n=int(sys.argv[1]) if len(sys.argv)>1 else 60
    with cf.ThreadPoolExecutor(6) as ex: list(ex.map(gen,range(n)))
    print('done',sum(1 for _ in open(f'{D}/hard_raw.jsonl')),flush=True)
