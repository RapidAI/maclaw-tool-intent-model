# Coding-subagent: pick the Qwen3 task/doc prompt on the TRAIN half only (calibrated F1 of the production fusion),
# recalibrate baseline on train, report TEST vs Gemma (production embedding, train-calibrated baseline) with paired bootstrap CI.
import json, hashlib, random, sys, numpy as np
sys.path.insert(0,'.'); from vec import load
C='/workspace/maclaw_reranker/out/replace_audit/cap3'
V=load('vec_eval')
cands=json.load(open(f'{C}/coding_cands.json')); names=[c['name'] for c in cands]; docs=[c['name']+' '+c['description'] for c in cands]
split=lambda i: 'train' if int(hashlib.md5(i.encode()).hexdigest(),16)%2==0 else 'test'
task={json.loads(l)['id']:json.loads(l)['task'] for l in open(f'{C}/coding.jsonl')}
Q={r['id']:r for r in map(json.loads,open(f'{C}/sig_coding_qwen3.jsonl'))}
G={}
for suf in ('','_qd','_sim'):
    G[suf]={r['id']:r for r in map(json.loads,open(f'{C}/sig_coding_gemma{suf}.jsonl'))}
ids=sorted(Q); tr=[i for i in ids if split(i)=='train']; te=[i for i in ids if split(i)=='test']
def rows_for(tv,dv):
    D=np.stack([V[f'doc:{dv}|{d}'] for d in docs]); D/=np.linalg.norm(D,axis=1,keepdims=True)
    out={}
    for i in ids:
        q=V[f'cod:{tv}|{task[i]}']; q=q/np.linalg.norm(q)
        r=dict(Q[i]); r['cos']=list((D@q).astype(float)); out[i]=r
    return out
def select(r,base,thr=0.15,k=5):
    sc=[(max(r['bm25'][j],r['bigram'][j],max(0.0,r['cos'][j]-base)),j) for j in range(len(names))]
    sc=[x for x in sc if x[0]>=thr]; sc.sort(key=lambda x:-x[0]); return [names[j] for _,j in sc[:k]]
def F1(rows,base):
    tp=fp=fn=h1=0
    for r in rows:
        g=set(r['gold']); s=select(r,base); tp+=len(g&set(s)); fp+=len(set(s)-g); fn+=len(g-set(s)); h1+=bool(s and s[0] in g)
    p=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn); return (2*p*rc/(p+rc) if p+rc else 0), h1/len(rows)
def MRR(rows):
    m=0
    for r in rows:
        o=sorted(range(len(names)),key=lambda j:-r['cos'][j]); g=set(r['gold'])
        m+=1/(min(k for k,j in enumerate(o) if names[j] in g)+1)
    return m/len(rows)
def calib(R):
    return max((F1([R[i] for i in tr],b/100)[0],b/100) for b in range(0,80,2))
lines=[]; P=lambda *a: (lines.append(' '.join(map(str,a))), print(*a))
P('== Qwen3 prompt variants, TRAIN half only (n=%d): calibrated F1 / baseline / MRR'%len(tr))
best=None
for tv in ('raw','CQ','WQ','SIM'):
    for dv in ('raw','SIM'):
        R=rows_for(tv,dv); f,b=calib(R); m=MRR([R[i] for i in tr])
        P(f'  task={tv:4s} doc={dv:4s}  F1={f:.3f} base={b:.2f} MRR={m:.3f}')
        if best is None or (f,m)>best[0]: best=((f,m),tv,dv,b,R)
(_,tv,dv,bq,RQ)=best
P(f'SELECTED (train): task={tv} doc={dv} baseline={bq:.2f}')
P('== Gemma variants, TRAIN half: calibrated F1 / MRR')
gbest=None
for suf,R in G.items():
    f,b=calib(R); m=MRR([R[i] for i in tr]); P(f'  gemma{suf or "_prod"} F1={f:.3f} base={b:.2f} MRR={m:.3f}')
    if suf=='' : gprod=(R,b)
    if gbest is None or (f,m)>gbest[0]: gbest=((f,m),suf,R,b)
def report(Rq,bq,Rg,bg,tag):
    fq,hq=F1([Rq[i] for i in te],bq); fg,hg=F1([Rg[i] for i in te],bg); mq=MRR([Rq[i] for i in te]); mg=MRR([Rg[i] for i in te])
    rnd=random.Random(7); dF=[];dM=[]
    for _ in range(1000):
        s=[te[rnd.randrange(len(te))] for _ in te]
        dF.append(F1([Rq[i] for i in s],bq)[0]-F1([Rg[i] for i in s],bg)[0]); dM.append(MRR([Rq[i] for i in s])-MRR([Rg[i] for i in s]))
    dF.sort(); dM.sort()
    P(f'TEST ({tag}, n={len(te)}): Qwen3 F1={fq:.3f} hit@1={hq:.3f} MRR={mq:.3f} | Gemma F1={fg:.3f} hit@1={hg:.3f} MRR={mg:.3f} | ΔF1={fq-fg:+.3f} [{dF[25]:+.3f},{dF[975]:+.3f}] ΔMRR={mq-mg:+.3f} [{dM[25]:+.3f},{dM[975]:+.3f}]')
report(RQ,bq,gprod[0],gprod[1],'vs Gemma production, both train-calibrated')
report(RQ,bq,gbest[2],gbest[3],f'vs Gemma best-on-train variant gemma{gbest[1] or "_prod"}')
open('coding_r3.txt','w').write('\n'.join(lines)+'\n')
