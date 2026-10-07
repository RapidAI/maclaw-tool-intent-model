# ONE-SHOT C eval (config frozen: task CT3, doc "Tool: name\nDescription: desc", base 0.50, BM25 weight 0.6; Gemma production, train-calibrated base).
import json,hashlib,sys,numpy as np
C='/workspace/maclaw_reranker/out/replace_audit/cap3'
cands=json.load(open(f'{C}/coding_cands.json')); names=[c['name'] for c in cands]
def load(p):
    k=[l.rstrip('\n') for l in open(p+'.keys')]; V=np.fromfile(p+'.f32','<f4').reshape(len(k),-1); return {x:V[i] for i,x in enumerate(k)}
nz=lambda x:x/np.linalg.norm(x,axis=-1,keepdims=True)
VC=load('vec_c'); Dt=nz(np.stack([VC[f'doc:tool|{n}'] for n in names]))
split=lambda i:'train' if int(hashlib.md5(i.encode()).hexdigest(),16)%2==0 else 'test'
def select(r,base,wl,thr=0.15,k=5):
    sc=[(max(wl*r['bm25'][j],r['bigram'][j],max(0.0,r['cos'][j]-base)),j) for j in range(len(names))]
    sc=[x for x in sc if x[0]>=thr]; sc.sort(key=lambda x:-x[0]); return [names[j] for _,j in sc[:k]]
def F1(rows,base,wl):
    tp=fp=fn=0
    for r in rows:
        g=set(r['gold']); s=set(select(r,base,wl)); tp+=len(g&s); fp+=len(s-g); fn+=len(g-s)
    p=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0; return 2*p*rc/(p+rc) if p+rc else 0
def MRR(rows):
    m=0
    for r in rows:
        o=np.argsort(-np.array(r['cos']),kind='stable'); g=set(r['gold']); hits=[k for k,j in enumerate(o) if names[j] in g]; m+=1/(hits[0]+1) if hits else 0
    return m/len(rows)
def report(tag,RQ,RG,bq,bg):
    ids=sorted(RQ); fq,fg=F1([RQ[i] for i in ids],bq,0.6),F1([RG[i] for i in ids],bg,1.0); mq,mg=MRR([RQ[i] for i in ids]),MRR([RG[i] for i in ids])
    rnd=np.random.RandomState(7); dF=[];dM=[]
    for _ in range(2000):
        s=[ids[j] for j in rnd.randint(0,len(ids),len(ids))]
        dF.append(F1([RQ[i] for i in s],bq,0.6)-F1([RG[i] for i in s],bg,1.0)); dM.append(MRR([RQ[i] for i in s])-MRR([RG[i] for i in s]))
    dF=np.sort(dF);dM=np.sort(dM); ci=lambda d:(float(d[50]),float(d[1949]))
    r=dict(n=len(ids),qwen3_F1=fq,gemma_F1=fg,dF1=fq-fg,dF1_ci=ci(dF),qwen3_MRR=mq,gemma_MRR=mg,dMRR=mq-mg,dMRR_ci=ci(dM)); print(tag,json.dumps(r)); return r
out={}
Q={r['id']:r for r in map(json.loads,open(f'{C}/sig_coding_qwen3.jsonl'))}; G={r['id']:r for r in map(json.loads,open(f'{C}/sig_coding_gemma.jsonl'))}
te=[i for i in sorted(Q) if split(i)=='test']; tr=[i for i in sorted(Q) if split(i)=='train']
bg=max((F1([G[i] for i in tr],b/100,1.0),b/100) for b in range(0,80,2))[1]; print('gemma train-calibrated base',bg)
RQ={}
for i in te:
    r=dict(Q[i]); r['cos']=list((Dt@nz(VC[f'cod:CT3|{i}'])).astype(float)); RQ[i]=r
out['test']=report('TEST',RQ,{i:G[i] for i in te},0.50,bg)
if len(sys.argv)>1 and sys.argv[1]=='confirm':
    GC={r['id']:r for r in map(json.loads,open('confirm_dir/sig_coding_gemma.jsonl'))}; VQ=load('vconf_c'); RQc={}
    for i,r0 in GC.items():
        r=dict(r0); r['cos']=list((Dt@nz(VQ[f'cod:CT3|{i}'])).astype(float)); RQc[i]=r
    out['C_confirm']=report('C_CONFIRM',RQc,GC,0.50,bg)
json.dump(out,open('oneshot_c.json','w'),indent=1)
