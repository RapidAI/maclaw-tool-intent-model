# Lever 2b sim: anchors mined from TRAIN errors (rule in STATUS_r4.md session 6). Run in ws root with LEV_PKLS=<pkl> python knn_rescue.py members T tau tau_s l2_train.jsonl l2_indep.jsonl train_in.jsonl out.json
import os, sys, json, numpy as np
sys.argv, args = sys.argv[:1], sys.argv[1:]
os.environ.setdefault('LEV_OUT','/tmp/le2_dummy.json')
exec(open('out/replace_audit/r4A/levers_eval2.py').read().split("keys=sorted(")[0])
mem=[(m.split(':')[0],int(m.split(':')[1])) for m in args[0].split(',')]; Tm,tau,ts=map(float,args[1:4])
L2={}
for f in args[4:6]:
    for l in open(f): r=json.loads(l); L2[r['id']]=r
t2id={json.loads(l)['text']:json.loads(l)['id'] for l in open(args[6])}
def heads(kind,R):
    P=T.softmax(Zs(mem,kind)/Tm); c=P.max(1); p=[LAB[j] for j in P.argmax(1)]
    a=c>=np.where(np.array([x in SENS for x in p]),max(tau,ts),tau); return p,a
# fold id per train record
recs=[]; 
p,a=heads('lofo',R_oof)
for r,pp,aa in zip(R_oof,p,a): recs.append((r,pp,aa,'x'+r['fam']))
p,a=heads('cv',R_cv); fo=sum([[k]*len(f) for k,f in enumerate(OF)],[])
for r,pp,aa,k in zip(R_cv,p,a,fo): recs.append((r,pp,aa,'o%d'%k))
pool=[(r,f) for r,pp,aa,f in recs if (not aa or not ok(r['intent'],pp)) and r['intent']!='unknown' and r['intent'] not in SENS and r['id'] in V]
PV=np.stack([V[r['id']] for r,_ in pool]); PL=[r['intent'] for r,_ in pool]; PF=np.array([f for _,f in pool])
print('pool',len(pool))
pi,pa=heads('final',IND)
def evalset(items):  # items: (id,gold,head_pred,head_acc,fold or None)
    out=[]
    for i,g,hp,ha,f in items:
        r=L2[i]; S=PV@V[i]
        if f is not None: S=np.where(PF==f,-1,S)
        j=int(S.argmax()); out.append((g,hp,ha,r['l2_confident'],r['l2_primary'],float(S[j]),PL[j]))
    return out
tr=[(t2id[r['text']],r['intent'],pp,aa,f) for r,pp,aa,f in recs if r['text'] in t2id and t2id[r['text']] in L2 and r['id'] in V]
# map train_in ids to vectors: train_in ids are 'train-XXXX'; use record id for vector
V.update({t2id[r['text']]:V[r['id']] for r,pp,aa,f in recs if r['text'] in t2id and r['id'] in V})
ind=[(r['id'],r['intent'],hp,ha,None) for r,hp,ha in zip(IND,pi,pa)]
K150s={json.loads(l)['id'] for l in open('data/go_in_GK_test_150.jsonl')}
E={'train':evalset(tr),'indep':evalset(ind)}; E['K150']=[e for e,x in zip(E['indep'],ind) if x[0] in K150s]
def st(E,rule):
    corr=acc=resc=rc=fe=0
    for g,hp,ha,l2c,l2p,cs,al in E:
        if l2c: p='L2',l2p
        elif ha: p='L3',hp
        elif rule and cs>=rule[0] and al not in SENS and (not rule[1] or hp==al): p='R',al
        else: p='none','unknown'
        c=ok(g,p[1]); corr+=c; a=p[1]!='unknown'; acc+=a
        if p[0]=='R': resc+=1; rc+=c
        if p[1] in SENS and p[1]!=g: fe+=1
    n=len(E); return dict(n=n,primary=corr/n,acc=acc,rescued=resc,resc_prec=rc/resc if resc else 1.0,sensFE=fe)
grid=[None]+[(c,ag) for c in (.80,.85,.88,.90,.92,.94,.96) for ag in (False,True)]
rows=[(g,st(E['train'],g)) for g in grid]; b=rows[0][1]
for g,s in rows: print(g,s)
basesel=None
def sel(s,E): 
    c=s['primary']*s['n']; return c/s['acc'] if s['acc'] else 0
bs=sel(b,E['train'])
el=[(g,s) for g,s in rows[1:] if s['resc_prec']>=0.976 and sel(s,E['train'])>=bs-0.002]
best=max(el,key=lambda x:(round(x[1]['primary'],6),x[0][0],x[0][1])) if el else (None,b)
print('TRAIN-SELECTED',best)
for k in ('indep','K150'): print(k,'base',st(E[k],None),'| rule',st(E[k],best[0]))
json.dump(dict(rows=rows,selected=best,indep=st(E['indep'],best[0]),K150=st(E['K150'],best[0])),open(args[7],'w'),indent=1,default=str)
