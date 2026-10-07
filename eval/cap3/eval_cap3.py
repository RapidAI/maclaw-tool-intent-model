# Evaluate topic-switch / interrupt relevance / coding-subagent scoring for Qwen3 vs Gemma
# using the raw signals dumped by corelib/agent/zz_cap3_audit_test.go (production code paths).
import json, hashlib, random, math, sys
D='/workspace/maclaw_reranker/out/replace_audit/cap3'
M=['qwen3','gemma']
import os
# optional role-variant suffixes (secondary analysis): SUF_QWEN3=_sim SUF_GEMMA=_qd OUT_TAG=_v1
SUF={m:os.environ.get('SUF_'+m.upper(),'') for m in M}; OUT_TAG=os.environ.get('OUT_TAG','')
def L(f): return [json.loads(l) for l in open(f'{D}/{f}')]
def split(i): return 'train' if int(hashlib.md5(i.encode()).hexdigest(),16)%2==0 else 'test'
def auc(pos,neg):
    s=0
    for p in pos:
        for n in neg: s+= 1 if p>n else 0.5 if p==n else 0
    return s/(len(pos)*len(neg))
def boot(fn, ids, B=2000, seed=7):
    # paired bootstrap of fn(sample)->(qwen-gemma); returns 95% CI
    r=random.Random(seed); n=len(ids); v=[]
    for _ in range(B):
        smp=[ids[r.randrange(n)] for _ in range(n)]
        v.append(fn(smp))
    v.sort(); return v[int(.025*B)], v[int(.975*B)]
out={}; lines=[]
P=lambda *a: lines.append(' '.join(str(x) for x in a))
# ---------------- topic ----------------
T={m:{r['id']:r for r in L(f'sig_topic_{m}{SUF[m]}.jsonl')} for m in M}
ids=sorted(T['qwen3']); tr=[i for i in ids if split(i)=='train']; te=[i for i in ids if split(i)=='test']
def tvote(r, cs=0.45, cn=0.25, bs=1.0, bn=0.3):
    if r['words']<4 or r['user_turns']<3: return 'same','guard'
    b='same' if r['bm25']>=bs else 'new' if r['bm25']<=bn else 'unsure'
    c='same' if r['cos']>=cs else 'new' if r['cos']<=cn else 'unsure'
    if 'same' in (b,c): return 'same','-'
    if b=='new' and c=='new': return 'new','-'
    if 'new' in (b,c): return 'same','llm'   # production: LLM tiebreak (nil client → same)
    return 'same','-'
def tmetrics(rows, **kw):
    n=len(rows); dec=[(r['label'],)+tvote(r,**kw) for r in rows]
    acc=sum(l==d for l,d,_ in dec)/n
    new=[x for x in dec if x[0]=='new']; same=[x for x in dec if x[0]=='same']
    return dict(acc=acc, switch_recall=sum(d=='new' for _,d,_ in new)/len(new), false_switch=sum(d=='new' for _,d,_ in same)/len(same),
                bal=(sum(d=='new' for _,d,_ in new)/len(new)+sum(d=='same' for _,d,_ in same)/len(same))/2, llm_escalation=sum(e=='llm' for *_,e in dec)/n)
P('## Topic-switch detection  (n=%d: train %d / test %d; test new=%d same=%d)'%(len(ids),len(tr),len(te),sum(T['qwen3'][i]['label']=='new' for i in te),sum(T['qwen3'][i]['label']=='same' for i in te)))
res={}
for m in M:
    rows=[T[m][i] for i in te]
    pos=[r['cos'] for r in rows if r['label']=='same']; neg=[r['cos'] for r in rows if r['label']=='new']
    prod=tmetrics(rows)
    # calibrate cosine thresholds on train: maximise balanced accuracy s.t. false-switch <= 5%
    best=None; trows=[T[m][i] for i in tr]
    for cn in [x/100 for x in range(-10,80)]:
        for cs in [x/100 for x in range(0,95,1)]:
            if cs<=cn: continue
            mt=tmetrics(trows,cs=cs,cn=cn)
            if mt['false_switch']<=0.05 and (best is None or mt['bal']>best[0]): best=(mt['bal'],cs,cn)
    cal=tmetrics(rows,cs=best[1],cn=best[2])
    cs_=sorted(r['cos'] for r in rows)
    res[m]=dict(auc_cos=auc(pos,neg),prod=prod,cal=dict(thr=(best[1],best[2]),**cal),cos_med_same=sorted(pos)[len(pos)//2],cos_med_new=sorted(neg)[len(neg)//2])
    P(f"{m:6s} cos-AUC(same>new)={res[m]['auc_cos']:.3f}  median cos same={res[m]['cos_med_same']:.3f} new={res[m]['cos_med_new']:.3f}")
    P(f"       production thr (same>=.45,new<=.25, BM25 1.0/0.3, no-LLM): acc={prod['acc']:.3f} switch-recall={prod['switch_recall']:.3f} false-switch={prod['false_switch']:.3f} bal={prod['bal']:.3f} llm-escalation={prod['llm_escalation']:.3f}")
    P(f"       train-calibrated cos thr same>={best[1]:.2f} new<={best[2]:.2f}: acc={cal['acc']:.3f} switch-recall={cal['switch_recall']:.3f} false-switch={cal['false_switch']:.3f} bal={cal['bal']:.3f} llm-escalation={cal['llm_escalation']:.3f}")
def d_auc(smp):
    a={}
    for m in M:
        pos=[T[m][i]['cos'] for i in smp if T[m][i]['label']=='same']; neg=[T[m][i]['cos'] for i in smp if T[m][i]['label']=='new']
        a[m]=auc(pos,neg) if pos and neg else 0.5
    return a['qwen3']-a['gemma']
lo,hi=boot(d_auc,te,B=500); P(f"       ΔAUC qwen3-gemma = {res['qwen3']['auc_cos']-res['gemma']['auc_cos']:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]")
cq,cg=res['qwen3']['cal']['thr'],res['gemma']['cal']['thr']
def d_bal(smp):
    return tmetrics([T['qwen3'][i] for i in smp],cs=cq[0],cn=cq[1])['bal']-tmetrics([T['gemma'][i] for i in smp],cs=cg[0],cn=cg[1])['bal']
lo,hi=boot(d_bal,te,B=500); P(f"       Δbal-acc (calibrated) = {res['qwen3']['cal']['bal']-res['gemma']['cal']['bal']:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]")
out['topic']=res
# ---------------- interrupt ----------------
I={m:{r['id']:r for r in L(f'sig_interrupt_{m}{SUF[m]}.jsonl')} for m in M}
ids=sorted(I['qwen3']); tr=[i for i in ids if split(i)=='train']; te=[i for i in ids if split(i)=='test']
def bands(rows,hi=0.60,lo=0.30):
    rel=[r for r in rows if r['label']=='related']; un=[r for r in rows if r['label']=='unrelated']
    return dict(rel_high=sum(r['rel']>=hi for r in rel)/len(rel), rel_low=sum(0<=r['rel']<lo for r in rel)/len(rel),
                un_low=sum(0<=r['rel']<lo for r in un)/len(un), un_high=sum(r['rel']>=hi for r in un)/len(un))
P(''); P('## IM interrupt relevance  (n=%d: train %d / test %d; test related=%d unrelated=%d)'%(len(ids),len(tr),len(te),sum(I['qwen3'][i]['label']=='related' for i in te),sum(I['qwen3'][i]['label']=='unrelated' for i in te)))
res={}
for m in M:
    rows=[I[m][i] for i in te]; trows=[I[m][i] for i in tr]
    pos=[r['rel'] for r in rows if r['label']=='related']; neg=[r['rel'] for r in rows if r['label']=='unrelated']
    prod=bands(rows)
    un=sorted(r['rel'] for r in trows if r['label']=='unrelated'); rl=sorted(r['rel'] for r in trows if r['label']=='related')
    hi_t=un[min(len(un)-1,math.ceil(0.95*len(un)))] if un else .6   # ≤5% unrelated at/above
    hi_t=max(hi_t, un[-1]+1e-9) if len(un)<20 else hi_t
    lo_t=rl[int(0.05*len(rl))]                                       # ≤5% related below
    cal=bands(rows,hi=hi_t,lo=min(lo_t,hi_t))
    res[m]=dict(auc=auc(pos,neg),prod=prod,cal=dict(hi=hi_t,lo=min(lo_t,hi_t),**cal),med_rel=sorted(pos)[len(pos)//2],med_un=sorted(neg)[len(neg)//2])
    P(f"{m:6s} AUC(related>unrelated)={res[m]['auc']:.3f}  median cos related={res[m]['med_rel']:.3f} unrelated={res[m]['med_un']:.3f}")
    P(f"       production bands (high>=.60, low<.30): related→high {prod['rel_high']:.3f}, related→low {prod['rel_low']:.3f} (wrong split) | unrelated→low {prod['un_low']:.3f}, unrelated→high {prod['un_high']:.3f} (wrong merge)")
    P(f"       train-calibrated (high>={hi_t:.3f}, low<{min(lo_t,hi_t):.3f}; ≤5% errors on train): related→high {cal['rel_high']:.3f}, related→low {cal['rel_low']:.3f} | unrelated→low {cal['un_low']:.3f}, unrelated→high {cal['un_high']:.3f}")
def d_auc(smp):
    a={}
    for m in M:
        pos=[I[m][i]['rel'] for i in smp if I[m][i]['label']=='related']; neg=[I[m][i]['rel'] for i in smp if I[m][i]['label']=='unrelated']
        a[m]=auc(pos,neg) if pos and neg else .5
    return a['qwen3']-a['gemma']
lo,hi=boot(d_auc,te,B=500); P(f"       ΔAUC qwen3-gemma = {res['qwen3']['auc']-res['gemma']['auc']:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]")
out['interrupt']=res
# ---------------- coding ----------------
C={m:{r['id']:r for r in L(f'sig_coding_{m}{SUF[m]}.jsonl')} for m in M}
cands=[c['name'] for c in json.load(open(f'{D}/coding_cands.json'))]
ids=sorted(C['qwen3']); tr=[i for i in ids if split(i)=='train']; te=[i for i in ids if split(i)=='test']
def select(r, base=0.2, thr=0.15, k=5, use_emb=True):
    sc=[]
    for j in range(len(cands)):
        s=max(r['bm25'][j], r['bigram'][j], max(0.0, r['cos'][j]-base) if use_emb else 0.0)
        if s>=thr: sc.append((s,j))
    sc.sort(key=lambda x:-x[0]); return [cands[j] for _,j in sc[:k]]
def cmetrics(rows, **kw):
    tp=fp=fn=0; h1=0; empty=0
    for r in rows:
        g=set(r['gold']); s=select(r,**kw)
        tp+=len(g&set(s)); fp+=len(set(s)-g); fn+=len(g-set(s)); h1+= (1 if s and s[0] in g else 0); empty+= (0 if s else 1)
    p=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn); f=2*p*rc/(p+rc) if p+rc else 0
    return dict(P=p,R=rc,F1=f,hit1=h1/len(rows),empty=empty/len(rows))
def rank(rows):
    r1=r5=mrr=0
    for r in rows:
        order=sorted(range(len(cands)),key=lambda j:-r['cos'][j]); g=set(r['gold'])
        ranks=[k for k,j in enumerate(order) if cands[j] in g]
        r1+= ranks[0]==0; r5+= ranks[0]<5; mrr+=1/(ranks[0]+1)
    n=len(rows); return dict(R1=r1/n,R5=r5/n,MRR=mrr/n)
P(''); P('## Coding-subagent skill/MCP scoring  (n=%d tasks × %d candidates: train %d / test %d)'%(len(ids),len(cands),len(tr),len(te)))
lex=cmetrics([C['qwen3'][i] for i in te],use_emb=False)
P(f"lexical only (no embedder; BM25+bigram): P={lex['P']:.3f} R={lex['R']:.3f} F1={lex['F1']:.3f} hit@1={lex['hit1']:.3f} empty={lex['empty']:.3f}")
res={}
for m in M:
    rows=[C[m][i] for i in te]; trows=[C[m][i] for i in tr]
    prod=cmetrics(rows); rk=rank(rows)
    best=max(((cmetrics(trows,base=b/100)['F1'],b/100) for b in range(0,70,2)))
    cal=cmetrics(rows,base=best[1])
    res[m]=dict(rank=rk,prod=prod,cal=dict(base=best[1],**cal))
    P(f"{m:6s} embedding-only ranking: R@1={rk['R1']:.3f} R@5={rk['R5']:.3f} MRR={rk['MRR']:.3f}")
    P(f"       production fusion (baseline .20, thr .15, top-5): P={prod['P']:.3f} R={prod['R']:.3f} F1={prod['F1']:.3f} hit@1={prod['hit1']:.3f} empty={prod['empty']:.3f}")
    P(f"       train-calibrated baseline {best[1]:.2f}: P={cal['P']:.3f} R={cal['R']:.3f} F1={cal['F1']:.3f} hit@1={cal['hit1']:.3f} empty={cal['empty']:.3f}")
bq,bg=res['qwen3']['cal']['base'],res['gemma']['cal']['base']
lo,hi=boot(lambda s: cmetrics([C['qwen3'][i] for i in s],base=bq)['F1']-cmetrics([C['gemma'][i] for i in s],base=bg)['F1'],te,B=500)
P(f"       ΔF1 (calibrated) qwen3-gemma = {res['qwen3']['cal']['F1']-res['gemma']['cal']['F1']:+.3f} 95% CI [{lo:+.3f},{hi:+.3f}]")
lo,hi=boot(lambda s: rank([C['qwen3'][i] for i in s])['MRR']-rank([C['gemma'][i] for i in s])['MRR'],te,B=500)
P(f"       ΔMRR (embedding-only) = {res['qwen3']['rank']['MRR']-res['gemma']['rank']['MRR']:+.3f} 95% CI [{lo:+.3f},{hi:+.3f}]")
out['coding']=res
json.dump(out,open(f'{D}/cap3_results{OUT_TAG}.json','w'),indent=1)
open(f'{D}/cap3_eval{OUT_TAG}.txt','w').write('\n'.join(lines)+'\n'); print('\n'.join(lines))
