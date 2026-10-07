"""r4 session 7: build candidates from (a) unjudged/leftover raw items in data/train_xfamily (READ-ONLY reuse) and (b) new raw files in r4A/xf3/raw_*.jsonl
(qwen / deepseek). Same flatten + NFKC exact dedupe as scripts/xf_build.py, extended to A/B/C confirm texts; near-dup pure-Go Qwen3 cos>0.95 vs
old-339 / indep839 / amb95 / A_confirm. Output r4A/xf3/candidates3.jsonl. Writes nothing outside r4A/xf3."""
import json, re, glob, unicodedata, collections, os, subprocess, numpy as np
R='/workspace/maclaw_reranker'; os.chdir(R); X='out/replace_audit/r4A/xf3'; CF='out/replace_audit/confirm_r4'
L=lambda f:[json.loads(l) for l in open(f)]
rows=L('data/intent_all.jsonl'); IND=L('data/indep_test.jsonl'); AMB=L('data/indep_ambiguous.jsonl'); K150=L('data/go_in_GK_test_150.jsonl')
AC=L(f'{CF}/A_confirm.jsonl'); BC=L(f'{CF}/B_confirm.jsonl'); CC=L(f'{CF}/C_confirm.jsonl')
def norm(s): return re.sub(r'[\W_]+','',unicodedata.normalize('NFKC',s).lower())
ref={}
for r in rows: ref[norm(r['text'])]='train' if r['split']=='train' else 'oldtest'
for r in IND+K150: ref[norm(r['text'])]='indep'
for r in AMB: ref[norm(r['text'])]='amb'
for r in AC: ref[norm(r['text'])]='Aconfirm'
for r in BC: ref[norm(r['message'])]='Bconfirm'
for r in CC: ref[norm(r['task'])]='Cconfirm'
judged=set()
for l in open('data/train_xfamily/judge_raw_cohere.jsonl'):
    for r in json.loads(l)['results']: judged.add(r['id'])
old_c={c['id']:c for c in L('data/train_xfamily/candidates.jsonl')}
seen={norm(c['text']) for i,c in old_c.items() if i in judged}   # texts already judged (agree or disagree) are not re-used
for c in L('data/train_xfamily/train_xf.jsonl'): seen.add(norm(c['text']))
out=[]; st=collections.Counter()
files=sorted(glob.glob('data/train_xfamily/raw_*.jsonl'))+sorted(glob.glob(f'{X}/raw_*.jsonl'))
for f in files:
    tag=f.split('raw_')[1][:-6]; fam=tag.split('-')[0]; new=f.startswith(X)
    for ln,l in enumerate(open(f)):
        r=json.loads(l); intent=r['key'].split('|')[0]; conf=r['key'].split('|')[1] if r['kind']=='hard' else None
        for k_it,it in enumerate(r['items'] if isinstance(r['items'],list) else []):
            t=(it.get('text') or '').strip() if isinstance(it,dict) else ''
            iid=f'xf-{tag}-{ln:03d}-{k_it:02d}' if not new else f'xf3-{tag}-{ln:04d}-{k_it:02d}'
            if iid in judged: st['already_judged']+=1; continue
            st[f'raw_{fam}']+=1
            if not t or len(t)<2: st['empty']+=1; continue
            k=norm(t)
            if k in seen: st['exact_dup_within_or_judged']+=1; continue
            if k in ref: st[f'exact_dup_vs_{ref[k]}']+=1; continue
            seen.add(k); lang=it.get('lang') if it.get('lang') in ('zh','mix','en') else 'mix'
            out.append(dict(id=iid,text=t,intent=intent,kind=r['kind'],confuser=conf,lang_gen=lang,style=it.get('style'),fam=fam,round=r['round'],
                            gen_model=r.get('model_resp'),src='new' if new else ('r3_unjudged' if iid in old_c else 'leftover_raw')))
# embed (own cache)
CACHE=f'{X}/emb_xf3.jsonl'; V={}
if os.path.exists(CACHE):
    for l in open(CACHE): r=json.loads(l); V[r['id']]=np.array(r['vec'],np.float32)
AV={}
ACC=f'{X}/emb_Aconfirm_texts.jsonl'   # A_confirm TEXT vectors only (for near-dup removal); labels never read here
need=[o for o in out if o['id'] not in V]+([] if os.path.exists(ACC) else [dict(id=r['id'],text=r['text']) for r in AC])
if need:
    with open(f'{X}/emb_in.jsonl','w') as f:
        for o in need: f.write(json.dumps({'id':o['id'],'text':o['text']},ensure_ascii=False)+'\n')
    subprocess.run(['bin/qwen3dump_xf','-model','models/Qwen3-Embedding-0.6B-Q8_0.gguf','-in',f'{X}/emb_in.jsonl','-out',f'{X}/emb_new.jsonl','-role','classification','-batch','32'],
                   check=True,stderr=open(f'{X}/embed.log','a'))
    acids={r['id'] for r in AC}
    with open(CACHE,'a') as g, open(ACC,'a') as h:
        for l in open(f'{X}/emb_new.jsonl'):
            r=json.loads(l)
            (h if r['id'] in acids else g).write(l)
            if r['id'] not in acids: V[r['id']]=np.array(r['vec'],np.float32)
for l in open(ACC): r=json.loads(l); AV[r['id']]=np.array(r['vec'],np.float32)
RV={}
for fn in ('out/emb_qwen3go.npy','out/emb_indep_qwen3go.npy'):
    Xn=np.load(fn); ids=json.load(open(fn+'.ids.json')); RV.update({i:x for i,x in zip(ids,Xn)})
RV.update(AV)
test_ids=[r['id'] for r in rows if r['split']=='test']+[r['id'] for r in IND]+[r['id'] for r in AMB]+[r['id'] for r in AC]
miss=[i for i in test_ids if i not in RV]; assert not miss,miss[:5]
XT=np.stack([RV[i]/np.linalg.norm(RV[i]) for i in test_ids])
keep=[]
if out:
    XC=np.stack([V[o['id']]/np.linalg.norm(V[o['id']]) for o in out]); S=XC@XT.T; mx=S.max(1)
    for o,m in zip(out,mx):
        o['max_cos_test']=round(float(m),4)
        if m>0.95: st['rm_cos>0.95_vs_test_or_Aconfirm']+=1
        else: keep.append(o)
st['kept_pre_judge']=len(keep)
with open(f'{X}/candidates3.jsonl','w') as f:
    for o in keep: f.write(json.dumps(o,ensure_ascii=False)+'\n')
print(dict(st)); print(collections.Counter((o['fam'],o['src']) for o in keep))
json.dump(dict(st),open(f'{X}/build3_stats.json','w'),indent=1)
