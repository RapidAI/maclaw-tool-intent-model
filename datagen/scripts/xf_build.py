"""Section 17 data pipeline: flatten raw multi-family generations -> exact dedupe (vs train/anchor, ALL test sets, within) ->
pure-Go Qwen3 (RoleClassification, bin/qwen3dump_xf built from /workspace/maclaw-final) embedding (cached) ->
near-dup removal cos > 0.95 vs every test item (indep 839, ambiguous 95, old test 339) -> data/train_xfamily/candidates.jsonl
(judge runs separately: scripts/xf_judge.py)."""
import json, re, glob, unicodedata, collections, os, subprocess, numpy as np
R = '/workspace/maclaw_reranker'; os.chdir(R)
rows = [json.loads(l) for l in open('data/intent_all.jsonl')]
IND = [json.loads(l) for l in open('data/indep_test.jsonl')]; AMB = [json.loads(l) for l in open('data/indep_ambiguous.jsonl')]
K150 = [json.loads(l) for l in open('data/go_in_GK_test_150.jsonl')]
def norm(s): return re.sub(r'[\W_]+', '', unicodedata.normalize('NFKC', s).lower())
seen_ref = {}
for r in rows: seen_ref[norm(r['text'])] = 'train' if r['split'] == 'train' else 'oldtest'
for r in IND + K150: seen_ref[norm(r['text'])] = 'indep'
for r in AMB: seen_ref[norm(r['text'])] = 'amb'
out, st, seen = [], collections.Counter(), set()
for f in sorted(glob.glob('data/train_xfamily/raw_*.jsonl')):
    tag = f.split('raw_')[1][:-6]; fam = tag.split('-')[0]
    for ln, l in enumerate(open(f)):
        r = json.loads(l); intent = r['key'].split('|')[0]; conf = r['key'].split('|')[1] if r['kind'] == 'hard' else None
        for k_it, it in enumerate(r['items'] if isinstance(r['items'], list) else []):
            t = (it.get('text') or '').strip() if isinstance(it, dict) else ''
            st['raw'] += 1; st[f'raw_{fam}'] += 1
            if not t or len(t) < 2: st['empty'] += 1; continue
            k = norm(t)
            if k in seen: st['exact_dup_within'] += 1; continue
            if k in seen_ref: st[f'exact_dup_vs_{seen_ref[k]}'] += 1; continue
            seen.add(k)
            lang = it.get('lang') if it.get('lang') in ('zh', 'mix', 'en') else 'mix'
            out.append(dict(id=f'xf-{tag}-{ln:03d}-{k_it:02d}', text=t, intent=intent, kind=r['kind'], confuser=conf, lang_gen=lang,
                            style=it.get('style'), fam=fam, round=r['round'], gen_model=r['model_resp']))
for o in out:
    cjk = len(re.findall(r'[\u4e00-\u9fff]', o['text'])); asc = len([w for w in re.findall(r'[A-Za-z]{2,}', o['text']) if w.lower() != 'maclaw'])
    L = o['lang_gen']
    if cjk == 0: L = 'en'
    elif L == 'en': L = 'mix'
    elif L == 'mix' and asc == 0: L = 'zh'
    elif L == 'zh' and asc > 2: L = 'mix'
    o['lang'] = L
# embed (cache)
CACHE = 'out/xf/emb_xf_qwen3go.jsonl'; V = {}
if os.path.exists(CACHE):
    for l in open(CACHE): r = json.loads(l); V[r['id']] = np.array(r['vec'], np.float32)
todo = [o for o in out if o['id'] not in V]
if todo:
    with open('out/xf/emb_in.jsonl', 'w') as f:
        for o in todo: f.write(json.dumps({'id': o['id'], 'text': o['text']}, ensure_ascii=False) + '\n')
    subprocess.run(['bin/qwen3dump_xf', '-model', 'models/Qwen3-Embedding-0.6B-Q8_0.gguf', '-in', 'out/xf/emb_in.jsonl', '-out', 'out/xf/emb_new.jsonl',
                    '-role', 'classification', '-batch', '32'], check=True, stderr=open('logs/xf/embed.log', 'a'))
    with open(CACHE, 'a') as g:
        for l in open('out/xf/emb_new.jsonl'):
            r = json.loads(l); V[r['id']] = np.array(r['vec'], np.float32); g.write(l)
# reference test vectors (same Go runner, same role; section 10)
RV = {}
for fn in ('out/emb_qwen3go.npy', 'out/emb_indep_qwen3go.npy'):
    X = np.load(fn); ids = json.load(open(fn + '.ids.json')); RV.update({i: x / np.linalg.norm(x) for i, x in zip(ids, X)})
test_ids = [r['id'] for r in rows if r['split'] == 'test'] + [r['id'] for r in IND] + [r['id'] for r in AMB]
miss = [i for i in test_ids if i not in RV]; assert not miss, miss[:5]
XT = np.stack([RV[i] for i in test_ids])
XC = np.stack([V[o['id']] / np.linalg.norm(V[o['id']]) for o in out])
S = XC @ XT.T; mx = S.max(1); keep = []
for o, m, a in zip(out, mx, S.argmax(1)):
    o['max_cos_test'] = round(float(m), 4); o['nn_test'] = test_ids[a]
    if m > 0.95: st['rm_cos>0.95_vs_test'] += 1
    else: keep.append(o)
st['kept_pre_judge'] = len(keep)
with open('data/train_xfamily/candidates.jsonl', 'w') as f:
    for o in keep: f.write(json.dumps(o, ensure_ascii=False) + '\n')
print(dict(st)); print(collections.Counter(o['fam'] for o in keep))
print('max cos to test p50 %.3f p90 %.3f p99 %.3f' % tuple(np.percentile(mx, [50, 90, 99])))
json.dump(dict(st), open('out/xf/build_stats.json', 'w'), indent=1)
