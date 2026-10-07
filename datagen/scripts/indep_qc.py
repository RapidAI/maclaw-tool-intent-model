"""Step 1: flatten raw generations, exact-dedupe -> data/indep_candidates.jsonl (+ ZZ embed input)."""
import json, re, glob, unicodedata, collections
rows = [json.loads(l) for l in open('data/intent_all.jsonl')]
def norm(s): return re.sub(r'[\W_]+', '', unicodedata.normalize('NFKC', s).lower())
seen_train = {norm(r['text']): r['split'] for r in rows}
out, stats, seen = [], collections.Counter(), set()
for f in sorted(glob.glob('data/indep_raw_*.jsonl')):
    tag = f.split('indep_raw_')[1][:-6]; k_raw = -1
    for l in open(f):
        r = json.loads(l)
        intent = r['key'].split('|')[0]; conf = r['key'].split('|')[1] if r['kind'] == 'hard' else None
        for it in r['items']:
            t = (it.get('text') or '').strip(); stats['raw'] += 1; stats[f'raw_{tag}'] += 1; k_raw += 1
            if not t: stats['empty'] += 1; continue
            k = norm(t)
            if k in seen: stats['exact_dup_within'] += 1; continue
            if k in seen_train: stats[f'exact_dup_vs_{seen_train[k]}'] += 1; continue
            seen.add(k)
            lang = it.get('lang') if it.get('lang') in ('zh', 'mix', 'en') else 'mix'
            out.append(dict(id=f'indep-{tag}-{k_raw:04d}', text=t, intent=intent, kind=r['kind'], confuser=conf,
                            lang_gen=lang, style=it.get('style'), gen=tag, gen_model=r['model_resp']))
# language: trust the generator's tag unless the script contradicts it
for o in out:
    cjk = len(re.findall(r'[\u4e00-\u9fff]', o['text']))
    asc = len([w for w in re.findall(r'[A-Za-z]{2,}', o['text']) if w.lower() != 'maclaw'])
    L = o['lang_gen']
    if cjk == 0: L = 'en'
    elif L == 'en': L = 'mix'
    elif L == 'mix' and asc == 0: L = 'zh'
    elif L == 'zh' and asc > 2: L = 'mix'
    o['lang'] = L
with open('data/indep_candidates.jsonl', 'w') as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + '\n')
stats['kept'] = len(out)
print(dict(stats)); print(collections.Counter((o['gen'], o['lang']) for o in out))
json.dump(dict(stats), open('out/indep_qc_step1.json', 'w'), indent=1)
