"""r4 s7 copy of scripts/xf_judge.py (same judge, prompt, batch, agreement rule); reads/writes only r4A/xf3/."""
"""Blind judge (Cohere Command-A, a family not used as generator) on r4A/xf3/candidates3.jsonl; same prompt as indep_judge.py."""
import json, sys, os, threading, concurrent.futures as cf
sys.path.insert(0, 'scripts'); import llmc
JUDGE = os.environ.get('J3_JUDGE', 'cohere:command-a-03-2025'); J3F = set(os.environ.get('J3_FAMS', '').split(',')) - {''}; BATCH, CONC = 15, int(sys.argv[1]) if len(sys.argv) > 1 else 4
OUT = 'out/replace_audit/r4A/xf3/judge_raw3.jsonl'
src = open('scripts/indep_judge.py').read(); SYS = None
exec(src[src.index('D = json.load'):src.index('C = [json.loads')])
C = [json.loads(l) for l in open('out/replace_audit/r4A/xf3/candidates3.jsonl')]
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        for r in json.loads(l)['results']: done.add(r['id'])
todo = [c for c in C if c['id'] not in done and (not J3F or c['fam'] in J3F) and (os.environ.get('J3_PART') is None or int(c['id'].rsplit('-',2)[1]) % 2 == int(os.environ['J3_PART']))]; batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
lock = threading.Lock()
def run(b):
    msg = "\n".join(f"{k}. {c['text']}" for k, c in enumerate(b)); err = ''
    for attempt in range(3):
        try:
            c, m, dt = llmc.chat(JUDGE, [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': "Messages:\n" + msg}], temperature=0, max_tokens=4000, timeout=120)
            i, j = c.find('{'), c.rfind('}'); res = json.loads(c[i:j + 1])['results']
            out = [{'id': b[int(r['i'])]['id'], 'label': r.get('label'), 'second': r.get('second') or '', 'conf': r.get('conf')} for r in res if 0 <= int(r['i']) < len(b)]
            if len(out) < len(b) * 0.8: raise ValueError('short')
            with lock, open(OUT, 'a') as f: f.write(json.dumps({'judge': JUDGE, 'model': m, 'sec': round(dt, 1), 'results': out}, ensure_ascii=False) + '\n')
            return len(out)
        except Exception as e: err = str(e)[:120]
    return 'FAIL ' + err
print(len(todo), 'to judge', flush=True)
with cf.ThreadPoolExecutor(CONC) as ex: res = list(ex.map(run, batches))
print('done', sum(r for r in res if isinstance(r, int)), [r for r in res if not isinstance(r, int)][:3])
# merge -> data/train_xfamily/train_xf.jsonl
SAFEU = set(json.load(open('data/intent_meta.json'))['safe_unknown']); J = {}
for l in open(OUT):
    for r in json.loads(l)['results']: J[r['id']] = r
import collections; st = collections.Counter(); keep = []
for c in C:
    if c['id'] not in J: st['unjudged'] += 1; continue
    j = J[c['id']]; c['judge'] = j['label']; c['judge_second'] = j['second']
    if j['label'] == c['intent'] or (c['intent'] == 'unknown' and j['label'] in SAFEU): keep.append(c); st[f'agree_{c["fam"]}'] += 1
    else: st[f'disagree_{c["fam"]}'] += 1
with open('out/replace_audit/r4A/xf3/train_xf3.jsonl', 'w') as f:
    for c in keep: f.write(json.dumps(c, ensure_ascii=False) + '\n')
print(dict(st), 'kept', len(keep)); json.dump(dict(st), open('out/replace_audit/r4A/xf3/judge3_stats.json', 'w'), indent=1)
