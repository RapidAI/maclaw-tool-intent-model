# r4 s7: same blind judge prompt (scripts/indep_judge.py SYS) on the local Qwen/Qwen3.8-27B-FP8 vLLM; only for NON-Qwen-generated items.
import json, sys, os, threading, concurrent.futures as cf
sys.path.insert(0, 'scripts'); import llmc
JUDGE = 'local:Qwen/Qwen3.8-27B-FP8'; BATCH = 15; CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 24
SYS = open('judge_sys.txt').read(); C = [json.loads(l) for l in open(sys.argv[1])]; OUT = 'judge_raw_qwenjudge.jsonl'
assert not any(c['fam'] == 'qwen' for c in C)
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        for r in json.loads(l)['results']: done.add(r['id'])
todo = [c for c in C if c['id'] not in done]; batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]; lock = threading.Lock()
def run(b):
    msg = "\n".join(f"{k}. {c['text']}" for k, c in enumerate(b)); err = ''
    for attempt in range(3):
        try:
            c, m, dt = llmc.chat(JUDGE, [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': "Messages:\n" + msg}], temperature=0, max_tokens=4000, timeout=300)
            i, j = c.find('{'), c.rfind('}'); res = json.loads(c[i:j + 1])['results']
            out = [{'id': b[int(r['i'])]['id'], 'label': r.get('label'), 'second': r.get('second') or '', 'conf': r.get('conf')} for r in res if 0 <= int(r['i']) < len(b)]
            if len(out) < len(b) * 0.8: raise ValueError('short')
            with lock, open(OUT, 'a') as f: f.write(json.dumps({'judge': JUDGE, 'model': m, 'sec': round(dt, 1), 'results': out}, ensure_ascii=False) + '\n')
            return len(out)
        except Exception as e: err = str(e)[:120]
    return 'FAIL ' + err
print(len(todo), 'to judge', flush=True)
with cf.ThreadPoolExecutor(CONC) as ex: res = list(ex.map(run, batches))
print('done', sum(r for r in res if isinstance(r, int)), [r for r in res if not isinstance(r, int)][:3], flush=True)
