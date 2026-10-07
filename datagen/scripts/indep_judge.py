"""Step 3: blind label verification by a third model family (judge never sees the intended label)."""
import json, sys, os, re, threading, concurrent.futures as cf
sys.path.insert(0, 'scripts'); import llmc
JUDGE = sys.argv[1] if len(sys.argv) > 1 else 'nvidia:z-ai/glm-5.3'
BATCH, CONC = 15, int(sys.argv[2]) if len(sys.argv) > 2 else 4
TAGJ = sys.argv[3] if len(sys.argv) > 3 else 'cohere'
OUT = f'data/indep_judge_raw_{TAGJ}.jsonl'
SUBSET = sys.argv[4] if len(sys.argv) > 4 else None  # optional file of ids to judge
D = json.load(open('data/intent_defs.json'))['defs']
UNK = "闲聊、寒暄、情绪表达或与本机 agent 任务无关、没有可执行任务的消息。"
menu = "\n".join(f"- {x['label']}: {x['tree_text'][:420]}" for x in D if x['label'] not in ('ambiguous', 'unknown')) + f"\n- unknown: {UNK}"
labels = {x['label'] for x in D if x['label'] != 'ambiguous'}
SYS = ("You are a careful annotator for the intent taxonomy of MaClaw, a desktop AI agent. For each user message choose the single best intent label "
       "from the list, judged by the official descriptions and their boundary rules, plus an optional second label if genuinely ambiguous. "
       "Labels:\n" + menu + "\n\nReturn ONLY JSON: {\"results\":[{\"i\":<index>,\"label\":\"...\",\"second\":\"...or empty\",\"conf\":0-1}]}")
C = [json.loads(l) for l in open('data/indep_filtered.jsonl')]
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        for r in json.loads(l)['results']: done.add(r['id'])
todo = [c for c in C if c['id'] not in done]
if SUBSET: keep = set(json.load(open(SUBSET))); todo = [c for c in todo if c['id'] in keep]
batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
lock = threading.Lock()
def run(b):
    msg = "\n".join(f"{k}. {c['text']}" for k, c in enumerate(b))
    for attempt in range(3):
        try:
            c, m, dt = llmc.chat(JUDGE, [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': "Messages:\n" + msg}], temperature=0, max_tokens=6000, timeout=300)
            i, j = c.find('{'), c.rfind('}'); res = json.loads(c[i:j + 1])['results']
            out = []
            for r in res:
                k = int(r['i'])
                if 0 <= k < len(b): out.append({'id': b[k]['id'], 'label': r.get('label'), 'second': r.get('second') or '', 'conf': r.get('conf')})
            if len(out) < len(b) * 0.8: raise ValueError(f'only {len(out)}/{len(b)} parsed')
            with lock, open(OUT, 'a') as f: f.write(json.dumps({'judge': JUDGE, 'model': m, 'sec': round(dt, 1), 'results': out}, ensure_ascii=False) + '\n')
            return f'ok {len(out)} {dt:.0f}s'
        except Exception as e: err = str(e)[:160]
    return 'FAIL ' + err
print(len(todo), 'to judge in', len(batches), 'batches', flush=True)
with cf.ThreadPoolExecutor(CONC) as ex:
    for r in ex.map(run, batches): print(r, flush=True)
