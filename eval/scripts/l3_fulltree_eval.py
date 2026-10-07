"""Accuracy of a small local LLM on maclaw's full L3 tree prompt (no deadline), subset of test.
Static-prefix variant: message moved to the user turn so the 6.3k-token tree stays KV-cached."""
import json, re, sys, time, numpy as np, requests
port, tag, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open('data/intent_defs.json'))
P = d['tree_prompt'].replace('"<<USER_MESSAGE>>"', '(the user message is given in the next user turn)')
meta = json.load(open('data/intent_meta.json')); SENS = set(meta['sensitive']); SAFEU = set(meta['safe_unknown'])
te = [json.loads(l) for l in open('data/intent_all.jsonl') if json.loads(l)['split'] == 'test'][::step]
def ok(g, p): return p == g or (g == 'unknown' and p in SAFEU)
res = []
requests.post(f'http://127.0.0.1:{port}/v1/chat/completions', json=dict(messages=[{"role": "system", "content": P}, {"role": "user", "content": "warmup /no_think"}], max_tokens=5, cache_prompt=True))
for r in te:
    t = time.time()
    out = requests.post(f'http://127.0.0.1:{port}/v1/chat/completions', json=dict(
        messages=[{"role": "system", "content": P}, {"role": "user", "content": r['text'] + " /no_think"}],
        temperature=0, max_tokens=80, cache_prompt=True), timeout=300).json()
    ms = (time.time() - t) * 1000
    txt = out['choices'][0]['message']['content']
    m = re.search(r'"skill"\s*:\s*"([a-z_]+)"', txt); pred = m.group(1) if m else 'PARSE_ERR'
    m2 = re.search(r'"score"\s*:\s*([0-9.]+)', txt); sc = float(m2.group(1)) if m2 else 0
    res.append(dict(id=r['id'], text=r['text'], gold=r['intent'], pred=pred, score=sc, ms=ms, valid_json=txt.strip().endswith('}') and '...' not in txt))
    print(len(res), r['intent'], pred, f"{ms:.0f}ms", flush=True)
json.dump(res, open(f'out/l3_fulltree_{tag}.json', 'w'), ensure_ascii=False, indent=0)
ms = [x['ms'] for x in res]
print(f"{tag}: n={len(res)} acc={np.mean([ok(x['gold'], x['pred']) for x in res]):.3f} acc(score>=0.5)={np.mean([ok(x['gold'], x['pred']) for x in res if x['score']>=0.5]):.3f} "
      f"parse_err={sum(x['pred']=='PARSE_ERR' for x in res)} strict_json_ok={np.mean([x['valid_json'] for x in res]):.2f} "
      f"sens_false={sum(1 for x in res if x['pred'] in SENS and x['pred']!=x['gold'])} lat ms p50={np.percentile(ms,50):.0f} p95={np.percentile(ms,95):.0f} p99={np.percentile(ms,99):.0f}")
