import json, sys, time, requests
d = json.load(open('data/intent_defs.json')); P = d['tree_prompt']
port = sys.argv[1]
for q in sys.argv[2:]:
    sysmsg = P.replace('<<USER_MESSAGE>>', q)
    t = time.time()
    r = requests.post(f'http://127.0.0.1:{port}/v1/chat/completions', json=dict(messages=[{"role":"system","content":sysmsg},{"role":"user","content":q+" /no_think"}], temperature=0, max_tokens=300, cache_prompt=True)).json()
    u = r.get('usage', {}); tm = r.get('timings', {})
    print(f"{time.time()-t:.2f}s prompt={u.get('prompt_tokens')} cached={tm.get('cache_n')} gen={u.get('completion_tokens')}", repr(r['choices'][0]['message']['content'][:300]))
