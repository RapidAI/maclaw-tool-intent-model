import json, sys, time, requests
d = json.load(open('data/intent_defs.json')); P = d['tree_prompt'].replace('"<<USER_MESSAGE>>"', '(the user message is given in the next user turn)')
port = sys.argv[1]
for q in sys.argv[2:]:
    t = time.time()
    r = requests.post(f'http://127.0.0.1:{port}/v1/chat/completions', json=dict(messages=[{"role":"system","content":P},{"role":"user","content":q+" /no_think"}], temperature=0, max_tokens=300, cache_prompt=True)).json()
    u = r.get('usage', {}); tm = r.get('timings', {})
    print(f"{time.time()-t:.2f}s prompt={u.get('prompt_tokens')} cached={tm.get('cache_n')} gen={u.get('completion_tokens')} pp={tm.get('prompt_per_second',0):.0f}t/s tg={tm.get('predicted_per_second',0):.1f}t/s", repr(r['choices'][0]['message']['content'][:200]))
