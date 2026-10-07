"""For head-uncertain test cases (p_max < tau): resolve with (a) cross-encoder over
label descriptions, (b) small local LLM restricted to the head's top-5 labels."""
import json, sys, time, numpy as np, requests
rows = [json.loads(l) for l in open('data/intent_all.jsonl')]
te = [r for r in rows if r['split'] == 'test']
meta = json.load(open('data/intent_meta.json')); SAFEU = set(meta['safe_unknown'])
defs = {d['label']: d for d in json.load(open('data/intent_defs.json'))['defs']}
defs['unknown'] = dict(tree_text='闲聊、问候或没有明确任务的消息 (chit-chat / greeting / no actionable task).')
h = json.load(open('heads/head_mlp_gemma.json')); P = np.load('heads/head_mlp_gemma.json.test_probs.npy'); L = h['labels']
unc = [(r, p) for r, p in zip(te, P) if p.max() < h['tau']]
def ok(g, p): return p == g or (g == 'unknown' and p in SAFEU)
mode = sys.argv[1]
res = []
if mode == 'rerank':
    from sentence_transformers import CrossEncoder
    import torch; torch.set_num_threads(4)
    name = sys.argv[2]
    m = CrossEncoder(name, device='cpu', max_length=384, trust_remote_code=True)
    for r, p in unc:
        top = [L[j] for j in np.argsort(-p)[:5]]
        t = time.time(); s = m.predict([(r['text'], f"{l}: {defs[l]['tree_text'][:400]}") for l in top], show_progress_bar=False)
        ms = (time.time() - t) * 1000
        s = np.asarray(s, dtype=float)
        hp = np.array([p[L.index(l)] for l in top])
        # fuse: rerank score as logit-ish evidence + head prior
        pick_rr = top[int(s.argmax())]
        pick_fuse = top[int((np.log(hp + 1e-9) + s).argmax())]
        res.append(dict(text=r['text'], gold=r['intent'], head=top[0], rr=pick_rr, fuse=pick_fuse, ms=ms))
    tag = name.split('/')[-1]
elif mode == 'llm':
    port, tag = sys.argv[2], sys.argv[3]
    for r, p in unc:
        top = [L[j] for j in np.argsort(-p)[:5]]
        opts = "\n".join(f"- {l}: {defs[l]['tree_text'][:300]}" for l in top)
        sysmsg = ("You are an intent classifier for a desktop agent. Choose which ONE intent best matches the user's message, "
                  "or none_of_these if none fits. Candidates:\n" + opts +
                  '\nReturn ONLY JSON: {"intent": "<name>", "score": 0.0-1.0}')
        schema = {"type": "object", "required": ["intent", "score"], "additionalProperties": False,
                  "properties": {"intent": {"type": "string", "enum": top + ["none_of_these"]}, "score": {"type": "number"}}}
        t = time.time()
        out = requests.post(f'http://127.0.0.1:{port}/v1/chat/completions', json=dict(
            messages=[{"role": "system", "content": sysmsg}, {"role": "user", "content": r['text'] + " /no_think"}],
            temperature=0, max_tokens=60, cache_prompt=True), timeout=120).json()
        ms = (time.time() - t) * 1000
        txt = out['choices'][0]['message']['content']
        try: j = json.loads(txt[txt.find('{'):txt.rfind('}') + 1]); pick = j['intent']; sc = float(j.get('score', 0))
        except Exception: pick, sc = 'PARSE_ERR', 0
        res.append(dict(text=r['text'], gold=r['intent'], head=top[0], llm=pick, llm_score=sc, ms=ms,
                        prompt_tokens=out.get('usage', {}).get('prompt_tokens')))
json.dump(res, open(f'out/fallback_{mode}_{tag}.json', 'w'), ensure_ascii=False, indent=0)
n = len(res); ms = [x['ms'] for x in res]
print(f"{mode}/{tag}: uncertain n={n}  head top1 acc on these={np.mean([ok(x['gold'], x['head']) for x in res]):.3f}")
for k in ('rr', 'fuse', 'llm'):
    if k in res[0]:
        print(f"   {k}: acc={np.mean([ok(x['gold'], x[k]) for x in res]):.3f}" + (f"  none_of_these={sum(x[k]=='none_of_these' for x in res)}" if k == 'llm' else ''))
print("   latency ms p50 %.0f p95 %.0f p99 %.0f" % tuple(np.percentile(ms, [50, 95, 99])))
