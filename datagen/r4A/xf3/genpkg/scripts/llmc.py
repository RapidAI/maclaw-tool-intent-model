# local-only llmc stand-in: OpenAI-compatible vLLM on 127.0.0.1:8011 (Qwen/Qwen3.8-27B-FP8), thinking disabled. Same chat() signature as scripts/llmc.py.
import json, time, urllib.request
def chat(spec, messages, temperature=0.9, max_tokens=6000, timeout=300, retries=4, json_mode=False):
    prov, model = spec.split(':', 1); assert prov == 'local'
    body = {'model': model, 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'top_p': 0.95,
            'chat_template_kwargs': {'enable_thinking': False}}
    err = None
    for a in range(max(1, retries)):
        t = time.time()
        try:
            r = urllib.request.Request('http://127.0.0.1:8011/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            d = json.load(urllib.request.urlopen(r, timeout=timeout)); return d['choices'][0]['message']['content'], d.get('model', model), time.time() - t
        except Exception as e: err = e; time.sleep(2)
    raise RuntimeError(str(err))
