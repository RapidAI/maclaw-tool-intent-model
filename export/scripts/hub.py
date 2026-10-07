"""Thin OpenAI-compatible client for the user's token-bank hub (key read from the box's existing client config, never printed)."""
import json, time, urllib.request
CFG = json.load(open('<HUB_CONFIG_JSON>'))
BASE, KEY = CFG['base_url'].rstrip('/'), CFG['api_key']
def _req(path, body=None, timeout=180):
    r = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                               headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read())
def models(): return _req('/models')
def chat(model, messages, temperature=0.9, max_tokens=4000, timeout=240):
    t = time.time()
    d = _req('/chat/completions', {'model': model, 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens}, timeout)
    return d['choices'][0]['message'].get('content') or '', d.get('model'), time.time() - t
