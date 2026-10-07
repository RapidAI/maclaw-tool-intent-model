#!/usr/bin/env python3
"""Regenerate the first-run release asset table + upload manifest.

Writes:
  <worktree>/corelib/embedding/release_assets.json   (go:embed'ed; size + sha256 pins)
  /workspace/maclaw_reranker/out/replace_audit/release_manifest.md
Re-run whenever a head changes (e.g. after the cross-family retrain):
  python3 scripts/gen_release_manifest.py [--worktree /workspace/maclaw-replace] [--intent-head PATH] [--tool-head PATH] [--interrupt-head PATH]
Then rebuild/commit. Nothing is uploaded."""
import argparse, hashlib, json, os, datetime
R = '/workspace/maclaw_reranker'
ap = argparse.ArgumentParser()
ap.add_argument('--worktree', default='/workspace/maclaw-replace')
ap.add_argument('--model', default=f'{R}/models/Qwen3-Embedding-0.6B-Q8_0.gguf')
ap.add_argument('--intent-head', default=f'{R}/heads/head_mlp_qwen3go_xf.json')
ap.add_argument('--tool-head', default=f'{R}/out/replace_audit/release_r4/tool_head_ovr_qwen3go.json')  # r4 pin e5235285; heads/tool_head_ovr_qwen3go.json is a stale 08:24 file (ae9fad86)
ap.add_argument('--interrupt-head', default=f'{R}/heads/interrupt_relevance_qwen3go.json')
ap.add_argument('--out', default=f'{R}/out/replace_audit/release_manifest.md')
a = ap.parse_args()
BASE = 'https://github.com/RapidAI/MaClaw/releases/download/Model_Release'
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
items = [('model', '', a.model), ('intent_head', 'intent_heads', a.intent_head), ('tool_head', 'tool_heads', a.tool_head), ('interrupt_head', 'interrupt_heads', a.interrupt_head)]
assets = []
for kind, sub, p in items:
    assets.append({'filename': os.path.basename(p), 'kind': kind, 'subdir': sub, 'size': os.path.getsize(p), 'sha256': sha(p), '_local': os.path.realpath(p)})
doc = {'release': BASE, 'generated': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
       'assets': [{k: v for k, v in x.items() if not k.startswith('_')} for x in assets]}
dst = os.path.join(a.worktree, 'corelib/embedding/release_assets.json')
json.dump(doc, open(dst, 'w'), indent=1); open(dst, 'a').write('\n')
L = ['# First-run release assets — upload manifest (NOT uploaded)', '',
     f'Release: tag `Model_Release` → `{BASE}/<filename>`', f'Generated: {doc["generated"]} by `scripts/gen_release_manifest.py` (re-run after any head change; it also rewrites `corelib/embedding/release_assets.json`).', '',
     '| filename | kind | install dir (under ~/.maclaw/models) | local path | size (bytes) | sha256 | download URL |', '|---|---|---|---|---|---|---|']
for x in assets:
    L.append(f"| `{x['filename']}` | {x['kind']} | `{x['subdir'] or '.'}` | `{x['_local']}` | {x['size']:,} | `{x['sha256']}` | {BASE}/{x['filename']} |")
L += ['', 'Verify after upload: `curl -L -o f <url> && sha256sum f` must match. Hub mirrors the same filenames at `<hub>/api/v1/models/<filename>`.']
open(a.out, 'w').write('\n'.join(L) + '\n')
print(dst); print(a.out)
for x in assets: print(x['filename'], x['size'], x['sha256'])
