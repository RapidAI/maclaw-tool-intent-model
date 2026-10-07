"""Make an xf OvR tool head loadable by corelib/tool/toolhead.go (LoadToolHead): add label_table_version
(core<N>@sha256(tools joined by \\n)[:12]), trained[] (False for all-zero W rows, i.e. no positives in training), theta, top_k.
W/b are not changed. usage: xf_toolhead_compat.py <in.json> <out.json>"""
import json, sys, hashlib
h = json.load(open(sys.argv[1])); ref = json.load(open('heads/tool_head_ovr_qwen3go.json'))
assert h['format'] == ref['format'] == 'maclaw-tool-head-ovr-v1' and h['tools'] == ref['tools'] and h['dims'] == 1024
assert h['embedder_model_id'] == ref['embedder_model_id'] and h['embed_role'] == ref['embed_role'] and h['l2norm']
ver = 'core%d@%s' % (len(h['tools']), hashlib.sha256('\n'.join(h['tools']).encode()).hexdigest()[:12]); assert ver == ref['label_table_version']
h['label_table_version'] = ver
h['trained'] = [any(abs(x) > 0 for x in w) for w in h['W']]
h.setdefault('theta', ref['theta']); h.setdefault('top_k', ref['top_k'])
assert len(h['W']) == len(h['b']) == len(h['trained']) == len(h['tools']) and all(len(w) == 1024 for w in h['W'])
json.dump(h, open(sys.argv[2], 'w')); print(sys.argv[2], ver, 'untrained:', [t for t, k in zip(h['tools'], h['trained']) if not k])
