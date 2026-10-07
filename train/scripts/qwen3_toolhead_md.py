"""Render §16 tables (markdown) from out/qwen3_toolhead/toolhead_eval_final.json and union_final.json."""
import json
R = json.load(open('out/qwen3_toolhead/toolhead_eval_final.json')); U = json.load(open('out/qwen3_toolhead/union_final.json'))
f = lambda x: '–' if x is None else f'{x:.3f}'
print('| 方法 | 集合 | n | R@1 | R@3 | R@5 | R@20 | zh R@1/R@5 | mix R@1/R@5 | en R@1/R@5 | MRR |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for r in R:
    if r['n'] == 0: continue
    print(f"| {r['method']} | {r['set']} | {r['n']} | {f(r['all@1'])} | {f(r['all@3'])} | {f(r['all@5'])} | {f(r['all@20'])} | {f(r['zh@1'])} / {f(r['zh@5'])} | {f(r['mix@1'])} / {f(r['mix@5'])} | {f(r['en@1'])} / {f(r['en@5'])} | {r['mrr']:.3f} |")
print()
print('| Track A 变体 | 集合 | n | k | θ | N | 召回 | 平均候选数 | 其中工具头贡献 | 仅意图 scope |')
print('|---|---|---|---|---|---|---|---|---|---|')
keep = {(0, 0, 0), (0, 0, 10), (0, 0, 20), (3, 0.5, 0), (3, 0.5, 10), (5, 0.3, 0), (5, 0.3, 5), (5, 0.3, 10), (5, 0.5, 0), (5, 0.5, 10), (5, 0.7, 10), (5, 0.9, 10), (5, 0.0, 0), (5, 0.0, 10)}
for u in U:
    if (u['k'], u['theta'], u['N']) in keep:
        print(f"| {u['trackA']} | {u['set']} | {u['n']} | {u['k']} | {u['theta']} | {u['N']} | {u['recall']:.3f} | {u['mean_cands']:.1f} | {u['mean_head_cands']:.2f} | {u['scope_only']:.3f} |")
