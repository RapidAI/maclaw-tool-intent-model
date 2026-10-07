"""Section 9: summarize Go e2e runs (any input set). Usage: analyze_go_cls.py out/go_cls_*.jsonl"""
import json, sys, numpy as np, collections
sys.argv, files = [sys.argv[0]], sys.argv[1:]
exec(open('scripts/analyze_go_indep.py').read().split("f = sys.argv[1]")[0])
for f in files:
    R = []
    for l in open(f):
        try: R.append(json.loads(l))
        except Exception: pass
    rep(f, [r for r in R if r['id'] in gold])
