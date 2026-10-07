# Freeze: record sha256 + counts of the confirmation sets; make them read-only. Run once; never edit *_confirm.jsonl afterwards.
import json, hashlib, os, collections, datetime, sys
D=os.path.dirname(os.path.abspath(__file__)); out={}
for k in sys.argv[1:]:
    p=f'{D}/{k}_confirm.jsonl'; b=open(p,'rb').read(); R=[json.loads(l) for l in b.decode().splitlines()]
    lab=collections.Counter(r.get('label') or r.get('intent') or 'coding' for r in R)
    out[k]=dict(file=os.path.basename(p),sha256=hashlib.sha256(b).hexdigest(),n=len(R),gen=sorted({r['gen'] for r in R}),judge=sorted({r['judge'] for r in R}),
                langs=dict(collections.Counter(r.get('lang') for r in R)),labels=len(lab) if k=='A' else dict(lab),frozen=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
    os.chmod(p,0o444)
prev=json.load(open(f'{D}/FROZEN.json')) if os.path.exists(f'{D}/FROZEN.json') else {}
prev.update(out); json.dump(prev,open(f'{D}/FROZEN.json','w'),indent=1)
with open(f'{D}/SHA256SUMS','w') as f:
    for k,v in sorted(prev.items()): f.write(f"{v['sha256']}  {v['file']}\n")
print(json.dumps(out,indent=1))
