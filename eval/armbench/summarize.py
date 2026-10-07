import json,glob,os
def one(path):
  if os.path.getsize(path)==0: return {"file":os.path.basename(path),"empty":True}
  d=json.load(open(path))
  o={"file":os.path.basename(path),"load_ms":round(d.get("load_ms",0),1),"rss":round(d.get("rss_after_load_mb",0),1)}
  for k,v in d.items():
    if not isinstance(v,dict): continue
    tag="short" if "short" in k else ("long" if "long" in k else None)
    if not tag: continue
    if "stats" in v:
      s=v["stats"]; o[tag+"_p50"]=round(s["p50_ms"],1); o[tag+"_p95"]=round(s["p95_ms"],1)
      o[tag+"_load"]=v.get("load_start")
    if "texts_per_s" in v:
      tps=v["texts_per_s"]; o[tag+"_tps"]=round(sum(tps)/len(tps),2); o[tag+"_tps_all"]=[round(x,2) for x in tps]
  return o
for p in sorted(glob.glob(os.path.expanduser("~/maclaw_q8p_armbench/out/r*.json"))):
  print(one(p))
