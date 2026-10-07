# Dump head probs (train: LOFO-OOF for xf rows, old-train CV for old_tr rows; indep: final head) for the lever-2 L2 rescue sim.
# usage: LEV_PKLS=... python dump_probs.py "recipe:seed,recipe:seed" T tau tau_s out.json
import os, sys, json
sys.argv, args = sys.argv[:1], sys.argv[1:]
os.environ.setdefault('LEV_OUT','/tmp/le2_dummy.json')
src=open('out/replace_audit/r4A/levers_eval2.py').read().split("keys=sorted(")[0]; exec(src)
mem=[(m.split(':')[0],int(m.split(':')[1])) for m in args[0].split(',')]; Tm,tau,ts=map(float,args[1:4])
out=dict(T=Tm,tau=tau,tau_s=ts,train={},indep={})
for kind,R,dst,key in (('lofo',R_oof,'train','text'),('cv',R_cv,'train','text'),('final',IND,'indep','id')):
    P=T.softmax(Zs(mem,kind)/Tm)
    for r,p in zip(R,P): j=int(p.argmax()); out[dst][r[key]]=[r['intent'],LAB[j],float(p[j])]
json.dump(out,open(args[4],'w'))
print('dumped',len(out['train']),len(out['indep']))
