# Diagnostic only (no selection): for candidates with no locked-rule point, show the closest sweep points.
import os, re
src=open('out/replace_audit/r4A/levers_eval7.py').read().split("keys=sorted(")[0]
src=src.replace("c=[d for d in sw if d['sel']>=0.976 and d['sens_prec']>=0.997]; return Tm,(max(c,key=lambda d:(d['cov'],-d['tau_s'],-d['tau'])) if c else None)","return Tm,sw")
exec(src)
keys=sorted({(k[0],k[1]) for k in res}); groups={}
for r,s in keys: groups.setdefault(r,[]).append((r,s))
for name,mem in list(groups.items()):
    Tm,sw=rule(Zs(mem,'lofo'),R_oof)
    a=max(sw,key=lambda d:d['sel']); b=max(sw,key=lambda d:d['sens_prec']-1e-3*d['sensFE']*0)
    sp=[d for d in sw if d['sens_prec']>=0.997]; s1=[d for d in sw if d['sel']>=0.976]
    print(name,'x%d'%len(mem),'T=%.3f'%Tm,'maxsel',{k:round(v,4) if isinstance(v,float) else v for k,v in a.items()})
    print('   best sens_prec>=.997 pt:',max(sp,key=lambda d:d['sel']) if sp else None)
    print('   best sel>=.976 pt:',max(s1,key=lambda d:d['sens_prec']) if s1 else None)
