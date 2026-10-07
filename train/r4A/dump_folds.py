# Dump the exact train/test sets of the levers recipe (ids + label idx) so a torch trainer can reproduce them.
import os, json, numpy as np, pickle
exec(open('out/replace_audit/r4A/levers_a.py').read().split("def job(spec):")[0])
LAB=sorted({r['intent'] for r in old_tr+FIX+XF})
def enc(items): return [r['id'] for r in items],np.array([LAB.index(r['intent']) for r in items])
out={}
for rec,extra in RECIPES.items():
    for k,F in enumerate(FAMS):
        out[(rec,'lofo',k)]=(enc(old_tr*OLDW+FIX+[r for r in XF if r['fam']!=F]+extra),[r['id'] for r in XF if r['fam']==F])
    for k in range(5):
        out[(rec,'cv',k)]=(enc([old_tr[i] for i in np.setdiff1d(idx,OF[k])]*OLDW+FIX+XF+extra),[old_tr[i]['id'] for i in OF[k]])
    out[(rec,'final',0)]=(enc(old_tr*OLDW+FIX+XF+extra),[r['id'] for r in IND])
pickle.dump(dict(folds=out,LAB=LAB,FAMS=FAMS),open('out/replace_audit/r4A/folds.pkl','wb')); print(len(out),len(LAB))
