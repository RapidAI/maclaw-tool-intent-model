# Merge run7 pkls, export the levers_eval7 STAGE1 WINNER as one MLP head (export_ens.py logic). usage (ws root): python export7.py TAG
import pickle, json, sys, os, subprocess
P='out/replace_audit/r4A'; tag=sys.argv[1]; r={}
for f in ('levers_raw.pkl','ls7_s0.pkl','ls7_s1.pkl','ls7_s2.pkl'):
    D=pickle.load(open(f'{P}/{f}','rb')); r.update(D['res']); F=D['FAMS']
pickle.dump(dict(res=r,FAMS=F),open(f'{P}/run7_all.pkl','wb'))
w=json.load(open(f'{P}/levers_eval7.json'))['winner']; mem=','.join(f'{a}:{b}' for a,b in (eval(m) if isinstance(m,str) else m for m in w['members']))
print('winner',mem,w['T'],w['oof']['tau'],w['oof']['tau_s'])
subprocess.run([sys.executable,f'{P}/export_ens.py',mem,tag,str(w['T']),str(w['oof']['tau']),str(w['oof']['tau_s'])],env=dict(os.environ,LEV_PKL=f'{P}/run7_all.pkl'),check=True)
os.remove(f'{P}/run7_all.pkl')
