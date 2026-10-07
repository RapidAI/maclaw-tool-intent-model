# r4 intent head: r3W5F recipe + hard_train pairs (indep-FE-motivated, cross-family judged).
# Selection: LOFO OOF (same as xf_final) with the LOCKED r4 τ_s rule
#   (OOF sel>=0.976 AND sens_prec>=0.997 → max cov).
# Also report old-train 5-fold CV. Never uses old-339 / indep / confirm_r4 for selection.
# usage: XF_TAG=r4H python3 out/replace_audit/r4A/train_r4.py
import os, sys, json, numpy as np, collections, hashlib, time
os.chdir('/workspace/maclaw_reranker')
TAG = os.environ.get('XF_TAG', 'r4H'); OLDW = int(os.environ.get('XF_OLDW', '5'))
src = open('scripts/xf_train.py').read(); exec(src.split("RES = dict(")[0])
# Load hard_train embeddings
HARD = [json.loads(l) for l in open('out/replace_audit/r4A/hard_train.jsonl')]
for l in open('out/replace_audit/r4A/emb_hard_qwen3go.jsonl'):
    r = json.loads(l); v = np.array(r['vec'], np.float32); V[r['id']] = v / np.linalg.norm(v)
HARD = [r for r in HARD if r['id'] in V]
# Ensure hard items look like train rows
for r in HARD:
    r.setdefault('fam', 'r4hard'); r.setdefault('split', 'train')
print(f'[{TAG}] old_tr={len(old_tr)} x{OLDW} fix={len(FIX)} xf={len(XF)} hard={len(HARD)}', flush=True)
assert not ({r['id'] for r in HARD} & TEST_IDS)
assert not ({r['text'].strip() for r in HARD} & TEST_TXT)
# LOFO OOF over XF families (hard pairs included in every train fold)
def lofo_oof():
    Z, rows_, P_lab = [], [], None
    for F in FAMS:
        test = [r for r in XF if r['fam'] == F]; others = [r for r in XF if r['fam'] != F]
        h = Head(old_tr * OLDW + FIX + others + HARD); assert P_lab is None or h.labels == P_lab; P_lab = h.labels
        Z.append(h.logits(test)); rows_ += test; print(f'oof {F}: n={len(test)}', flush=True)
    return np.concatenate(Z), rows_, P_lab
Z, R, LAB = lofo_oof()
y = np.array([LAB.index(r['intent']) for r in R]); Tm = T.fit_temp(Z, y)
P = T.softmax(Z / Tm); conf = P.max(1); pred = [LAB[j] for j in P.argmax(1)]
corr = np.array([ok(r['intent'], p) for r, p in zip(R, pred)]); isS = np.array([p in SENS for p in pred])
print(f'pooled OOF n={len(R)} top1={corr.mean():.3f} T={Tm:.3f}', flush=True)
def at(tau, tau_s):
    need = np.where(isS, max(tau, tau_s), tau); a = conf >= need
    fe = int(sum(1 for r, p, k in zip(R, pred, a) if k and p in SENS and p != r['intent']))
    sa = a & isS
    return dict(tau=round(float(tau), 3), tau_s=round(float(tau_s), 3), cov=float(a.mean()),
                sel=float(corr[a].mean()) if a.any() else float('nan'),
                sensFE=fe, sens_prec=float(corr[sa].mean()) if sa.any() else float('nan'), n=len(R))
sweep = [at(tau, ts) for ts in (0.95, 0.97, 0.98, 0.99) for tau in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99)]
# LOCKED r4 rule
cands = [d for d in sweep if d['sel'] >= 0.976 and d['sens_prec'] >= 0.997]
best = max(cands, key=lambda d: (d['cov'], -d['tau_s'], -d['tau'])) if cands else None
print('R4 RULE SELECTED:', best, flush=True)
tau, tau_s = best['tau'], best['tau_s']
# old-train 5-fold CV (reporting / secondary gate; never old-339)
def old_cv():
    rng = np.random.RandomState(0); idx = rng.permutation(len(old_tr)); folds = np.array_split(idx, 5)
    Zc, Rc = [], []
    for f in folds:
        te = [old_tr[i] for i in f]; tr = [old_tr[i] for i in np.setdiff1d(idx, f)]
        h = Head(tr * OLDW + FIX + XF + HARD)
        Zc.append(h.logits(te)); Rc += te
    return np.concatenate(Zc), Rc
Zc, Rc = old_cv()
Pc = T.softmax(Zc / Tm); cc = Pc.max(1); prc = [LAB[j] for j in Pc.argmax(1)]
corrc = np.array([ok(r['intent'], p) for r, p in zip(Rc, prc)]); isSc = np.array([p in SENS for p in prc])
need = np.where(isSc, max(tau, tau_s), tau); ac = cc >= need
fe_cv = int(sum(1 for r, p, k in zip(Rc, prc, ac) if k and p in SENS and p != r['intent']))
print(f'old-train CV: n={len(Rc)} top1={corrc.mean():.3f} cov={ac.mean():.3f} sel={corrc[ac].mean():.3f} sensFE={fe_cv}', flush=True)
# Final head
h = Head(old_tr * OLDW + FIX + XF + HARD); assert h.labels == LAB; h.T, h.tau, h.tau_s = float(Tm), float(tau), float(tau_s)
di, do_ = evaluate(h, IND), evaluate(h, old_te)
print('REPORT indep839:', fmt(di), flush=True)
print('REPORT old339  :', fmt(do_), flush=True)
out = dict(labels=h.labels, temperature=float(Tm), tau=float(tau), tau_s=float(tau_s), l2norm=True, sensitive=sorted(SENS),
           embed_role='classification', embedder_model_id='Qwen3 Embedding 0.6b:1024:prompt-v1:qwen3-instruct-v1',
           label_table_version='intent-defs-50-v1 (data/intent_defs.json, 49 labels incl. unknown)',
           calibration=f'r4: T on LOFO OOF; tau/tau_s by LOCKED r4 rule (sel>=0.976 AND sens_prec>=0.997 → max cov) on OOF n={len(R)}; hard={len(HARD)}',
           oof_tau_sweep=sweep, train_counts=dict(old_train=len(old_tr), old_train_weight=OLDW, fix=len(FIX), xf=len(XF), hard=len(HARD)),
           note=f'r4H: r3W5F recipe + {len(HARD)} hard pairs (indep-FE-motivated, Llama/Nemotron judged); no indep/old-test/confirm in train or selection',
           w1=h.w1.T.tolist(), b1=h.b1.tolist(), w2=h.w2.T.tolist(), b2=h.b2.tolist())
path = f'out/replace_audit/r4A/head_mlp_qwen3go_xf_{TAG}.json'
json.dump(out, open(path, 'w'))
print('wrote', path, 'sha', hashlib.sha256(open(path,'rb').read()).hexdigest(), flush=True)
json.dump(dict(oof_selected=best, old_cv=dict(top1=float(corrc.mean()), cov=float(ac.mean()), sel=float(corrc[ac].mean()), sensFE=fe_cv),
               report_indep=di, report_old339=do_), open(f'out/replace_audit/r4A/{TAG}_summary.json','w'), indent=1, default=float)
