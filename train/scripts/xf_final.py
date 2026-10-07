"""§21 final head: T / tau / tau_s fitted on POOLED LEAVE-ONE-FAMILY-OUT predictions (each xf family scored by a head trained
without it: old train + fix + all other families), then the exported head is trained on old train + fix + ALL xf.
Old-339 and indep-839 are never used for training or selection (reported only).
usage: XF_TAG=r3 python scripts/xf_final.py"""
import os, sys, json, numpy as np, collections, hashlib, time
src = open('scripts/xf_train.py').read(); exec(src.split("RES = dict(")[0])
exec("tools = sorted" + src.split("tools = sorted", 1)[1].split("old_tool_tr =")[0])
SENSL = SENS
OLDW = int(os.environ.get("XF_OLDW", "1"))  # §23: up-weight (duplicate) the original old-train rows
def lofo_oof():
    Z, Y, P_lab, fams, rows_ = [], [], None, [], []
    for F in FAMS:
        test = [r for r in XF if r['fam'] == F]; others = [r for r in XF if r['fam'] != F]
        h = Head(old_tr * OLDW + FIX + others); assert P_lab is None or h.labels == P_lab; P_lab = h.labels
        Z.append(h.logits(test)); rows_ += test; print(f'oof {F}: n={len(test)} train={len(old_tr)+len(FIX)+len(others)}', flush=True)
    return np.concatenate(Z), rows_, P_lab
Z, R, LAB = lofo_oof()
y = np.array([LAB.index(r['intent']) for r in R]); Tm = T.fit_temp(Z, y)
P = T.softmax(Z / Tm); conf = P.max(1); pred = [LAB[j] for j in P.argmax(1)]
corr = np.array([ok(r['intent'], p) for r, p in zip(R, pred)]); isS = np.array([p in SENSL for p in pred])
print(f'pooled OOF n={len(R)} top1={corr.mean():.3f} T={Tm:.3f} ECE={ece(conf, corr):.3f}', flush=True)
def at(tau, tau_s, conf=conf, corr=corr, isS=isS, R=R, pred=pred):
    need = np.where(isS, max(tau, tau_s), tau); a = conf >= need
    fe = int(sum(1 for r, p, k in zip(R, pred, a) if k and p in SENSL and p != r['intent']))
    sa = a & isS
    return dict(tau=round(float(tau), 3), tau_s=round(float(tau_s), 3), cov=float(a.mean()), sel=float(corr[a].mean()) if a.any() else float('nan'),
                sensFE=fe, sensFE_pct=fe / len(R), sens_prec=float(corr[sa].mean()) if sa.any() else float('nan'), n=len(R))
sweep = []
for tau_s in (0.95, 0.97, 0.98, 0.99):
    for tau in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99):
        d = at(tau, tau_s); sweep.append(d)
        print(f"  tau={tau:.2f} tau_s={tau_s:.2f} | cov={d['cov']:.3f} sel={d['sel']:.4f} sensFE={d['sensFE']} ({100*d['sensFE_pct']:.2f}%) sens_prec={d['sens_prec']:.4f}", flush=True)
# selection on OOF only: tau = smallest grid value (0.01 steps) with OOF accepted accuracy >= 0.976;
# tau_s = smallest >= max(tau, 0.95) with OOF accepted-sensitive precision >= 0.99
tau = 0.99
for t in np.arange(0.30, 0.995, 0.01):
    if at(t, 0.0)['sel'] >= 0.976: tau = float(round(t, 2)); break
tau_s = 0.99
for t in np.arange(max(tau, 0.95), 0.995, 0.01):
    if at(tau, t)['sens_prec'] >= 0.99: tau_s = float(round(t, 2)); break
chosen = at(tau, tau_s); print('CHOSEN (OOF only):', chosen, flush=True)
# per-family at chosen
perfam = {}
for F in FAMS:
    k = np.array([r['fam'] == F for r in R])
    perfam[F] = at(tau, tau_s, conf[k], corr[k], isS[k], [r for r, kk in zip(R, k) if kk], [p for p, kk in zip(pred, k) if kk])
    print(f'  {F}: top1={corr[k].mean():.3f}', perfam[F], flush=True)
# final head on everything (old train + fix + all xf)
h = Head(old_tr * OLDW + FIX + XF); assert h.labels == LAB; h.T, h.tau, h.tau_s = Tm, tau, tau_s
di = evaluate(h, IND); do = evaluate(h, old_te)
print('FINAL head (report only, not selection) indep839:', fmt(di), flush=True)
print('FINAL head (report only, not selection) old339  :', fmt(do), flush=True)
rep = {}
for name, items in (('indep839', IND), ('old339', old_te)):
    Pz = T.softmax(h.logits(items) / Tm); c2 = Pz.max(1); p2 = [LAB[j] for j in Pz.argmax(1)]
    k2 = np.array([ok(r['intent'], p) for r, p in zip(items, p2)]); s2 = np.array([p in SENSL for p in p2])
    rep[name] = [at(t, ts, c2, k2, s2, items, p2) for t, ts in ((0.5, 0.95), (0.7, 0.95), (0.8, 0.95), (0.9, 0.97), (0.95, 0.99), (tau, tau_s))]
    for d in rep[name]: print(f'  [{name}] tau={d["tau"]} tau_s={d["tau_s"]} cov={d["cov"]:.3f} sel={d["sel"]:.4f} sensFE={d["sensFE"]}', flush=True)
out = dict(labels=h.labels, temperature=float(Tm), tau=tau, tau_s=tau_s, l2norm=True, sensitive=sorted(SENSL), embed_role='classification',
           embedder_model_id='Qwen3 Embedding 0.6b:1024:prompt-v1:qwen3-instruct-v1',
           label_table_version='intent-defs-50-v1 (data/intent_defs.json, 49 labels incl. unknown)',
           calibration='T/tau/tau_s fitted on pooled leave-one-family-out predictions over %d xf items (families %s); tau = min with OOF accepted acc >= 0.976; tau_s = min >= max(tau,0.95) with OOF accepted-sensitive precision >= 0.99' % (len(R), ','.join(FAMS)),
           oof_tau_sweep=sweep, train_counts=dict(old_train=len(old_tr), old_train_weight=OLDW, fix=len(FIX), xf=len(XF), xf_by_family=dict(collections.Counter(r['fam'] for r in XF))),
           note=f'FINAL §21 ({TAG}): MLP 1024->256 (sklearn, seed 0) on pure-Go Qwen3 vectors; no indep/old-test items in training or selection',
           w1=h.w1.T.tolist(), b1=h.b1.tolist(), w2=h.w2.T.tolist(), b2=h.b2.tolist())
json.dump(out, open('heads/head_mlp_qwen3go_xf.json', 'w'))
W, b = tool_fit(old_tr + XF)
json.dump(dict(format='maclaw-tool-head-ovr-v1', tools=tools, W=W.tolist(), b=b.tolist(), dims=1024, l2norm=True, embedder_model_id='Qwen3 Embedding 0.6b:1024:prompt-v1:qwen3-instruct-v1',
               embed_role='classification', train=f'old_train {len(old_tr)} + xf {len(XF)} (no indep, no old test)',
               note=f'FINAL §21 ({TAG}): OvR logistic (C=10, balanced) on L2-normalized pure-Go Qwen3 vectors; gold via scripts/tool_gold.py; sigmoid(W x + b)'),
          open('heads/tool_head_ovr_qwen3go_xf.json', 'w'))
ti = tool_eval((W, b), IND); print('FINAL tool head indep839 (clean, no indep in train):', ti, flush=True)
json.dump(dict(tag=TAG, oof_top1=float(corr.mean()), T=float(Tm), tau=tau, tau_s=tau_s, chosen=chosen, perfam=perfam, sweep=sweep, final_indep839=di, final_old339=do,
               report_at_taus=rep, tool_indep839=ti, counts=out['train_counts']), open(f'out/xf/xf_final_{TAG}.json', 'w'), indent=1,
          default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
