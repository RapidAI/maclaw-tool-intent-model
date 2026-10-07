import json, sys, numpy as np
rows = [json.loads(l) for l in open('data/intent_all.jsonl')]
te = [r for r in rows if r['split'] == 'test']
meta = json.load(open('data/intent_meta.json')); SENS = set(meta['sensitive']); SAFEU = set(meta['safe_unknown'])
h = json.load(open(sys.argv[1])); P = np.load(sys.argv[1] + '.test_probs.npy'); L = h['labels']
def ok(g, p): return p == g or (g == 'unknown' and p in SAFEU)
print("tau  tau_s  coverage  sel_acc  wrong_accepted  sens_false_exposed  sens_recall(gold sensitive accepted correctly)")
for tau in (0.5, 0.6, 0.7, h['tau'], 0.85, 0.9):
    for tau_s in (tau, 0.9, 0.95):
        if tau_s < tau: continue
        acc = wrong = fexp = 0; n_acc = 0; sens_gold = sens_hit = 0
        for r, p in zip(te, P):
            j = int(p.argmax()); lab = L[j]; c = p[j]
            need = tau_s if lab in SENS else tau
            accepted = c >= need
            if r['intent'] in SENS:
                sens_gold += 1; sens_hit += accepted and lab == r['intent']
            if accepted:
                n_acc += 1; good = ok(r['intent'], lab); acc += good; wrong += not good
                fexp += (lab in SENS and lab != r['intent'])
        print(f"{tau:.2f} {tau_s:.2f}  {n_acc/len(te):.3f}  {acc/max(n_acc,1):.3f}  {wrong}  {fexp}  {sens_hit/sens_gold:.3f}")
print("\nsensitive false exposures at default tau:")
for r, p in zip(te, P):
    j = int(p.argmax()); lab = L[j]
    if p[j] >= h['tau'] and lab in SENS and lab != r['intent']:
        print(f"  {r['text']!r} gold={r['intent']} pred={lab} p={p[j]:.2f}")
print("\nuncertain (below tau) count:", sum(p.max() < h['tau'] for p in P))
