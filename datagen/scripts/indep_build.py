"""Step 4: merge blind judge labels -> agree set (data/indep_test.jsonl) + disagreement bucket (data/indep_ambiguous.jsonl)."""
import json, collections, random
SAFEU = set(json.load(open('data/intent_meta.json'))['safe_unknown'])
C = {json.loads(l)['id']: json.loads(l) for l in open('data/indep_filtered.jsonl')}
J = {}
for l in open('data/indep_judge_raw_cohere.jsonl'):
    for r in json.loads(l)['results']: J[r['id']] = r
agree, amb, st = [], [], collections.Counter()
for i, c in C.items():
    if i not in J: st['unjudged'] += 1; continue
    j = J[i]; c['judge'] = j['label']; c['judge_second'] = j['second']; c['judge_conf'] = j['conf']
    same = j['label'] == c['intent'] or (c['intent'] == 'unknown' and j['label'] in SAFEU)
    if same: agree.append(c); st[f'agree_{c["gen"]}'] += 1; st[f'agree_{c["kind"]}'] += 1
    else: amb.append(c); st[f'disagree_{c["gen"]}'] += 1; st[f'disagree_{c["kind"]}'] += 1; st['disagree_but_second_ok'] += j['second'] == c['intent']
for name, xs in (('data/indep_test.jsonl', agree), ('data/indep_ambiguous.jsonl', amb)):
    with open(name, 'w') as f:
        for c in xs: f.write(json.dumps(c, ensure_ascii=False) + '\n')
st['agree'] = len(agree); st['ambiguous'] = len(amb)
print(dict(st))
print('per lang', collections.Counter((c['gen'], c['lang']) for c in agree))
print('per label min/max', min(collections.Counter(c['intent'] for c in agree).values()), max(collections.Counter(c['intent'] for c in agree).values()))
print('top judge disagreements intended->judge', collections.Counter((c['intent'], c['judge']) for c in amb).most_common(15))
random.seed(7); spot = random.sample(agree, min(30, len(agree)))
json.dump([{k: c[k] for k in ('id', 'text', 'intent', 'judge', 'gen', 'kind', 'lang')} for c in spot], open('data/indep_spotcheck30.json', 'w'), ensure_ascii=False, indent=1)
json.dump(dict(st), open('out/indep_qc_step3.json', 'w'), indent=1)
