"""Section 17: multi-family TRAINING data generation (same schema / info policy as indep_gen.py: generator sees only label name +
official description + official tool descriptions; never anchors / training / test sentences). Prompt is varied per task
(persona, style emphasis, language mix) so the data isn't one author's style.
usage: xf_gen.py <spec[,fallback]> <tag> <round> <n_per_label> [conc] [labels_csv|all] [hard 0/1]"""
import json, sys, os, random, threading, concurrent.futures as cf, time
sys.path.insert(0, 'scripts'); import llmc
import indep_gen_defs as G  # desc(), PAIRS, STYLES, D
GENS = sys.argv[1].split(','); TAG = sys.argv[2]; RND = sys.argv[3]; N = int(sys.argv[4])
CONC = int(sys.argv[5]) if len(sys.argv) > 5 else 3
LABS = sys.argv[6] if len(sys.argv) > 6 else 'all'; HARD = int(sys.argv[7]) if len(sys.argv) > 7 else 1
REPS = int(os.environ.get('XF_REPS', '1'))
OUT = f'data/train_xfamily/raw_{TAG}.jsonl'
PERSONAS = ['一名后端程序员', '一名运维工程师', '一名财务/行政办公室职员', '一名在校大学生', '一位不太懂电脑的中年用户', '一名产品经理',
            '一名数据分析师', '一名自由职业设计师', '一名销售', '一名研究生（写论文）', '一名小公司老板', '一名测试工程师']
EMPH = ['口语化、随手打字', '极简命令式', '有错别字或拼音缩写', '先讲一段背景再提需求', '委婉间接、不直接说要做什么', '多步骤但以该意图为主',
        '语气着急', '用很礼貌的书面语', '夹杂英文命令/文件名/术语', '像在 IM 里发的短消息']
SYS = ("You write realistic messages that real users would type to a desktop AI agent called MaClaw "
       "(it can code, run shell/ssh, operate files, browser, desktop GUI, office docs, knowledge base, schedules, IM delivery, audio, etc.). "
       "Output ONLY a JSON object: {\"items\":[{\"text\":...,\"lang\":\"zh|mix|en\",\"style\":...}]}.")
def task_label(label, n, rng):
    p = rng.choice(PERSONAS); e = rng.sample(EMPH, 3); nz = max(1, round(n * 0.45)); nm = max(1, round(n * 0.35)); ne = max(0, n - nz - nm)
    return (f"{G.desc(label)}\n\n假设你是{p}。请写 {n} 条你会发给 MaClaw 的真实消息，**主要意图必须明确属于上述标签**（按官方说明判断）。"
            f"语言：{nz} 条纯中文(zh)、{nm} 条中英混杂(mix)、{ne} 条英文(en)。风格侧重：{'、'.join(e)}，并保持多样。"
            "不要复述说明原句，不要出现标签名，不要彼此雷同；可以虚构具体的文件名、路径、服务器、人名、时间。")
def task_pair(a, b, n, rng):
    p = rng.choice(PERSONAS)
    return (f"目标标签 A：\n{G.desc(a)}\n\n易混标签 B：\n{G.desc(b)}\n\n假设你是{p}。请写 {n} 条真实用户消息，正确意图是 **A**，但表面上容易被误判为 B"
            f"（按官方说明仔细判断应归 A）。zh/mix/en 都要有，中文和中英混杂为主，风格多样（{G.STYLES}）。不要出现标签名。")
labels = sorted(l for l in G.D if l not in ('ambiguous',)) + ['unknown'] if 'unknown' not in G.D else sorted(l for l in G.D if l != 'ambiguous')
if LABS != 'all': labels = LABS.split(',')
tasks = []
for rep in range(REPS):
    for l in labels:
        tasks.append(('label', l, rep, N * (2 if l == 'unknown' else 1)))
    if HARD:
        for a, b in G.PAIRS: tasks.append(('hard', f'{a}|{b}', rep, 3))
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l); done.add((r['round'], r['kind'], r['key'], r.get('rep', 0)))
lock = threading.Lock(); dead = {}
def run(t):
    kind, key, rep, n = t
    if (RND, kind, key, rep) in done: return f'skip {key}'
    rng = random.Random(hash((TAG, RND, key, rep)) & 0xffffffff)
    prompt = task_label(key, n, rng) if kind == 'label' else task_pair(*key.split('|'), n, rng)
    err = ''
    for attempt in range(5):
        spec = [g for g in GENS if dead.get(g, 0) < time.time()] or GENS
        spec = spec[0]
        try:
            c, m, dt = llmc.chat(spec, [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': prompt}], temperature=1.0, max_tokens=4000, timeout=180, retries=2)
            i, j = c.find('{'), c.rfind('}'); items = json.loads(c[i:j + 1])['items']
            with lock, open(OUT, 'a') as f:
                f.write(json.dumps({'round': RND, 'kind': kind, 'key': key, 'rep': rep, 'gen': spec, 'model_resp': m, 'sec': round(dt, 1), 'items': items}, ensure_ascii=False) + '\n')
            return f'ok {key} {len(items)} {dt:.0f}s'
        except Exception as e:
            err = str(e)[:140]
            if 'QUOTA' in err or '429' in err: dead[spec] = time.time() + 60
    return f'FAIL {key} {err}'
with cf.ThreadPoolExecutor(CONC) as ex:
    for r in ex.map(run, tasks): print(r, flush=True)
