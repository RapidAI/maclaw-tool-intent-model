import json, re, sys
sys.path.insert(0, 'scripts')
from intent_data_src import D

HARD = [  # confusable boundary cases (test only)
 ("打开桌面上的 PPT 看一下", "document_open"), ("删除服务器 10.0.0.3 上的 /tmp/a.log", "ssh"),
 ("把关于我旧公司的记忆删掉", "memory_manage"), ("删除知识库里关于报销的来源", "knowledge_admin"),
 ("删除每天 9 点的那个定时任务", "schedule_manage"), ("每天 9 点把日报发到群里", "schedule_dispatch"),
 ("在浏览器里给这个网页截个图", "browser"), ("截一下我电脑桌面", "screenshot"),
 ("把我刚上传的附件发给张三", "document_delivery"), ("把我刚上传的附件发回给我", "attachment_delivery"),
 ("把这段文字读出来", "audio_synthesize"), ("把这段文字用语音发到群里", "audio_deliver"),
 ("开始录音", "audio_record"), ("把录音文件转成文字", "audio_transcribe"),
 ("帮我搜一下 https 证书怎么配", "search"), ("读一下 https://go.dev/doc/effective_go", "web_fetch"),
 ("下载 https://go.dev/dl/go1.25.5.src.tar.gz", "file_download"), ("今天上海天气", "live_data"),
 ("把今天上海天气画成图", "live_data_visual"), ("把今天上海天气生成 PDF", "document_generate"),
 ("帮我写一份研究报告", "workflow_task"), ("把这份报告导出为 PDF", "document_generate"),
 ("基于这份文档做个 PPT", "office"), ("读一下这份文档", "document_read"),
 ("执行 rm -rf build 目录", "shell_command"), ("把 build.log 文件删除", "file_delete"),
 ("看看 git 改了什么", "git_inspect"), ("把改动提交并 push", "git_mutate"),
 ("查一下知识库里的报销规定", "knowledge_read"), ("把这份报销规定存到知识库", "knowledge_write"),
 ("帮我报销这张发票", "business_data"), ("查询 MySQL 里的发票表", "database"),
 ("打开微信", "app_launch"), ("打开微信给张三发消息", "computer_use"),
 ("列出编码会话", "session_manage"), ("列出会话模板", "template_manage"),
 ("加个待办：周五交周报", "task_track"), ("周五下午提醒我交周报", "schedule_manage"),
 ("切换模型到 deepseek-v3", "config_manage"), ("记住我喜欢 deepseek", "memory_manage"),
 ("登录服务器看日志", "ssh"), ("看看本地 app.log 日志", "file_read"),
 ("修复登录 bug", "bug_fix"), ("重构登录模块", "maintenance"), ("新增登录模块", "coding"),
 ("继续", "continuation"), ("让子 agent 并行修这三个 bug", "delegate_task"),
]
# Labels whose capability is privileged / side-effectful: a false positive here
# exposes a sensitive tool surface (fail-closed must not happen by mistake).
SENSITIVE = {"ssh", "browser", "computer_use", "screenshot", "audio_record", "file_delete",
             "git_mutate", "shell_command", "document_delivery", "schedule_dispatch", "audio_deliver",
             "business_data", "config_manage", "knowledge_write", "knowledge_admin", "database",
             "memory_manage", "delegate_task"}
SAFE_FOR_UNKNOWN = {"unknown", "ambiguous", "non_coding", "continuation"}

def lang(t):
    cjk = bool(re.search(r'[\u4e00-\u9fff]', t))
    asc = re.findall(r'[A-Za-z]{3,}', t)
    if cjk and asc: return "mix"
    return "zh" if cjk else "en"

defs = json.load(open('data/intent_defs.json'))['defs']
anchors = {(x['label'], t) for x in defs for t in (x['embed_texts'] or [])}
anchor_texts = {t.strip() for _, t in anchors}
rows = []
for lab, t in sorted(anchors):
    rows.append(dict(text=t, intent=lab, split="train", src="anchor"))
for lab, (tr, te) in D.items():
    for t in tr:
        rows.append(dict(text=t, intent=lab, split="train", src="synthetic"))
    for t in te:
        if t.strip() in anchor_texts: continue
        rows.append(dict(text=t, intent=lab, split="test", src="heldout"))
for t, lab in HARD:
    if t.strip() in anchor_texts: continue
    rows.append(dict(text=t, intent=lab, split="test", src="hard"))
for i, r in enumerate(rows):
    r["id"] = f"{r['split']}-{i:04d}"; r["lang"] = lang(r["text"])
json.dump(dict(sensitive=sorted(SENSITIVE), safe_unknown=sorted(SAFE_FOR_UNKNOWN)), open('data/intent_meta.json', 'w'))
with open('data/intent_all.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
print(Counter((r['split'], r['src']) for r in rows)); print(Counter((r['split'], r['lang']) for r in rows))
