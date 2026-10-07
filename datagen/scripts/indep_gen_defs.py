import json
D = {x['label']: x for x in json.load(open('data/intent_defs.json'))['defs']}
TOOLS = {x.get('function', x)['name']: (x.get('function', x).get('description') or '') for x in json.load(open('data/core_tool_defs.json'))}
UNKNOWN_DESC = "闲聊、寒暄、情绪表达、对 AI 的随口提问或与本机桌面 agent 的任何任务都无关的消息（没有可执行的任务）。"
def desc(label):
    if label == 'unknown': return f"标签: unknown\n说明: {UNKNOWN_DESC}"
    x = D[label]; s = f"标签: {label}\n领域: {x['domain']}\n官方说明: {x['tree_text'][:1200]}"
    td = [f"  - {t}: {TOOLS[t][:300]}" for t in x['tools'] if t in TOOLS]
    if td: s += "\n相关工具（官方描述）:\n" + "\n".join(td)
    return s
_SYS = ("You write realistic test messages that real users would type to a desktop AI agent called MaClaw "
       "(it can code, run shell/ssh, operate files, browser, desktop GUI, office docs, knowledge base, schedules, IM delivery, audio, etc.). "
       "Output ONLY a JSON object: {\"items\":[{\"text\":...,\"lang\":\"zh|mix|en\",\"style\":...}]}.")
STYLES = "口语化(colloquial) / 极简命令(terse) / 带错别字或拼音缩写(typo) / 长上下文先交代背景再提需求(long_context) / 间接委婉(indirect) / 多步骤但以该意图为主(multi_step)"
def task_label(label, n=8):
    return (f"{desc(label)}\n\n请写 {n} 条真实用户会发给 MaClaw 的消息，这些消息的**主要意图必须明确属于上述标签**。"
            f"语言分布：{n*3//8} 条纯中文(zh)、{n*3//8} 条中英混杂(mix，中文为主夹英文术语/命令/文件名)、其余纯英文(en)。"
            f"风格要多样，尽量覆盖：{STYLES}。不要彼此雷同，不要复述说明里的原句，不要出现标签名本身。"
            "具体化（文件名、路径、服务器、人名、时间等可以虚构）。")
def task_pair(a, b, n=3):
    return (f"目标标签 A：\n{desc(a)}\n\n易混标签 B：\n{desc(b)}\n\n请写 {n} 条真实用户消息，它们的正确意图是 **A**，但表面上很容易被误判为 B"
            f"（处在两者边界上，但按官方说明仔细判断应归 A）。语言混合 zh/mix/en 各至少一条，风格多样（{STYLES}）。不要出现标签名。")
PAIRS = [('screenshot', 'computer_use'), ('computer_use', 'screenshot'), ('screenshot', 'document_delivery'), ('document_delivery', 'screenshot'),
         ('config_manage', 'file_write'), ('file_write', 'config_manage'), ('shell_command', 'app_launch'), ('app_launch', 'shell_command'),
         ('file_delete', 'ssh'), ('ssh', 'file_delete'), ('ssh', 'shell_command'), ('shell_command', 'ssh'),
         ('delegate_task', 'audit_read'), ('audit_read', 'delegate_task'), ('delegate_task', 'task_track'), ('browser', 'app_launch'),
         ('browser', 'web_fetch'), ('knowledge_write', 'memory_manage'), ('memory_manage', 'knowledge_write'), ('schedule_dispatch', 'schedule_manage'),
         ('database', 'business_data'), ('git_mutate', 'git_inspect'), ('file_delete', 'file_write'), ('document_delivery', 'attachment_delivery')]
