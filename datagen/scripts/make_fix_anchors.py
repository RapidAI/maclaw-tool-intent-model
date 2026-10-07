"""Cheap fix: hand-written diverse contrastive training sentences for the dominant confusion families.
Written from label definitions + confusion-pair names only (NOT paraphrased from indep test items). Training-only data."""
import json
S = {
 'shell_command': [
  '在我这台电脑上看看哪个进程占着 3000 端口', '本机跑一下 lsof -i :5432 看看谁在监听', '我本地 terminal 里执行 docker ps，把容器列表给我',
  '用 curl 从我这台机器请求一下 https://api.example.com/health 看看返回码', '在本机 ping 一下公司网关 10.0.0.1 看延迟', '本地跑 traceroute 到 github.com 看在哪一跳卡住',
  'run `du -sh ~/Downloads/*` on this mac and tell me what is taking space', 'execute ./build.sh in my local project folder', 'on my laptop, run kubectl get pods against the staging context',
  '帮我在本机终端执行 scp 把服务器上的 log 拉到我本地 Downloads（命令在本地跑）', '本地执行一下 brew upgrade', '我的 mac 上 top 看下 cpu 谁最高',
  'zai benji pao yixia npm run test 看看挂没挂', 'my disk is almost full, run df -h locally and summarize', '先 cd 到 ~/work/api，再执行 make lint，最后把输出贴给我',
  '在这台机器上用 nslookup 查一下 example.com 解析到哪个 IP'],
 'ssh': [
  'ssh 到 192.168.1.20 上看一下 nginx 的 error log', '登录生产服务器 prod-01，重启一下 api 服务', '连上那台云主机，在远端执行 df -h',
  'log into the staging box as deploy and tail the app log there', '去远程机器 10.1.2.3 上把 docker 容器 web 重启一下', '帮我上服务器看看 mysql 进程还在不在（远端）',
  'connect to my VPS and check which process is listening on 443 over there', '在跳板机后面那台 db 主机上跑一下 free -m'],
 'screenshot': [
  '截个屏给我看看现在桌面', '把我第二块显示器截一张图', 'grab a screenshot of my whole screen right now', '帮我截一下当前屏幕，看看那个报错弹窗',
  'jie ge ping 发我', 'take a picture of what is on my monitor at the moment', '屏幕右上角那块区域截个图', '我现在屏幕上是什么样子？截图看看',
  'capture the screen and send it to me here', '先别操作，单纯截一张屏幕图'],
 'computer_use': [
  '帮我在桌面上那个软件里点一下"导出"按钮', '用鼠标把这个窗口拖到右边屏幕', '在打开的微信窗口里输入"收到"并回车',
  'click the OK button on the dialog that is open on my desktop', '在 Photoshop 里帮我点菜单"图像-调整大小"', '把这个应用窗口最小化，然后切到 Finder'],
 'unknown': [
  '哈哈今天好累啊', '你吃饭了吗', '谢谢你啊，辛苦了', 'good morning! how are you doing', '讲个笑话听听', '今天心情不太好，随便聊聊',
  '你觉得周末去爬山还是看电影好', 'lol that was funny', '晚安', '你是男生还是女生', '刚才那个回答挺好的，赞一个', '我今天写了一天 bug 好崩溃，吐槽一下',
  'thanks, that is all for now', '最近天气真冷', '嗯嗯好的', '周五了终于要放假了 hhh'],
 'document_open': [
  '把桌面上的 合同.pdf 打开给我看', '用 Word 打开 ~/Documents/周报.docx', 'open budget.xlsx with the default app', '打开下载文件夹里那个 PPT，我要自己翻',
  '帮我把这份 PDF 用系统预览打开', 'just open the report file on my desktop, I will read it myself', '打开 D 盘那个 方案v2.docx', '把昨天那份 Excel 打开就行，不用帮我读'],
 'document_read': [
  '帮我读一下我刚发的 PDF，总结三点', '这个 Word 附件里第二章讲了什么', 'read the attached contract and list the payment terms', '看看这份表格里哪几行数据异常'],
 'app_launch': [
  '打开计算器', '帮我启动 VS Code', 'open Spotify', '启动一下微信', '打开这个文件夹 ~/Projects', 'launch the terminal app',
  '把 https://example.com 用默认浏览器打开就行', '开一下钉钉', 'start Slack for me', '打开系统设置'],
 'database': [
  '查一下 mysql 里 orders 表最近 10 条', '连上 postgres 看看 public schema 下有哪些表', '在 rapidbi 库里跑一句 select count(*) from users',
  'show me the columns of the invoices table in our SQL Server', '把 Excel 数据源里 Sales sheet 当表预览一下', '看看 finance 库里报销记录表的结构'],
 'business_data': [
  '帮我提交一张差旅报销单', '继续填我昨天没填完的采购申请', '发起一个请假申请，下周一到周三', 'approve the pending expense report from Li'],
 'audio_deliver': [
  '把这段话用语音发到群里', '用语音消息告诉张三我晚点到', 'send a voice note to the team saying the build is green', '把这条通知念出来并以语音发给大家',
  '直接发条语音到当前会话：会议改到三点', '把今天的总结转成语音消息发群', 'speak this and send as audio to the chat now', '生成一段语音发给客户群，内容是放假通知'],
 'knowledge_write': [
  '把这篇文章收进知识库，以后能搜到', '这个网页链接帮我存到外脑里', '把 ~/notes 目录下的 md 全部导入本地知识库', 'save this PDF into the knowledge base for future retrieval',
  '这段会议纪要入库，以后查得到', '把这个 URL 的内容抓下来存进知识库', 'add these three docs to my knowledge base', '把我选的这些文件录入外脑'],
 'memory_manage': [
  '记住我喜欢用 tab 缩进', '你还记得我说过我的服务器用户名吗？查一下记忆', '把我住址那条记忆删掉', 'remember that I prefer short answers',
  '更新一下记忆：我现在负责前端', '你都记了我哪些偏好，列出来'],
}
with open('data/fix_anchors.jsonl', 'w') as f:
    k = 0
    for lab, xs in S.items():
        for t in xs:
            f.write(json.dumps({'id': f'fix-{k:04d}', 'text': t, 'intent': lab, 'split': 'train', 'src': 'fix'}, ensure_ascii=False) + '\n'); k += 1
print(k)
