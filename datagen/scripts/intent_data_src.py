# Hand-written intent queries for maclaw's 50-label taxonomy (+ "unknown" chit-chat).
# Each label: (train_synthetic, test_heldout). Written independently of maclaw's
# EmbedTexts anchors; exact anchor duplicates are removed by build_intent_data.py.
D = {
"coding": ([
 "帮我用 Go 写一个命令行记账工具", "做一个贪吃蛇小游戏，网页版就行", "用 Python 实现一个爬虫抓取豆瓣电影 Top250",
 "给这个项目加一个用户登录注册功能", "write a REST API in FastAPI for a todo app", "从零搭一个 Vue3 + Vite 的后台管理前端",
 "实现一个 LRU 缓存，要有单元测试", "帮我开发一个微信小程序的点餐页面", "build me a small chrome extension that blocks ads",
 "写个脚本批量把 heic 转成 jpg"],[
 "帮我写一个 Rust 版的 markdown 转 html 小工具", "给后端新增一个导出 CSV 的接口", "做一个俄罗斯方块游戏，用 pygame",
 "implement a websocket chat server in Node.js", "用 Flutter 开发一个天气 App 的界面", "写个 shell 脚本每天自动备份数据库到 S3",
 "新建一个 Spring Boot 项目，实现商品管理 CRUD"]),
"bug_fix": ([
 "程序一启动就 panic: nil pointer dereference，帮我看看", "登录接口返回 500，帮我排查一下", "这个函数在并发下数据错乱，修一下",
 "npm run build 报错 Module not found，怎么修", "fix the segfault when parsing empty input", "页面点击提交没反应，控制台报 undefined",
 "单元测试 TestParse 一直失败，帮我修好", "内存泄漏越跑越大，定位一下原因"],[
 "App 闪退了，日志里是 IndexOutOfBoundsException，帮我修", "这个 SQL 查询结果重复了，代码哪里有问题", "debug why the CI pipeline fails on go test",
 "上传大文件时报 413，帮我解决", "goroutine 死锁了，帮忙找出来修掉", "Python 脚本报 UnicodeDecodeError，修一下", "the login button crashes the app on Android, please fix"]),
"maintenance": ([
 "把这个 2000 行的文件拆分成几个模块", "重构一下 router.go，消除重复代码", "把项目依赖升级到最新版本",
 "优化这个接口的性能，现在太慢了", "refactor this class to use dependency injection", "清理掉项目里没用的代码和注释",
 "把回调风格改成 async/await", "升级 Go 版本到 1.25 并修好兼容问题"],[
 "把这堆 if-else 重构成策略模式", "给代码做一次性能优化，减少内存分配", "migrate the codebase from Python 3.8 to 3.12 idioms",
 "把重复的工具函数抽到公共包里", "清理一下过期的依赖和废弃 API", "rename variables to be more descriptive across the module"]),
"ssh": ([
 "登录 10.0.0.12 看一下 nginx 日志", "ssh 到生产服务器重启一下 redis", "去 192.168.1.20 上看看磁盘还剩多少",
 "check the docker containers running on prod-web-01", "远程服务器上的 java 进程占用 CPU 太高，看看", "到跳板机上把 /var/log/app.log 最后 100 行拉出来",
 "在 staging 机器上执行 systemctl status mysql", "连一下那台阿里云 ECS 看看端口 8080 有没有监听"],[
 "去服务器 172.16.3.5 上重启 tomcat", "ssh root@my-vps 看下内存使用情况", "remote host db-02 的 mysql 慢查询日志帮我看看",
 "把线上那台机器的 nginx 配置 reload 一下", "check uptime and load average on the build server", "登录测试服务器看看 kubelet 的状态", "上那台服务器把 /tmp 下的大文件清一清"]),
"browser": ([
 "打开浏览器登录我的 GitHub 然后点 star", "用浏览器帮我填写这个报名表单", "自动化点一下京东的签到按钮",
 "录制一下我在网页上的操作流程以后回放", "open the website in a browser and click the login button", "在浏览器里打开淘宝搜索机械键盘并截图",
 "帮我在网页上一页页翻，把每页的标题记下来", "用浏览器自动提交这个工单表单"],[
 "打开浏览器进入公司 OA 帮我点审批通过", "在 Chrome 里打开 B 站搜索 Go 教程", "fill out the signup form on that site using the browser",
 "用浏览器模拟登录然后下载对账单页面", "在网页上自动点击下一页直到最后一页", "回放上次录制的浏览器操作"]),
"computer_use": ([
 "打开本机的微信，给文件传输助手发个消息", "帮我在桌面上点开 Excel 软件", "操作一下电脑上的 Photoshop 新建一个画布",
 "click the Start menu and open Settings", "在记事本里输入一段文字", "帮我把桌面上那个窗口最大化",
 "用键盘快捷键关掉当前这个应用窗口", "打开系统设置把屏幕亮度调高"],[
 "打开电脑上的网易云音乐播放日推", "在 Word 软件界面里点一下保存按钮", "use the mouse to drag that window to the second monitor",
 "帮我在本机的钉钉客户端里点开最新消息", "打开计算器算一下 123*456，用界面操作", "在桌面应用里滚动到底部看看"]),
"screenshot": ([
 "截个屏给我看看", "把当前桌面截图发我", "截一下第二块显示器的画面", "take a screenshot of my screen",
 "看看我电脑屏幕上现在显示的是什么，截个图", "截全屏", "帮我截一张当前屏幕的图片", "capture the main display"],[
 "截一下屏幕", "拍一张当前桌面的截图", "screenshot the desktop please", "把主显示器截个图发过来", "截个图看下现在电脑上开了什么", "grab an image of my monitor"]),
"search": ([
 "搜一下 Rust 的异步运行时有哪些", "帮我查一下 transformer 论文原文", "网上找找 Go 泛型的最佳实践文章",
 "search for papers on retrieval augmented generation", "查一下 PostgreSQL 分区表怎么用", "找一些关于 RAG 评测方法的资料",
 "上网搜索一下 llama.cpp 的量化格式说明", "look up how to configure nginx rate limiting"],[
 "帮我搜搜 Kubernetes HPA 的原理", "网上查一下 BERT 和 GPT 的区别", "search for best practices of OCR post-processing",
 "找几篇关于向量数据库选型的博客", "查一下 Python asyncio 的官方文档", "搜索一下 PDF 表格抽取有哪些开源方案"]),
"non_coding": ([
 "把这段中文翻译成日语", "帮我总结一下这篇文章的要点", "把会议记录整理成要点列表", "写一封请假邮件",
 "translate this paragraph into Chinese", "帮我润色一下这段自我介绍", "把模型转换成 GGUF 格式", "写一段产品宣传文案"],[
 "帮我把这段英文翻成中文", "总结一下上面这段对话", "给这份需求写个摘要", "polish my cover letter",
 "帮我写一首关于秋天的诗", "把这个 PyTorch 模型导出成 ONNX", "整理一下这些笔记，按主题分类"]),
"live_data": ([
 "今天北京天气怎么样", "现在美元兑人民币汇率多少", "茅台今天股价多少", "what's the bitcoin price right now",
 "查一下今天的油价", "最新的 NBA 比分", "上海明天会下雨吗", "今天有什么科技新闻"],[
 "深圳现在气温多少度", "查下今天黄金价格", "what's the weather in Tokyo today", "苹果公司最新股价是多少",
 "今天欧元汇率多少", "最近有什么 AI 大新闻", "杭州这周末天气预报"]),
"live_data_visual": ([
 "把今天的天气做成一张图", "画一下特斯拉最近一个月的股价走势图", "把最新汇率做成信息图", "plot bitcoin price trend for the past week",
 "给我一张实时空气质量信息图", "画出茅台近三个月的 K 线图", "把本周气温变化画成折线图", "create a chart of today's gold price"],[
 "画一张英伟达近半年股价走势图", "把北京今天的天气生成一张卡片图片", "make an infographic of current exchange rates",
 "把比特币最近价格画成折线图", "生成一张上证指数今日行情图", "plot the 30-day trend of Apple stock"]),
"current_time": ([
 "现在几点了", "今天几号", "今天星期几", "what time is it now", "今天是农历几月几号", "现在是几月份",
 "距离今天是第几周", "what's today's date"],[
 "现在几点", "今天是周几", "what day is it today", "今天日期是多少", "现在北京时间几点了", "今年是哪一年"]),
"document_delivery": ([
 "把 report.pdf 发到 zhang@example.com", "把这个文件发给王经理", "把刚才生成的表格发到项目群里", "send the slides to bob@corp.com",
 "把这个文件拷贝到 D:\\backup 目录", "把合同发到法务那个会话", "把总结文档发送给张三的微信", "forward this file to the design channel"],[
 "把 data.xlsx 发给李四", "把周报发到 team@company.com 邮箱", "send the generated pdf to alice",
 "把这份文件转发到另一个群", "把这个压缩包传到共享目录 /mnt/share", "把截图发给运维群"]),
"document_generate": ([
 "把上面的内容导出成 PDF", "生成一份 PDF 报告", "把查到的数据做成 PDF", "export this as a pdf",
 "把这段内容排版成 PDF 文件", "把今天的天气信息生成 PDF", "生成个 pdf 给我", "把会议纪要输出为 PDF"],[
 "把这些结果整理成 PDF 发我", "导出 PDF", "render the summary above into a PDF file", "把搜到的资料生成一份 pdf",
 "把这份清单转成 PDF 文档", "make a pdf of these notes"]),
"document_read": ([
 "帮我看看这份合同里有哪些风险条款", "读一下这个 PDF 讲了什么", "这个 Excel 里销售额最高的是哪个月", "summarize the attached document",
 "这份 PPT 的主要内容是什么", "提取一下这份发票里的金额", "这个 Word 文档里提到了哪些人", "what does this pdf say about pricing"],[
 "看一下我发的这个 PDF 的结论", "这份表格里有多少行数据", "read the attached contract and list obligations",
 "帮我提取这份简历里的工作经历", "这个 pptx 一共几页，讲了什么", "附件里的报表利润是多少"]),
"document_open": ([
 "打开桌面上的 report.pdf", "用 Word 打开 合同.docx", "打开 D 盘那个 Excel 文件", "open the quarterly.pptx on my desktop",
 "把下载目录里的说明书 PDF 打开", "打开刚才生成的 PDF", "用默认程序打开这个 xlsx", "open 简历.docx"],[
 "打开桌面上的方案.pptx", "帮我打开 Downloads 里的 invoice.pdf", "open the budget spreadsheet with excel",
 "用系统默认程序打开 readme.pdf", "打开刚导出的那个 Word 文档", "把 ~/Documents/plan.xlsx 打开"]),
"attachment_delivery": ([
 "把我刚上传的文件再发给我", "把这个附件回传到当前聊天", "把我发你的那个 PDF 原样发回来", "send me back the attachment",
 "把这份附件在这里再发一次", "return the uploaded spreadsheet to me", "把刚才那个附件发回给我", "re-send the file I uploaded"],[
 "把我上传的那份合同发回来", "把刚才的附件再传给我一下", "send the attached image back here",
 "把这个上传的 Excel 回传给我", "把我给你的文件原样发我", "give me back the attached pdf in this chat"]),
"business_data": ([
 "我要报销上周出差的费用", "提交一个采购申请", "继续填写我没完成的报销单", "create a leave request for next Monday",
 "录入一张发票到系统里", "帮我发起一个合同审批", "查一下我待审批的单据", "submit a travel reimbursement"],[
 "帮我填一张差旅报销单", "提交请假申请，周五请一天", "record this invoice in the business system",
 "继续我上次保存的采购单", "查询客户 ABC 的合同记录", "审批一下这个费用单"]),
"database": ([
 "查一下 MySQL 里 orders 表有多少行", "连接 PostgreSQL 看看有哪些 schema", "列出数据库里所有的表", "show tables in the sales database",
 "查询 users 表最近注册的 10 个用户", "看看 SQL Server 里 customer 表的结构", "统计一下订单表每个月的销售额", "run select count(*) from logs"],[
 "看看 pg 里 public schema 下有哪些表", "查下数据库 orders 表昨天的订单数", "describe the products table in mysql",
 "连接 Access 数据库看看有哪些表", "从 sales 库里查出销售额前十的商品", "query the inventory table for items out of stock"]),
"office": ([
 "帮我做一份年终总结 PPT", "基于这份文档做个演示文稿", "设计一套产品发布会的幻灯片", "make a slide deck about our Q3 results",
 "把这个 PPT 的风格改得更高级一点", "做一个 10 页的公司介绍 PPT", "生成培训课件 PPT", "create a presentation on AI trends"],[
 "做一份项目汇报的 PPT", "帮我把这份报告做成幻灯片", "design a pitch deck for my startup",
 "把当前这个 PPT 配色优化一下", "制作一个关于碳中和的演示文稿", "生成一套新员工入职培训的 slides"]),
"workflow_task": ([
 "帮我写一份商业计划书", "做一份产品需求文档 PRD", "写一篇关于大模型的研究报告", "帮我写一份投标书",
 "策划一场公司年会活动", "帮我写个专利申请", "write a business plan for a coffee shop", "做一份竞品分析报告"],[
 "帮我出一份新产品的 PRD", "写一份市场调研报告", "帮我准备一份项目投标文件", "draft a research paper on OCR methods",
 "策划一个线下技术沙龙活动方案", "写一份融资商业计划"]),
"continuation": ([
 "继续", "接着来", "开始吧", "go on", "好的开始", "接着干", "go ahead", "继续执行"],[
 "继续吧", "开搞", "proceed", "接着做", "好，开始", "continue"]),
"knowledge_write": ([
 "把这段内容存进知识库", "把这个网址收录到知识库", "把 docs 目录导入知识库", "save this article to the knowledge base",
 "把这几个 PDF 导入知识库", "把这篇文章加到知识库里以后查", "导入这个知识包", "add this url to my knowledge base"],[
 "把这份说明文档录入知识库", "把这个链接保存到知识库", "import the folder ~/notes into the knowledge base",
 "把刚才那段总结存到知识库", "把这些文件加入知识库", "收录这个网页到知识库"]),
"file_read": ([
 "看一下 config.yaml 的内容", "列出当前目录下的文件", "搜一下项目里哪里用了 TODO", "show me the contents of main.go",
 "读一下 ~/notes/todo.txt", "找出所有 .log 文件", "看看 src 目录下有哪些文件", "grep for 'password' in the repo"],[
 "打印一下 README.md 内容", "看下 logs 目录里都有什么", "find all files named *.proto in this project",
 "读取 /etc/hosts 文件", "在代码里搜索 InitRouter 的定义", "看看 package.json 里有哪些依赖"]),
"file_write": ([
 "把这段内容保存到 notes.md", "新建一个 todo.txt 写入今天的任务", "在 config.yaml 里把端口改成 9090", "append this line to log.txt",
 "把刚才的总结写进 summary.md", "修改 README 第一段", "把这些数据保存成 data.csv", "write this json to output.json"],[
 "把上面的回答保存到 answer.md", "在 .env 里加一行 DEBUG=true", "create a file hello.txt with 'hi'",
 "把这段代码写到 utils.py", "把会议要点追加到 meeting.md 末尾", "把 config.json 里的 timeout 改成 30"]),
"file_delete": ([
 "删掉 temp.txt", "把刚才生成的 md 文件删除", "删除 output.json", "delete the file draft.md",
 "把 notes.txt 删了", "remove old_report.pdf from the workspace", "删除那个临时文件", "把 build.log 删掉"],[
 "把 summary.md 删了", "删除刚刚保存的那个文件", "delete test_output.csv", "把桌面上的 a.txt 删掉", "remove the file I just wrote", "删除 backup.json"]),
"shell_command": ([
 "执行一下 ls -la", "运行 npm install", "跑一下 make build", "run python main.py",
 "执行 docker ps", "在本机运行 go test ./...", "帮我执行 pip install requests", "run the script deploy.sh"],[
 "执行 df -h 看看", "运行一下 npm run dev", "run `cargo build --release`", "本机执行 brew update", "跑一下 ./start.sh", "执行 ping baidu.com"]),
"git_inspect": ([
 "看看 git status", "有哪些未提交的修改", "看一下最近改动的 diff", "show me the git diff",
 "当前分支是什么，有啥改动", "列出改过的文件", "看下工作区状态", "what changed since last commit"],[
 "git diff 看一下", "看看我改了哪些文件还没提交", "show uncommitted changes", "当前仓库状态怎么样", "看下暂存区有什么", "diff of main.go please"]),
"git_mutate": ([
 "提交一下代码", "把改动 commit 了", "push 到远程", "commit these changes with message 'fix bug'",
 "提交并推送", "git push origin main", "把这些修改提交到仓库", "commit and push"],[
 "帮我 commit 一下", "推送到 GitHub", "commit with message 'update docs'", "把代码提交上去", "把当前改动推到远程分支", "git commit 一下然后 push"]),
"audio_record": ([
 "开始录音", "帮我录一下这次会议", "打开麦克风录音", "start recording audio",
 "录制接下来的讨论", "开个录音", "record the meeting", "用麦克风录一段音频"],[
 "录音", "帮我把接下来的会议录下来", "start a voice recording", "打开录音功能", "录一段我说的话", "begin recording the discussion"]),
"audio_transcribe": ([
 "把这个录音转成文字", "识别一下 meeting.mp3 里说了什么", "把会议录音整理成纪要", "transcribe this audio file",
 "把这段语音转写出来", "这个 wav 文件说了啥", "语音识别一下这个文件", "transcribe interview.m4a"],[
 "把这个音频转写成文字稿", "识别 voice.ogg 的内容", "transcribe the recording I uploaded", "把会议录音转成文字", "这段语音说的什么", "speech to text for call.mp3"]),
"audio_synthesize": ([
 "把这段话读出来", "用语音朗读这篇文章", "把这段文字合成语音播放", "read this text aloud",
 "朗读一下上面的回复", "TTS 播放这句话", "用中文念一下这段", "speak this paragraph"],[
 "把这段念给我听", "朗读一下这首诗", "read the summary out loud", "用语音播放一下这段话", "把回复读出来", "text to speech this sentence"]),
"audio_deliver": ([
 "把这段话转成语音发到群里", "发一条语音消息给他", "用语音把这个通知发出去", "send this as a voice note",
 "把这段文字合成语音发到当前群", "send a voice message saying hello", "发个语音给大家", "speak this and send it to the group"],[
 "把这句话用语音发到群里", "发一条语音说我晚点到", "send this text as voice to the chat", "用语音消息把结果发给我", "发个语音通知", "deliver this as a voice bubble"]),
"web_fetch": ([
 "读一下 https://example.com/article 这篇文章", "抓取 https://news.ycombinator.com 首页内容", "看看这个链接讲了什么 https://go.dev/blog", "fetch https://arxiv.org/abs/2401.00001",
 "总结一下 https://zhuanlan.zhihu.com/p/12345", "打开这个网址读正文 https://www.ruanyifeng.com/blog/", "get the content of https://docs.python.org/3/", "这个页面写了啥 http://blog.example.cn/post/1"],[
 "帮我读一下 https://github.com/cactus-compute/needle 的内容", "抓取 https://www.qq.com 页面正文", "fetch and summarize https://openai.com/blog",
 "看看 https://pkg.go.dev/net/http 写了什么", "总结 https://mp.weixin.qq.com/s/abcdef 这篇", "read https://example.org/changelog"]),
"audit_read": ([
 "查一下今天的审计日志", "搜一下我们之前聊过的关于 OCR 的对话", "看看最近有没有危险的工具调用", "show the security audit log",
 "检查一下项目健康状态", "查历史会话里提到 redis 的记录", "看看昨天执行过哪些命令的审计记录", "search my chat history for 'invoice'"],[
 "看下最近一周的安全审计记录", "我们以前讨论过部署方案吗，查一下历史", "show recent risky tool calls", "检查系统健康状况", "搜历史对话里关于报销的内容", "audit log for yesterday"]),
"knowledge_read": ([
 "在知识库里查一下报销流程", "知识库里有没有关于 OCR 的资料", "从知识库检索部署文档", "search the knowledge base for onboarding",
 "知识库里怎么说的 API 限流", "查一下我存的笔记里关于 Go 泛型的内容", "根据知识库回答：年假怎么算", "find docs about SSO in my knowledge base"],[
 "知识库里关于发票的规定是什么", "从知识库找一下产品价格表", "what does my knowledge base say about vpn setup", "检索知识库里的会议纪要", "知识库里有没有 PDF 翻译相关文档", "根据我保存的资料回答这个问题"]),
"app_launch": ([
 "打开 VS Code", "启动微信", "打开 https://www.baidu.com", "open Spotify",
 "打开下载文件夹", "启动终端", "帮我打开网易云音乐", "open the folder ~/projects"],[
 "打开 Chrome", "启动钉钉", "open https://github.com in my browser", "打开桌面文件夹", "运行一下计算器", "open Finder"]),
"file_download": ([
 "把 https://example.com/a.pdf 下载下来", "下载这个安装包 https://dl.google.com/go1.25.linux-amd64.tar.gz", "download https://arxiv.org/pdf/2401.00001.pdf",
 "把这个链接的图片保存到本地", "下载 https://x.com/data.csv 到桌面", "把这个文件下载到 Downloads", "save https://site.com/report.xlsx locally", "下载这个模型文件到本地"],[
 "下载 https://example.org/manual.pdf", "把这个压缩包下载到本机 https://a.com/b.zip", "download the dataset from https://data.gov/x.csv",
 "保存这个链接里的视频到本地", "把 https://cdn.x.com/logo.png 下载下来", "fetch this installer to disk: https://get.app/setup.exe"]),
"schedule_manage": ([
 "每天早上 9 点提醒我喝水", "创建一个每周一的定时任务", "列出所有定时任务", "pause the daily backup schedule",
 "删除那个每天备份的定时任务", "把提醒改到下午三点", "每小时检查一次磁盘空间", "set a reminder for 6pm"],[
 "明天早上八点提醒我开会", "看看我有哪些定时任务", "delete the weekly report schedule", "暂停每晚的同步任务", "每天凌晨两点自动跑一次清理脚本", "恢复之前暂停的定时任务"]),
"schedule_dispatch": ([
 "每天早上 9 点把天气发到群里", "每周五下午把周报发给团队群", "每天定时把销售数据推送到微信", "every morning send the news summary to the team chat",
 "每小时把服务器状态发到运维群", "定时把日报发给张三", "每天晚上 8 点把汇率推送到蓝信", "schedule a daily message to the group at 10am"],[
 "每天早上把今日新闻推送到群里", "每周一把上周数据汇总发给老板", "send the build status to the dev channel every hour",
 "每天下班前把待办提醒发到群", "定时每晚把监控截图发给我微信", "每月 1 号把账单汇总发到财务群"]),
"config_manage": ([
 "把模型切换成 deepseek", "看看你现在用的是哪个 LLM", "把最大推理轮数改成 50", "switch your model to gpt-4o",
 "修改一下你的配置", "换一个模型服务商", "更新一下我的用户画像，我是后端工程师", "show your current configuration"],[
 "切换到 qwen 模型", "你现在用的哪个模型提供商", "set max iterations to 30", "把你的配置改一下，温度调低", "修正一下我的画像，我不写前端", "change your llm provider to openrouter"]),
"memory_manage": ([
 "记住我喜欢用 Go", "以后都用中文回复我，记下来", "忘掉我之前说的那个偏好", "remember that my project uses PostgreSQL",
 "看看你记得关于我的哪些事", "更新记忆：我换到上海工作了", "删除关于旧项目的记忆", "list what you remember about me"],[
 "记住我的生日是 5 月 3 号", "你都记得我什么", "forget my old email address", "记一下：这个项目用 pnpm", "把那条过时的记忆删了", "update memory: I prefer tabs over spaces"]),
"task_track": ([
 "加一个待办：写周报", "把待办列表列出来", "把第二个任务标记为完成", "add a todo: review PR",
 "删掉待办里的买菜", "更新任务状态为进行中", "看看还有哪些任务没完成", "create a task to fix login bug"],[
 "待办里加一条：联系客户", "列出当前任务清单", "mark the first todo as done", "把那个任务删掉", "还有什么待办没做", "新增任务：更新文档"]),
"goal_manage": ([
 "创建一个长期目标：一个月内把测试覆盖率提到 80%", "看看当前的长期目标进度", "结束那个长期目标", "create a long-running goal with a budget of 100 steps",
 "设一个目标，持续优化首页性能直到 LCP<2s", "列出所有长期目标", "停止当前目标", "check the status of my goal"],[
 "建一个长期目标：持续修复所有 lint 警告", "我的长期目标进展如何", "end the current goal", "创建目标，自动推进直到所有测试通过", "取消那个长期目标", "show my active goals"]),
"template_manage": ([
 "创建一个会话模板", "列出我的模板", "用代码审查模板启动新会话", "save this as a session template",
 "把当前工具和项目路径存成模板", "删除那个旧模板", "用模板开一个 Claude Code 会话", "list session templates"],[
 "新建一个前端项目的会话模板", "有哪些会话模板", "start a session from my review template", "把这套配置保存为模板", "用上次的模板起一个新会话", "show my templates"]),
"session_manage": ([
 "列出所有编码会话", "看看那个 codex 会话的输出", "中断当前会话", "list coding sessions",
 "给会话发送继续指令", "终止卡住的会话", "切换到另一个项目", "show providers of coding tools"],[
 "看下现在有哪些会话在跑", "把那个卡死的会话停掉", "send 'continue' to the running session", "查看会话最近的输出", "切换当前项目到 maclaw", "kill the stuck claude code session"]),
"delegate_task": ([
 "把这个任务分给子 agent 去做", "并行执行这三个任务", "组织几个专家讨论一下这个方案", "delegate this to a sub-agent",
 "开几个子代理分别调研这三个方向", "让多个 agent 会诊一下这个 bug", "把调研和编码分别委派出去", "run these tasks in parallel with sub agents"],[
 "委派一个子 agent 去整理文档", "同时并行处理这几个文件", "spin up sub-agents to research each option", "找几个专家角色一起评审方案", "把这个任务交给子代理", "parallelize these three jobs"]),
"knowledge_admin": ([
 "刷新知识库来源", "禁用那个知识库数据源", "删除知识库里的旧来源", "show knowledge base stats",
 "跑一下知识库质量维护", "检查知识库健康状况", "给知识库条目加标签", "remove duplicates in the knowledge base"],[
 "知识库里那个来源重新刷新一下", "停用这个知识源", "knowledge base health check", "清理知识库重复条目", "看看知识库统计信息", "删除知识库中过期的来源"]),
"unknown": ([
 "你好", "谢谢", "你是谁", "hello", "哈哈", "早上好", "讲个笑话", "你能做什么"],[
 "嗨", "好的谢谢", "who are you", "晚安", "今天心情不错", "thanks!"]),
}
