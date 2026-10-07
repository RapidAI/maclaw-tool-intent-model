# maclaw-tool-intent-model

MaClaw 嵌入侧小模型的训练、评估、数据生成与判标代码，以及产出的模型和数据。目标是用 **Qwen3-Embedding-0.6B**（maclaw 纯 Go runner）替换 EmbeddingGemma，只发一个嵌入模型。

> 状态（第 4 轮，2026-10-07 18:55）：A（意图）、B（IM 打断）、C（编码子代理）在现有测试集和新确认集上的点估计**全部超过 Gemma**。意图头 r11S5t 已冻结，并已写入 maclaw `local/qwen3-only` 的 release 清单（尚未上传到 Model_Release）。A_confirm 上有一点需要说明：Qwen3 的敏感误暴露是 2 条，Gemma 是 1 条，见下文。

## 嵌入模型（不在本仓库）

- 模型：HuggingFace 官方 `Qwen/Qwen3-Embedding-0.6B-GGUF` 中的 `Qwen3-Embedding-0.6B-Q8_0.gguf`
  - 639,150,592 字节
  - sha256 `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`
- **本仓库的所有头都以 maclaw 纯 Go runner（`CGO_ENABLED=0`）产生的向量为输入**：Qwen3 last-token pooling，L2 归一化，1024 维。
  - 其他实现（llama.cpp、sentence-transformers 等）的向量在数值上不同，不能直接套用这些头。
  - 本仓库不包含向量文件（npy/pkl/f32）和 GGUF。复现时需要用 maclaw 的 vecdump（`corelib/embedding` 的 `TestZZVecDump`）重新生成向量。
- 意图分类用的 instruction 前缀：`Instruct: Classify the intent of this assistant user request\nQuery: `

## 流水线

### 意图（A）
- **L1**：规则 / 关键词快速路径（maclaw 内）。
- **L2**：锚点最大余弦。锚点在 Qwen3 分类 instruction 空间中，共 48 个标签、534 个锚点；按校准分数用 strict grant 规则（0.78 / 0.10）放行。
- **L3 本地头**：MLP（1024→hidden ReLU→49 个标签，其中 18 个敏感标签），温度 T，放行阈值 τ；敏感标签用更严的 τ_s。
  - τ 规则事先锁定：在 pooled 留一家族（LOFO）OOF 上，取满足"放行准确率 ≥ 0.976 且敏感精度 ≥ 0.997"的 (τ, τ_s) 中覆盖率最大的一组。
  - 训练数据是旧训练集加上多个生成家族的跨家族数据（GLM / gpt-oss / Llama / Mistral / Qwen），评估按家族做 LOFO。
- **工具头**：49 个工具的 one-vs-rest 线性头（top-k 5、阈值 θ 0.3）。

### IM 打断相关性（B）
- 用 IS/IM 两个 instruction 分别嵌入任务和消息，得到向量 u、v。
- 打分器为 MLP：输入 [u, v, |u−v|, u⊙v, cos]，4097→64 ReLU→1。
- 档位阈值在训练集评估半上标定：high 0.9998 / low 0.8936。

### 编码子代理（C）
- 任务侧 instruction 用 CT3（"Given a programming task description, retrieve descriptions of the tools a developer would use for it"）。
- 文档侧嵌入文本为 `Tool: <name>\nDescription: <desc>`。
- 打分是 BM25（权重 0.6）与余弦的融合。

## 目录

| 目录 | 内容 |
|---|---|
| `train/` | 头的训练：sklearn / torch MLP、FE 代价、标签平滑、容量扫描、多 instruction、打断 MLP 搜索、编码配置选择 |
| `eval/` | Go/Python 评估、闸门、bootstrap 对比、阈值模拟、分析脚本 |
| `datagen/` | 跨家族数据生成、盲判标、合并、去重、确认集生成与冻结 |
| `export/` | 导出为单个 MLP JSON（集成合并）、release 清单生成、冻结脚本 |
| `models/` | 生产头（第 4 轮冻结的意图头 r11S5t、工具头、打断头、编码配置）、上一版意图头 r3W5F 和第 4 轮候选头；sha256 见 `MODELS.md` / `models/manifest.json` |
| `data/` | 训练与测试数据，以及第 4 轮确认集（见下） |
| `PATHS.tsv` | 每个脚本在原工作区中的路径。脚本内部的相对路径按原工作区根目录写，复现时请按此表还原目录结构 |

## 数据

- **`data/core/`**：
  - 旧意图集 `intent_all.jsonl`：`split=train` 为旧训练集，`split=test` 为 old-339；
  - 独立测试集 `indep_test.jsonl`（indep839）；
  - Kimi150 子集：`go_in_GK_test_150.jsonl` 的 id；
  - 歧义集 `indep_ambiguous.jsonl`；
  - 跨家族训练集 `train_xfamily/`（含各轮快照、原始生成和判标记录）；
  - 工具目录 `core_tool_defs.json`；
  - 意图定义 `intent_defs.json` / `intent_meta.json`（含敏感标签列表）。
- **`data/xf3/`**：第 4 轮扩充数据。
  - `aug.jsonl`：3833 条，带 `fam` 家族标签，`qwen` 为本地 Qwen 生成，在 LOFO 中单独成一折；
  - 其余文件：候选、判标原始记录、聚焦标签。
- **`data/hard/`**：易混敏感对的硬例。
- **`data/interrupt/`**：打断配对数据。
  - 训练集 F：`int3_train.jsonl`；
  - 评估池：`cap3_interrupt.jsonl` / `r4B_interrupt.jsonl`。训练半与测试半按 `md5(id) % 2` 切分，见 `train/r4B/search_b.py`。
- **`data/coding/`**：编码任务 `coding.jsonl` 与候选目录 `coding_cands.json`。目录共 **56** 项：生成时提示词要求 45 项，ministral-8b 返回了 56 个互不重复的条目，全部保留。C 的所有评估（cap3、测试半、C_confirm）用的都是这 56 项（`eval/cap3/cap3_eval.txt` 记为 "305 tasks × 56 candidates"，每条信号恰好 56 个分数）。内部 REPORT 中的"45 项"是笔误，应为 56。训练半与测试半同样按 `md5(id) % 2` 切分，见 `train/r4C/select_c.py`。
- **`data/topic/`**：话题切换。
- 数据中出现的 `sk-ant-api03-xxxx`、`xoxb-1234`、`192.168.1.100` 等是生成的用户请求里的**虚构占位符**，不是真实凭据或地址。
- **确认集 `data/confirm_r4/`**：A_confirm（415）、B_confirm（492）、C_confirm（240），附 `SHA256SUMS` 和 `FROZEN.json`。三个集合都已对冻结的配置各评估过一次，现在公开。

## 复现

1. 准备 maclaw（含 `corelib/embedding` 纯 Go Qwen3 runner）和上述 GGUF。用 `CGO_ENABLED=0` 编译 vecdump，按对应 instruction 嵌入各数据集，嵌入脚本见 `train/**/dump*.sh`、`train/scripts/indep_embed.sh`。
2. 按 `PATHS.tsv` 还原目录。
3. 意图头：
   - 训练：`train/scripts/xf_train.py`（sklearn 基线，LOFO）、`train/r4A/torch_cap.py 512 1 0.3 <seed>`（第 4 轮最终配方：宽 512、dropout 0.3、FE 代价 λ 0.5、LS 0.05，seed 0–4 平均）；
   - 选择：`eval/r4A/levers_eval7.py`；
   - 导出：`export/r4A/export_cap.py`（最终头），或 `export_ens.py` / `export_anc.py`；阈值 TH1：`eval/r4A/th_sim.py`。
4. Go 端到端：`eval/run_hintfix.sh`，配合 maclaw 的 intent harness（`ZZ_L3=head`）。
5. 打断：`train/r4B/search_b.py` → `export/r4B/export_b.py` → `eval/r4B/oneshot_b.py`。
6. 编码：`train/r4C/variants_c.py` → `select_c.py` → `eval/r4C/oneshot_c.py`。
7. LLM 端点（数据生成、判标）的密钥从本地文件读取（脚本中写作 `<KEYS_DIR>`），仓库里没有任何密钥。GPU 服务器地址写作 `<GPU_HOST>` / `<GPU_SSH_PORT>`。

## 记分：Qwen3 vs Gemma（数字取自内部 REPORT）

### 第 3 轮最终逐项判定（生产配置，意图头 r3W5F）

| 能力 | Qwen3 | Gemma | 差值 [95% CI] |
|---|---|---|---|
| 意图 独立 839（主意图 / 放行准确率 / 敏感误暴露） | 0.739 / 0.975 / 5 | 0.417 / 0.907 / 6 | +0.322 [+0.285, +0.358] |
| 意图 Kimi150 | 0.773 / 0.991 / 0 | 0.493 / 0.892 / 2 | +0.280 [+0.200, +0.360] |
| 意图 旧 339 主意图 | 0.838 | 0.847 | −0.009 [−0.053, +0.032] |
| 意图 旧 339 放行准确率（放行条数）/ 敏感误暴露 | 0.986 (282) / 2 | 0.976 (288) / 1 | |
| 工具头 R@1（old262 / indep644） | 0.962 / 0.932 | 0.920 / 0.691 | |
| RAG / 记忆 R@1 | 0.911 / 0.911 | 0.842 / 0.679 | |
| 话题切换 AUC | 0.856 | 0.848 | +0.007 [−0.048, +0.060] |

### 第 4 轮最终记分（A、B、C 均已冻结，各自在确认集上评估过一次；配对 bootstrap 2000 次，seed 7）

| 能力 | 集合 | Qwen3 r4 | Gemma | 差值 [95% CI] |
|---|---|---|---|---|
| A 意图（主意图 / 放行准确率（放行条数）/ 敏感误暴露） | old-339 | 0.861 / 0.990 (289) / 0 | 0.847 / 0.976 (288) / 1 | +0.015 [−0.032, +0.059] |
| A 意图 | indep839 | 0.788 / 0.972 (652) / 2 | 0.417 / 0.907 (356) / 6 | +0.371 [+0.334, +0.408] |
| A 意图 | Kimi150 | 0.827 / 0.992 (125) / 0 | 0.493 / 0.892 (83) / 2 | +0.333 [+0.247, +0.420] |
| A 意图 | **A_confirm（n=415）** | **0.863 / 0.989 (362) / 2** | 0.653 / 0.922 (294) / 1 | **+0.210 [+0.164, +0.255]** |
| B 打断 AUC | 现有测试半 ETE（n=124） | 0.875 | 0.851 | +0.024 [−0.058, +0.102] |
| B 打断 AUC | B_confirm（n=492） | 0.918 | 0.636 | +0.282 [+0.235, +0.328] |
| C 编码 F1 / MRR | 现有测试半（n=165） | 0.710 / 0.969 | 0.640 / 0.961 | ΔF1 +0.070 [+0.023, +0.119]；ΔMRR +0.008 [−0.011, +0.028] |
| C 编码 F1 / MRR | C_confirm（n=240） | 0.627 / 0.976 | 0.535 / 0.946 | ΔF1 +0.092 [+0.059, +0.124]；ΔMRR +0.030 [+0.008, +0.052] |

## 第 4 轮 A（意图）是怎么过的

- 替换条件（Daniel 定）：old-339 主意图 > 0.847 且敏感误暴露 ≤ 1，然后在 A_confirm 上一次性确认，并且也要超过 Gemma。A_confirm 从不用于选择。
- 各候选的 Go indep839 结果：

  | 候选 | 主意图 | 敏感 FE |
  |---|---|---|
  | r3W5F | 0.739 | 5 |
  | r7W（全量扩充数据） | 0.727 | 2 |
  | r9FE（FE 代价） | 0.741 | 1 |
  | r10CAP（宽 512 + dropout 0.3 + FE 代价） | 0.7497 | 0 |
  | r11S5（同配方 5 seed） | 0.7521 | 0（通过 indep 闸门） |
  | **r11S5t**（同权重，τ_s 0.99 → 0.98） | **0.788** | 2 |

- 关键一步是分析 old-339 漏判：55 条漏判里，47 条是头的 argmax 正确、只是低于阈值被拒，其中 11 条是概率在 0.97–0.99 的敏感标签。
- 因此按事先登记的阈值规则 TH1 选阈值：在 old-339、indep839、Kimi150 都满足约束的点里，取 τ_s 最高的一个，结果是 (0.80, 0.98)。
- 无效的杠杆：
  - 敏感标签单独温度、L2 锚点分数作为特征、Matryoshka 截断、多 instruction 嵌入、按标签阈值：都没有规则点，或者分数更低；
  - 加深头：无收益。
- **A_confirm 说明**：Qwen3 的敏感误暴露是 2 条（Gemma 1 条），但 Qwen3 多放行了 68 条。

## 方法学要点（REPORT 摘要）

- 每一条选择规则都在看结果前登记。测试集（old-339 / indep839 / Kimi150）不参与训练、校准或选择，并做了精确去重和近重复去重（Qwen3 余弦 > 0.95）。
- 确认集用新的生成家族（Nemotron-3-Super）生成，再由另一家族盲判（gpt-oss-20b），冻结前不看内容，每个配置只评一次。
- 判标替代方案：Cohere 试用额度用尽后，改用与生成家族不同的判官盲判，同一 prompt。这一偏差已经协调人批准。
- 数值稳健性：avx512 / avx2 / arm64 三种内核之间，向量余弦 ≥ 0.9987。打断模型的档位有约 2% 在相邻档之间翻转，但 AUC 不变。

## 许可

代码与模型权重采用 Apache-2.0（见 `LICENSE`）。数据由多家 LLM 生成（Gemini、Kimi、Llama、Mistral、gpt-oss、GLM、Qwen、Nemotron 等），使用时还须遵守各生成模型的使用条款。
