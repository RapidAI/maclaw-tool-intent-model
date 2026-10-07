# MODELS

所有头的输入是 maclaw 纯 Go runner 产生的 Qwen3-Embedding-0.6B（Q8_0 GGUF，sha256 `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`）向量。

| 文件 | 大小 (B) | sha256 | 说明 |
|---|---|---|---|
| `models/intent/head_mlp_qwen3go_xf.json` | 33,787,963 | `c0256107d90dbf9f381cc1d1c46872984714234d2d024ff3fe0a0f0e6f640c69` | intent head — PRODUCTION r4 (frozen r11S5t: 5-seed r10CAP-recipe MLP, hidden 2560, T 0.668, tau 0.80 / tau_s 0.98, calibration.source oof_rule; identical weights/thresholds to frozen file sha 2b15f3e4…) |
| `models/intent/head_mlp_qwen3go_xf_r3W5F.json` | 6,011,299 | `873385d4544fa0b153ff0ac63833eb8e76d22c55c26cc8c6b80ddfa5226ffb10` | intent head (previous production, r3W5F) |
| `models/intent/head_r9FE.json` | 10,038,066 | `bb894e362697038810c8c227280c7af814b34deaeea170cdb4324e9861f482d8` | intent head r4 candidate r9FE (FE-cost λ0.5 ×3; r4 2b winner, not frozen) |
| `models/intent/head_r10CAP.json` | 20,258,783 | `66843a2a62250655e832f26cf5e630695facb3805c1f9e9cd8a8d149b645dcb1` | intent head r4 candidate r10CAP (width 512, dropout 0.3, FE-cost λ0.5 ×3 seeds; Go indep 0.7497 / FE 0; gate missed by 2) |
| `models/intent/head_r7W.json` | 3,380,917 | `49a3aa94d24b10e7ea0c47a88b92a66d02cd059f86e95b92d66be762f5afe333` | intent head r4 candidate r7W (full aug data, base seed 0; not frozen) |
| `models/tool/tool_head_ovr_qwen3go.json` | 935,594 | `e52352851da9fdaa69d5ad33ceb33352a090b3f1961b15e5dd6684b050454a50` | tool head (production, OvR) |
| `models/interrupt/interrupt_relevance_qwen3go.json` | 2,662,440 | `7d5494785311997d46ea80a65425f7625ad4e86fe6a2fb2745759474e96eb4ed` | interrupt relevance scorer (production, r4) |
| `models/coding/select_c.json` | 5,732 | `d5eb4248eaa22df32c0320a3d1b3552bb56c53693c0b4971bd71322a4d73e0ee` | coding fusion config selection (train half) |
| `models/coding/ens_c.json` | 50,105 | `f4aa002852ad324ecb19b9bd5972507752c0059a592d50f98d35e392384e6542` | coding fusion ensemble config |
| `models/coding/oneshot_c.json` | 729 | `6b9bc73fddd270dd9284aebf945a8ec93c2a1fe59b0a248242a01e092bb08c8e` | coding one-shot confirm result |
| `models/interrupt/oneshot_b.json` | 1,015 | `d1db234ad63bf7835d4063041cd59728ca0dc8161d1f04b50f70a5958df5f5b7` | interrupt one-shot confirm result |
| `models/release_assets.json` | 973 | `d601432a50f489333f5b4637f5b01500600be9254ce6ad676af58732c3b67f94` | maclaw release manifest (local/qwen3-only) |

生产 release pin（maclaw local/qwen3-only `release_assets.json`，第 4 轮冻结后）：意图头 c0256107…（r11S5t），工具头 e5235285…，打断头 7d549478…。上一版意图头为 873385d4…（r3W5F）。
