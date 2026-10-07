# data/train_xfamily — multi-family TRAINING utterances (REPORT §19)
Never contains test items. Exact-dedupe (NFKC) vs train/anchor, old test 339, indep 839, ambiguous 95, Kimi 150;
near-dup = pure-Go Qwen3 RoleClassification cosine > 0.95 vs every test item. Indep Gemini/Kimi never used for training.

## Snapshots (judge-agree counts)

| snapshot | file | GLM | gpt-oss | Llama | Mistral | total |
|---|---|---|---|---|---|---|
| r1a | (early) | 26 | 152 | 36 | 29 | 243 |
| r1b | `train_xf_r1b.jsonl` | 314 | 533 | 373 | 292 | 1512 |
| r2a | `train_xf_r2a.jsonl` | 444 | 1511 | 605 | 833 | 3393 |
| r2b (FINAL head r2bF trained on this) | `train_xf_r2b.jsonl` | 514 | 2113 | 605 | 839 | 4071 |
| r3 (judged 10079/11319 candidates; 1240 unjudged — Cohere stalled — excluded) | `train_xf_r3.jsonl` | 514 | 3844 | 1179 | 2854 | 8391 |
| latest | `train_xf.jsonl` | = r3 | | | | |

## Generators / judge
| family | model | endpoint |
|---|---|---|
| Zhipu GLM | z-ai/glm-5.3-flash | NVIDIA NIM |
| OpenAI open-weight | openai/gpt-oss-20b | NVIDIA NIM |
| Meta Llama | meta/llama-3.2-90b-vision-instruct | NVIDIA NIM |
| Mistral | ministral-8b-latest | Mistral API |
| DeepSeek | deepseek-v4.1-flash | NIM — timed out, 0 items |
| Judge | Cohere command-a-03-2025 | Cohere (blind; not a generator family) |

Rounds: r1 all-labels×6 + 24 confusable×3; r2 weak×10; r3 all×10×2reps; r4–r5 weak×12–14×3–4reps (zh/mix heavy).
Scripts: `xf_gen.py` → `xf_build.py` → `xf_judge.py` → `xf_train.py` (`xf_remote.sh` on ARM). Vectors: `bin/qwen3dump_xf` from maclaw-final, RoleClassification.
Costs: free tiers / existing quota; no billing returned.

r3 judge: agree 8391 / disagree 1688 (glm 514/38, gpt-oss 3844/772, llama 1179/253, mistral 2854/625); lang of kept: zh 1918 / mix 3952 / en 2521. Raw items generated after the 09:21 candidate build (glm-r4/r5, llama-r3–r5 tails) are in raw_*.jsonl but NOT in candidates/train.

## Final (10:29 UTC+8)
FINAL head r3W5F = train_xf_r3.jsonl (8391) + old train 925 ×5 + fix 100. Selection signals: LOFO OOF + old-TRAIN 5-fold CV (scripts/xf_oldcv.py, out/xf/xf_oldcv_r3cv.json); never old-339. See REPORT §23 and out/crossfamily/FINAL_HEAD.txt.
