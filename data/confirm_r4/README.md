# confirm_r4（暂不公开）

第 4 轮确认集 A_confirm（415）、B_confirm（492）、C_confirm（240）已在生成后冻结（sha256 记录在内部 `SHA256SUMS` / `FROZEN.json`）。

A_confirm 仍处于封存状态（意图头尚未通过事先登记的闸门，还没有做一次性评估）。为了不污染这次评估，三个确认集都会在 A 的最终一次性评估完成后，连同 `SHA256SUMS` 和 `FROZEN.json` 一起发布。
