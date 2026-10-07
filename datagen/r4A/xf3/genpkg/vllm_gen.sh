#!/bin/bash
# temp: r4 A lever-3 generation server (localhost only). Remove after use.
export CPATH=/usr/local/cuda/include HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
exec /home/<GPU_USER>/vllm-env/bin/vllm serve Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 8011 --max-model-len 8192 --gpu-memory-utilization 0.80 --max-num-seqs 128 --max-num-batched-tokens 16384 --enable-prefix-caching --reasoning-parser qwen3
