#!/bin/bash
set -uo pipefail
D="$HOME/maclaw_q8p_armbench"; B="$D/replace/embbench"; Q="$D/Qwen3-Embedding-0.6B-Q8_0.gguf"; G="$D/replace/embeddinggemma-300M-Q8_0.gguf"
T="$D/texts/bench_short.txt,$D/texts/bench_long.txt"; O="$D/replace/out"; mkdir -p "$O"; cd "$D"
echo "=== HOST $(hostname) $(date -Iseconds)"; uptime
busy() { read -r _ a b c d e f g h _ < /proc/stat; t1=$((a+b+c+d+e+f+g+h)); i1=$((d+e)); sleep 3
  read -r _ a b c d e f g h _ < /proc/stat; t2=$((a+b+c+d+e+f+g+h)); i2=$((d+e)); awk "BEGIN{printf \"%.2f\", $(nproc)*(1-($i2-$i1)/($t2-$t1))}"; }
waitload() { for i in $(seq 1 15); do bz=$(busy); if awk "BEGIN{exit !($bz<=4.0)}"; then BUSY=$bz; return; fi; echo "  wait busy=$bz"; sleep 20; done; BUSY=$bz; }
run() { name=$1; shift; waitload; echo "== $name start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) busy=$BUSY"
  env "$@" > "$O/$name.json" 2>"$O/$name.err"; echo "== $name exit=$? $(date +%T)"; }
for rd in 1 2; do
  if [ $rd = 1 ]; then order="gemma q3_1p q3_2p"; else order="q3_2p q3_1p gemma"; fi
  for c in $order; do case $c in
    gemma) run r${rd}_gemma_lat "$B" -model "$G" -mode latency -n 200 -texts "$T" -role classification;;
    q3_1p) run r${rd}_q3_1p_lat MACLAW_QWEN3_Q8P_2PASS=none "$B" -model "$Q" -mode latency -n 200 -texts "$T" -role classification;;
    q3_2p) run r${rd}_q3_2p_lat "$B" -model "$Q" -mode latency -n 200 -texts "$T" -role classification;;
  esac; done
done
run gemma_load "$B" -model "$G" -mode load
echo ARMDONE $(date -Iseconds)
