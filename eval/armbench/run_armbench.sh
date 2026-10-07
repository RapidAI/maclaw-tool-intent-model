#!/bin/bash
# ARM SDOT Q8P bench — no set -x (password hygiene N/A here; no password)
set -euo pipefail
D="$HOME/maclaw_q8p_armbench"
B="$D/bin/embbench"
Q="$D/Qwen3-Embedding-0.6B-Q8_0.gguf"
T="$D/texts/bench_short.txt,$D/texts/bench_long.txt"
O="$D/out"
mkdir -p "$O"
cd "$D"

echo "=== HOST $(date -Iseconds) ==="
uname -a
echo "=== Features asimddp ==="
grep -o asimddp /proc/cpuinfo | head -1 || echo NO_asimddp
echo "=== load ==="; uptime; free -h | head -2

# Confirm path selection via a tiny probe: load mode under default env
echo "=== probe default path (embbench -mode load) ==="
"$B" -model "$Q" -mode load > "$O/probe_default.json" 2>"$O/probe_default.err" || true
# Also check cpu feature via /proc; kernel selects SDOT when HasASIMDDP && MACLAW_Q8P!=0

busy() {
  read -r _ a b c d e f g h _ < /proc/stat
  t1=$((a+b+c+d+e+f+g+h)); i1=$((d+e))
  sleep 3
  read -r _ a b c d e f g h _ < /proc/stat
  t2=$((a+b+c+d+e+f+g+h)); i2=$((d+e))
  awk "BEGIN{printf \"%.2f\", $(nproc)*(1-($i2-$i1)/($t2-$t1))}"
}
waitload() {
  for i in $(seq 1 20); do
    bz=$(busy)
    if awk "BEGIN{exit !($bz<=4.0)}"; then BUSY=$bz; return; fi
    echo "   wait busy=$bz $(date +%T)"
    sleep 20
  done
  BUSY=$bz
  echo "   proceeding with busy=$BUSY"
}

run() {
  name=$1; shift
  waitload
  echo "== $name start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) busy_others=$BUSY"
  # env vars then binary — "$@" may start with env assignments via env
  env "$@" > "$O/$name.json" 2>"$O/$name.err"
  echo "== $name exit=$? $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg)"
}

# Round 1 order: sdot_2p, sdot_1p, float
# Round 2 reverse
for rd in 1 2; do
  if [ "$rd" = 1 ]; then
    order="sdot_2p sdot_1p float"
  else
    order="float sdot_1p sdot_2p"
  fi
  for c in $order; do
    case $c in
      sdot_2p) ENVVARS="";;  # default: Q8P on, two-pass
      sdot_1p) ENVVARS="MACLAW_QWEN3_Q8P_2PASS=none";;
      float)   ENVVARS="MACLAW_QWEN3_Q8P=0";;
    esac
    # shellcheck: expand ENVVARS as separate words for env
    # shellcheck disable=SC2086
    run "r${rd}_${c}_latency" $ENVVARS "$B" -model "$Q" -mode latency -n 200 -texts "$T"
    # shellcheck disable=SC2086
    run "r${rd}_${c}_batch" $ENVVARS "$B" -model "$Q" -mode batch -batch 64 -rounds 3 -texts "$T"
  done
done

# Correctness: float / 2p / 1p dumps
DUMP="$D/bin/qwen3dump"
CIN="$D/texts/corr_in.jsonl"
echo "=== correctness dumps ==="
waitload
echo "== corr_float start $(date +%T)"
env MACLAW_QWEN3_Q8P=0 "$DUMP" -model "$Q" -in "$CIN" -role classification -batch 64 -tokens -out "$O/corr_float.jsonl" > "$O/corr_float.log" 2>&1
echo "== corr_sdot_2p start $(date +%T)"
env "$DUMP" -model "$Q" -in "$CIN" -role classification -batch 64 -tokens -out "$O/corr_sdot_2p.jsonl" > "$O/corr_sdot_2p.log" 2>&1
echo "== corr_sdot_1p start $(date +%T)"
env MACLAW_QWEN3_Q8P_2PASS=none "$DUMP" -model "$Q" -in "$CIN" -role classification -batch 64 -tokens -out "$O/corr_sdot_1p.jsonl" > "$O/corr_sdot_1p.log" 2>&1

python3 "$D/texts/cmp_q8k.py" "$O/corr_sdot_2p.jsonl" "$O/corr_float.jsonl" | tee "$O/corr_2p_vs_float.txt"
python3 "$D/texts/cmp_q8k.py" "$O/corr_sdot_1p.jsonl" "$O/corr_float.jsonl" | tee "$O/corr_1p_vs_float.txt"

echo ARMBENCHDONE $(date -Iseconds)
