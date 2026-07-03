#!/usr/bin/env bash
# A/B a net's raw search throughput across backends using KataGo's own `benchmark` subcommand,
# which reports neural-net evals/sec and full-MCTS visits/sec. Same net, same config, same board —
# only the backend/precision differs. Sweeps a few thread counts because the TensorRT-RTX vs CUDA
# crossover moves with batch size (TRT-RTX FP16 trails at small batch, catches up by batch ~32).
#
#   ./bench/run.sh                        # b28, TRT-FP16 vs TRT-FP32 vs CUDA, 19x19
#   NET=/path/to.bin.gz ./bench/run.sh    # override the net
#   THREADS="16 64 256" ./bench/run.sh    # override the thread sweep
#   MODES="trt-fp16 cuda" ./bench/run.sh  # subset of {trt-fp16, trt-fp32, cuda}
#
# Results and a summary land in bench/results/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NET="${NET:-/home/taro/code/katago/nets/kata1-b28c512nbt-s13255194368-d5935380940.bin.gz}"
THREADS="${THREADS:-16 32 64}"
MODES="${MODES:-trt-fp16 trt-fp32 cuda}"
BOARDSIZE="${BOARDSIZE:-19}"
VISITS="${VISITS:-3200}"
OUT="$ROOT/bench/results"
mkdir -p "$OUT/gtplogs"
[ -f "$NET" ] || { echo "net not found: $NET" >&2; exit 1; }

SUM="$OUT/run-summary.txt"; : > "$SUM"
echo "net: $NET   board: ${BOARDSIZE}x${BOARDSIZE}   visits: $VISITS" | tee -a "$SUM"

for mode in $MODES; do
  case "$mode" in
    trt-fp16) bin=trt;  fp16=1 ;;
    trt-fp32) bin=trt;  fp16=0 ;;
    cuda)     bin=cuda; fp16=0 ;;
    *) echo "unknown mode: $mode" >&2; continue ;;
  esac
  BIN="$ROOT/build/$bin/katago"
  [ -x "$BIN" ] || { echo "!! skip $mode: no binary at $BIN (./bench/build.sh)"; continue; }
  for t in $THREADS; do
    cfg="$OUT/run-$mode-$t.cfg"
    { echo "logDir = $OUT/gtplogs"
      echo "numSearchThreads = $t"
      echo "nnMaxBatchSize = $t"
      echo "nnCacheSizePowerOfTwo = 20"
      [ "$fp16" = 1 ] && { echo "trtDisableOnnx = true"; echo "useFP16 = true"; }
    } > "$cfg"
    log="$OUT/run-$mode-$t.log"
    "$BIN" benchmark -model "$NET" -config "$cfg" -boardsize "$BOARDSIZE" -v "$VISITS" -t "$t" > "$log" 2>&1 || true
    line=$(grep -oE 'visits/s = [0-9.]+ nnEvals/s = [0-9.]+ .*avgBatchSize = [0-9.]+' "$log" | tail -1)
    printf '%-9s t=%-3s : %s\n' "$mode" "$t" "$line" | tee -a "$SUM"
  done
done
echo | tee -a "$SUM"
echo "summary written to $SUM"
