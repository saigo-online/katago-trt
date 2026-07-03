#!/usr/bin/env bash
# Correctness gate for the TensorRT-RTX port: the ported backend must produce net evals that
# agree with a trusted FP32 reference to within FP16-class rounding. We run the SAME fixed
# queries (maxVisits=1 => pure net eval, no search noise) through three engines and diff:
#   * build/trt/katago    -> TensorRT-RTX (the port under test)
#   * build/cuda/katago   -> CUDA FP16     (KataGo's hand-pinned reference backend)
#   * build/cuda/katago   -> CUDA FP32     (useFP16=false; the ground truth)
# Exits non-zero if TRT-RTX diverges from FP32 beyond tolerance.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NET="${NET:-/home/taro/code/katago/nets/kata1-b28c512nbt-s13255194368-d5935380940.bin.gz}"
OUT="$ROOT/bench/results"
WIN_TOL="${WIN_TOL:-0.02}"      # winrate abs tolerance vs FP32 ground truth
POL_TOL="${POL_TOL:-0.05}"      # per-move policy abs tolerance vs FP32 ground truth
mkdir -p "$OUT"

[ -f "$NET" ] || { echo "net not found: $NET" >&2; exit 1; }
for b in trt cuda; do
  [ -x "$ROOT/build/$b/katago" ] || { echo "missing build/$b/katago (run ./bench/build.sh)" >&2; exit 1; }
done

cat > "$OUT/analysis.cfg" <<EOF
logDir = $OUT/analysislogs
numAnalysisThreads = 1
numSearchThreads = 1
nnMaxBatchSize = 32
nnCacheSizePowerOfTwo = 18
EOF
sed 's#^logDir.*#logDir = '"$OUT"'/analysislogs32#' "$OUT/analysis.cfg" > "$OUT/analysis-fp32.cfg"
echo "useFP16 = false" >> "$OUT/analysis-fp32.cfg"
# TRT-RTX FP16 path: strongly-typed half trunk via the ModelParser (convnets only).
sed 's#^logDir.*#logDir = '"$OUT"'/analysislogs16#' "$OUT/analysis.cfg" > "$OUT/analysis-fp16.cfg"
printf 'trtDisableOnnx = true\nuseFP16 = true\n' >> "$OUT/analysis-fp16.cfg"

cat > "$OUT/queries.jsonl" <<'EOF'
{"id":"empty","rules":"tromp-taylor","komi":7.5,"boardXSize":19,"boardYSize":19,"moves":[],"maxVisits":1,"includePolicy":true}
{"id":"opening","rules":"tromp-taylor","komi":7.5,"boardXSize":19,"boardYSize":19,"moves":[["B","Q16"],["W","D4"],["B","Q4"],["W","D16"]],"maxVisits":1,"includePolicy":true}
EOF

run() { # backend cfg outfile
  timeout 900 "$ROOT/build/$1/katago" analysis -model "$NET" -config "$2" \
    < "$OUT/queries.jsonl" > "$3" 2>/dev/null
}
echo ">>> TRT-RTX FP16 (mixed)…"; run trt  "$OUT/analysis-fp16.cfg" "$OUT/out-trt-fp16.jsonl"
echo ">>> TRT-RTX FP32/TF32…";    run trt  "$OUT/analysis.cfg"      "$OUT/out-trt.jsonl"
echo ">>> CUDA FP16…";            run cuda "$OUT/analysis.cfg"      "$OUT/out-cuda.jsonl"
echo ">>> CUDA FP32 (ground truth)…"; run cuda "$OUT/analysis-fp32.cfg" "$OUT/out-cuda32.jsonl"

WIN_TOL="$WIN_TOL" POL_TOL="$POL_TOL" python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
def load(p):
    d={}
    for line in open(p):
        line=line.strip()
        if line: j=json.loads(line); d[j["id"]]=j
    return d
ref=load(f"{out}/out-cuda32.jsonl")
cand={"TRT-RTX-FP16":load(f"{out}/out-trt-fp16.jsonl"),
      "TRT-RTX-FP32":load(f"{out}/out-trt.jsonl"),
      "CUDA-FP16":load(f"{out}/out-cuda.jsonl")}
win_tol=float(os.environ["WIN_TOL"]); pol_tol=float(os.environ["POL_TOL"])
worst={}
for name,c in cand.items():
    ww=wp=ws=0.0
    for k in ref:
        ri=ref[k]["rootInfo"]; ci=c[k]["rootInfo"]
        ww=max(ww,abs(ri["winrate"]-ci["winrate"]))
        ws=max(ws,abs(ri["scoreLead"]-ci["scoreLead"]))
        rp=ref[k].get("policy") or []; cp=c[k].get("policy") or []
        wp=max(wp,max((abs(a-b) for a,b in zip(rp,cp)), default=0.0))
    worst[name]=(ww,ws,wp)
    print(f"{name:14s} vs FP32:  winrate {ww:.2e}  score {ws:.2e}  policy {wp:.2e}")
# Gate on the FP16 build — that is the port we ship and benchmark.
ww,ws,wp=worst["TRT-RTX-FP16"]
ok = ww<=win_tol and wp<=pol_tol
print(f"\nTRT-RTX-FP16 tolerance: winrate<= {win_tol}  policy<= {pol_tol}  =>  {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY