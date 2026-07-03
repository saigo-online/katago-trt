#!/usr/bin/env bash
# Build one KataGo backend into its own bindir so we can A/B them side by side.
#
#   ./bench/build.sh TENSORRT     -> build/trt/katago
#   ./bench/build.sh CUDA         -> build/cuda/katago
#
# Environment probed and pinned for this box (NVIDIA GB10, Grace-Blackwell, aarch64):
#   CUDA 13.0, TensorRT 11.0 (headers in /usr/include/aarch64-linux-gnu), sm_121 (Blackwell).
set -euo pipefail

BACKEND="${1:-TENSORRT}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPP="$ROOT/cpp"

case "$BACKEND" in
  TENSORRT) SUB=trt ;;
  CUDA)     SUB=cuda ;;
  EIGEN)    SUB=eigen ;;
  *)        SUB="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')" ;;
esac
BUILD="$ROOT/build/$SUB"

# GB10 is Blackwell. CUDA 13 knows sm_121/sm_110; pin to keep ptxas from fanning out to every arch.
: "${CMAKE_CUDA_ARCHITECTURES:=121}"
: "${TENSORRT_INCLUDE_DIR:=/usr/include/aarch64-linux-gnu}"

echo ">>> Configuring $BACKEND -> $BUILD  (sm_$CMAKE_CUDA_ARCHITECTURES)"
mkdir -p "$BUILD"
cmake -S "$CPP" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_BACKEND="$BACKEND" \
  -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
  ${TENSORRT_INCLUDE_DIR:+-DTENSORRT_INCLUDE_DIR="$TENSORRT_INCLUDE_DIR"} \
  "${@:2}"

echo ">>> Building (this takes a while)…"
cmake --build "$BUILD" -j "$(nproc)"

echo ">>> Done: $BUILD/katago"
"$BUILD/katago" version || true
