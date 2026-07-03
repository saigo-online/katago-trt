# katago-trt — TensorRT backend benchmark harness

This is a **clean fork of upstream [lightvector/KataGo](https://github.com/lightvector/KataGo)**
(`main`, commit `a339686`), kept as a sibling to `katago-web` so its results aren't
confounded by our WebGPU fork's changes.

## Why this exists

KataGo ships a TensorRT backend (`cpp/neuralnet/trtbackend.cpp`, `-DUSE_BACKEND=TENSORRT`),
but it does **not** build unmodified against this box's SDK — the GB10 ships **TensorRT-RTX**,
which removed mainline TensorRT's precision API. This fork ports the backend to TensorRT-RTX
(strongly-typed mixed-precision FP16) and A/Bs it against the CUDA backend with KataGo's own
`benchmark` subcommand.

**Headline:** after the port, TensorRT-RTX FP16 is on par with CUDA FP16 (trails at small
batch, edges ahead by batch ~32). Full numbers, the correctness table, and the porting story
are in [RESULTS.md](RESULTS.md).

## This box

- GPU: **NVIDIA GB10** (Grace-Blackwell superchip, `aarch64`, Blackwell `sm_121`)
- CUDA **13.0**, TensorRT **11.0.0** (headers in `/usr/include/aarch64-linux-gnu`)
- Protobuf 3.21.12, ZLIB 1.3 — all found by CMake, no extra installs needed.
- No Boost required (modern KataGo uses `std::filesystem`).

## Build

```bash
./bench/build.sh TENSORRT     # -> build/trt/katago
./bench/build.sh CUDA         # -> build/cuda/katago   (the A/B baseline)
```

Each backend gets its own `build/<sub>/` dir so they don't clobber each other.
`CMAKE_CUDA_ARCHITECTURES` defaults to `121` (Blackwell); override via env if needed.

## Correctness gate

```bash
./bench/validate.sh            # net evals vs CUDA FP32 ground truth (incl. the FP16 build)
```

Runs fixed queries (`maxVisits=1` → pure net eval) through TRT-RTX FP16, TRT-RTX FP32/TF32,
CUDA FP16, and a CUDA FP32 reference, then diffs winrate/score/policy. Exits non-zero if the
FP16 build diverges beyond tolerance. Run this before trusting any speed number.

## Benchmark

```bash
./bench/run.sh                          # sweep TRT-FP16 vs TRT-FP32 vs CUDA at t=16/32/64
NET=/path/to/model.bin.gz ./bench/run.sh
THREADS="16 64 256" MODES="trt-fp16 cuda" ./bench/run.sh
```

Default net: `kata1-b28c512nbt-s13255194368-d5935380940.bin.gz` (the b28 the `katago-on-mac`
forwarder drives). Results and summaries land in `bench/results/`.

## Reading the result

`katago benchmark` reports, per thread count:

- **nnEvals/s** — raw neural-net throughput (the number the backend swap moves directly).
- **visits/s** — full-MCTS search speed, which is what converts to strength.

The FP16 path uses the hand-built ModelParser (`trtDisableOnnx = true`, convnets only) +
`useFP16 = true`; without those, TensorRT-RTX builds an FP32/TF32 engine (~1.9× slower — the
old weakly-typed `kFP16` flag no longer exists). The first TRT run on a net pays a one-time
plan-build cost (cached under `~/.katago/trtcache`); the crossover vs CUDA moves with batch
size, hence the thread sweep.
