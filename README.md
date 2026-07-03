# KataGo on TensorRT-RTX — FP16 mixed-precision + FP8 on Blackwell / GB10

A **TensorRT-RTX** neural-net backend for [KataGo](https://github.com/lightvector/KataGo).

KataGo already ships a TensorRT backend — but it does **not build unmodified** against
the SDK that NVIDIA ships on the **GB10 (Grace-Blackwell) / DGX Spark**: that's
**TensorRT-RTX**, which removed mainline TensorRT's precision API (`ILayer::setPrecision`,
`ITensor::setType`, `IBuilder::platformHasFastFp16()`, `BuilderFlag::kFP16`,
`kOBEY_/kPREFER_PRECISION_CONSTRAINTS`, `NetworkDefinitionCreationFlag::kEXPLICIT_BATCH`)
in favor of **always-strongly-typed** networks — precision comes from the tensor types in
the graph, not a builder flag.

This fork ports the backend to TensorRT-RTX and, more importantly, makes it **fast**: it
rebuilds the trunk as a strongly-typed **mixed-precision** graph (FP16 trunk, FP32-pinned
reductions/heads) and adds an optional **FP8** quantized trunk for the Blackwell tensor
cores.

> **Status (b28c512nbt @ 19×19, NVIDIA GB10, CUDA 13, TensorRT-RTX 11.0.0.114):**
> **FP8 runs KataGo ~1.4–1.7× faster than the stock CUDA backend** (e.g. +71 % at batch 16), and an
> Elo gauntlet confirms **+108 Elo vs stock CUDA at equal wall-clock** (z=3.0) — with the tuned σ=4
> activation scale it's statistically even at equal *visits*, so the speed is ~free strength.
> **FP16** matches or edges CUDA FP16 at
> batch ≥ 32 and is the most accurate non-reference engine (winrate within ~1e-3 of FP32).
> **FP4 (NVFP4)** executes on RTX but isn't a practical win yet (see below). Full numbers,
> the correctness table, and the porting story are in [`bench/RESULTS.md`](bench/RESULTS.md).

---

## Why FP16 needed a real port

TensorRT-RTX networks are *always strongly-typed*, so the old "build in FP32, flip
`kFP16`, let TRT auto-pick" path is gone. A naïve port that just guards out the removed
calls yields an all-FP32 graph → RTX runs it in FP32/TF32 → **~1.9× slower than CUDA
FP16**. The fix is to put FP16 *in the graph*: emit a mixed-precision trunk where the big
convs run FP16 and the numerically-sensitive reductions/heads stay FP32.

KataGo's hand-built `ModelParser` already threads a `forceFP32` flag that partitions the
graph on exactly that boundary, and its `applyCastLayer(kFLOAT)` calls sit on the FP16→FP32
crossings — so the port emits `__half` weights for the trunk, casts the inputs to FP16 at
entry, and lets those existing casts anchor the FP32 heads.

## Results

`katago benchmark`, all at builder optimization level 5, b28c512nbt @ 19×19:

| threads / batch | **FP8** | FP16 | CUDA FP16 |
|---|---|---|---|
| 32 / 16  | **1922** | 1292 | 1193 |
| 64 / 32  | **1856** | 1381 | 1320 |
| 128 / 63 | **1625** | 1188 | 1155 |

*(visits/s; raw `nnEvals/s` shows FP8 at ~1340–1360 vs FP16 ~925–990.)*

Accuracy vs a CUDA FP32 reference (max abs deviation over fixed positions):

| engine | winrate | scoreLead | policy |
|---|---|---|---|
| FP16 mixed | 7e-4 | 3.1e-2 | 1.7e-2 |
| **FP8** | 1.0e-2 | **0.077** | 2.3e-2 |
| CUDA FP16 | 4e-3 | 2.3e-2 | 1.9e-2 |

*(winrate deviations sit near the random-symmetry noise floor and wobble run-to-run;
scoreLead and policy are the stable signals — FP8 lands close to FP16/CUDA on both.)*

## Precision ladder

| precision | NN throughput vs FP16 | status |
|---|---|---|
| FP32 / TF32 | 0.5× | correct, slow (no weakly-typed FP16 on RTX) |
| **FP16 (mixed)** | 1.0× | ✅ default; ≈/> CUDA at batch ≥ 32 |
| **FP8** (`trtTrunkPrecision=fp8`) | ~1.4× | ✅ +35–45 % nnEvals, analytic per-conv activation scale |
| FP4 (NVFP4) | <1× here | ⚠️ executes (crash fixed) but 1×1-only + PTQ too lossy → needs NHWC + QAT |

FP8 quantizes the big residual convs (per-output-channel static weight scale from the
weights, plus a per-tensor activation scale sized analytically from the preceding folded
BatchNorm's running stats — **calibration-free**), keeping heads/reductions in FP32.

## Build

Requires CUDA + a TensorRT-RTX install (headers + libs). On the GB10 box:
CUDA 13, TensorRT-RTX 11 headers in `/usr/include/aarch64-linux-gnu`.

```bash
./bench/build.sh TENSORRT      # -> build/trt/katago   (this fork)
./bench/build.sh CUDA          # -> build/cuda/katago  (the A/B baseline)
```

The CMake `TENSORRT` block auto-detects the RTX SDK; the backend compiles under the
`KATAGO_TRT_RTX` guard (`NV_TENSORRT_MAJOR >= 11`). It falls back to the normal mainline
TensorRT path on older SDKs. Or configure directly:

```bash
cmake -S cpp -B build/trt -DCMAKE_BUILD_TYPE=Release -DUSE_BACKEND=TENSORRT \
      -DCMAKE_CUDA_ARCHITECTURES=121   # Blackwell sm_121
```

## Run it

```bash
# FP16 (default) — just point KataGo at a net
build/trt/katago benchmark -model kata1-b28c512nbt-*.bin.gz -config bench/results/bench.cfg

# FP8 trunk (convnets; hand-built ModelParser path)
#   trtDisableOnnx = true
#   useFP16 = true
#   trtTrunkPrecision = fp8
```

- [`bench/validate.sh`](bench/validate.sh) — correctness gate: net evals vs a CUDA FP32
  reference (FP16 + FP8), exits non-zero on divergence. **Run before trusting a speed number.**
- [`bench/run.sh`](bench/run.sh) — throughput sweep across FP8 / FP16 / CUDA and thread counts.
- [`bench/RESULTS.md`](bench/RESULTS.md) — the full write-up.

## What changed vs upstream

Everything lives in one file plus a one-line fix:

- [`cpp/neuralnet/trtbackend.cpp`](cpp/neuralnet/trtbackend.cpp) — the TensorRT-RTX port:
  removed-API guards, strongly-typed FP16 mixed-precision trunk, FP8/NVFP4 quantized convs,
  analytic activation calibration.
- [`cpp/command/sandbox.cpp`](cpp/command/sandbox.cpp) — `kEXPLICIT_BATCH` removal.
- [`bench/`](bench/) — build / validate / benchmark harness + results.

Config knobs added: `trtTrunkPrecision = fp16 | fp8 | fp4`, and
`trtTrunkPrecisionExperimentalFp4 = true` for the FP4 experiment.

## Honest limits

- **FP16 / FP8 are ModelParser-only** (convnets; enable FP8 with `trtDisableOnnx=true`).
  Transformer nets go through KataGo's ONNX emitter, which still builds FP32/TF32 on RTX —
  a strongly-typed FP16/FP8 ONNX emitter is the follow-up to cover those.
- **FP4 (NVFP4) executes but isn't a practical win.** The layout wall is beatable without a
  full NHWC trunk (reshape `[N,C,H,W]→[N,C,H·W]` so C is 2nd-to-last), and the RTX execution
  crash is fixed by *baking* the weights as pre-quantized FP4 (E2M1) + FP8 (E4M3) block-scale
  constants into an `IDequantize` (dynamic-quantizing weights segfaults inside `libnvinfer`).
  But the clean reshape only reaches **1×1 convs** (slower than FP16 there), and FP4
  post-training quantization is too lossy for this net (scoreLead ~0.47 from the 1×1 convs
  alone). A useful FP4 needs an **NHWC trunk** for the 3×3 convs *and* **quantization-aware
  training** for accuracy.
- FP8 accuracy is net-eval-verified near FP16; the final play-readiness check is a strength
  gauntlet.
- Numbers are single-GPU b28 at 19×19; other nets / board sizes shift the crossover batch.

## Credits & license

This is KataGo with a TensorRT-RTX backend added. **KataGo** is by David J. Wu
([@lightvector](https://github.com/lightvector)) and contributors — see [`LICENSE`](LICENSE)
and the upstream README at [`README-KATAGO-UPSTREAM.md`](README-KATAGO-UPSTREAM.md). The
TensorRT-RTX port and benchmark harness are added here under the same license. Sibling to
[katago-webgpu](https://github.com/saigo-online/katago-webgpu) (a WebGPU/WASM KataGo backend).
