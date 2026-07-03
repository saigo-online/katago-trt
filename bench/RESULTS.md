# katago-trt — TensorRT-RTX vs CUDA on b28 (NVIDIA GB10)

**Question:** does KataGo benefit from a TensorRT backend on this box?
**Net:** `kata1-b28c512nbt` (b28c512, nested-bottleneck convnet), 19×19.
**Box:** NVIDIA GB10 (Grace-Blackwell, aarch64, `sm_121`), CUDA 13.0, **TensorRT-RTX 11.0.0.114**.

## TL;DR

TensorRT already ships a backend — but it does **not** build against this box's SDK
unmodified, and once building, it only pays off **with FP16**, which required a real port.
After the port, **TensorRT-RTX FP16 beats the CUDA FP16 backend at batch ≥ 32** (+4–6% on
visits/s, +6–7% on raw NN evals) and trails only at small batch. Yes — KataGo benefits.

## Throughput (visits/s, higher is better)

| threads / batch | TensorRT-RTX FP16 | CUDA FP16 | TensorRT-RTX FP32/TF32 |
|---|---|---|---|
| t=16 / b≈8   | 858  | **978**  | 528 |
| t=32 / b≈16  | 1023 | **1074** | 567 |
| t=64 / b≈32  | **1340** | 1286 | — |
| t=128 / b≈63 | **1215** | 1143 | — |
| t=256 / b≈148 | **1048** | 1024 | — |

Raw NN throughput (nnEvals/s), the cleanest backend metric, at the same points:

| threads / batch | TensorRT-RTX FP16 | CUDA FP16 |
|---|---|---|
| t=64 / b≈32  | **982** | 928 |
| t=128 / b≈63 | **987** | 924 |
| t=256 / b≈148 | **944** | 920 |

FP16 lifted TensorRT-RTX by **+62%** at t=16 (528→858). The crossover with CUDA is around
batch 16–32: CUDA is faster at small batch, TensorRT-RTX FP16 wins at batch ≥ 32 (+4–6% on
visits/s, +6–7% on raw nnEvals/s). visits/s peaks near t=64 for both backends and then falls
off as MCTS thread overhead grows — that's search bookkeeping, not the net, hence the cleaner
nnEvals/s comparison.

## Correctness (max abs deviation vs CUDA FP32 ground truth)

| backend | winrate | scoreLead | policy |
|---|---|---|---|
| **TensorRT-RTX FP16 (mixed)** | 6.6e-4 | 1.6e-2 | 1.4e-2 |
| CUDA FP16 | 2.1e-4 | 2.3e-2 | 3.0e-2 |
| TensorRT-RTX FP32/TF32 | 4.7e-3 | 3.8e-2 | 2.0e-2 |

The mixed-precision FP16 build (FP16 trunk, FP32-pinned reductions/heads) is the **most
accurate** non-reference engine — the FP32 head pinning beats even plain TF32 on winrate.

## Why the first attempt looked 1.9× slower

TensorRT-RTX (the SDK on DGX Spark / GB10) removed mainline TensorRT's precision knobs:
`ILayer::setPrecision`/`setOutputType`, `ITensor::setType`, `IBuilder::platformHasFastFp16()`,
`BuilderFlag::kFP16`, and `kOBEY_/kPREFER_PRECISION_CONSTRAINTS`. Networks are now **always
strongly-typed**: precision comes from the *tensor types in the graph*, not a builder flag.

The initial port just guarded those calls out, yielding an all-FP32 graph → RTX ran it in
FP32/TF32 → ~1.9× slower than CUDA FP16. That was a porting artifact, not a TensorRT verdict.

## The fix: a strongly-typed mixed-precision graph

The port emits FP16 in the hand-built `ModelParser` (see `trtbackend.cpp`, `useHalfTrunk`):

- **FP16 trunk** — conv/matmul/BN weights emitted as `__half`; spatial/global inputs cast to
  FP16 at the trunk entry.
- **FP32 reductions & heads** — KataGo's existing `forceFP32` flag already partitions the graph
  exactly here, and its `applyCastLayer(kFLOAT)` calls sit on the FP16→FP32 boundaries.
- **Shared mask tensors** cast to the consumer's type at each use (cached per tensor).

## How to reproduce

```bash
./bench/build.sh TENSORRT       # build/trt/katago  (RTX port)
./bench/build.sh CUDA           # build/cuda/katago (baseline)
./bench/validate.sh             # correctness gate vs FP32 ground truth (incl. FP16)
./bench/run.sh                  # throughput sweep: TRT-FP16 vs TRT-FP32 vs CUDA
```

The FP16 path uses the ModelParser (`trtDisableOnnx = true`, convnets only) + `useFP16 = true`.

## Headline: FP8 TensorRT-RTX vs stock KataGo CUDA

The bottom line anyone actually cares about — this fork's best (FP8) vs what ships today
(the stock CUDA backend, FP16). Same net, same board, both at their best default precision,
builder opt-level 5, cache cleared per run:

| threads / batch | FP8 TRT visits/s | stock CUDA visits/s | speedup | FP8 nnEvals/s | CUDA nnEvals/s |
|---|---|---|---|---|---|
| 16 / 8   | 1611 | 1189 | **+36%** | 1196 | 855 |
| 32 / 16  | 1844 | 1077 | **+71%** | 1341 | 782 |
| 64 / 32  | 1800 | 1263 | **+42%** | 1334 | 913 |
| 128 / 70 | 1562 | 1105 | **+41%** | 1250 | 894 |

**FP8 TensorRT-RTX runs KataGo ~1.4–1.7× faster than the stock CUDA backend on the GB10**, at
near-FP16 accuracy (scoreLead within ~0.08 of FP32). `cudabackend.cpp` is unmodified upstream, so
this is a fair TRT-vs-stock comparison.

## Going below FP16: FP8 and FP4 on the Blackwell tensor cores

The GB10's peak throughput is at 4-bit (NVFP4); FP8 is the intermediate rung. Both are wired
behind `trtTrunkPrecision = fp8 | fp4` (ModelParser + FP16 trunk only), quantizing the big
residual convs while heads/reductions stay FP32.

**FP8 works and is a real win** (per-tensor static weight scales from the weights + a per-tensor
activation scale; layout-agnostic, so no NHWC needed). Full sweep, all at builder opt-level 5:

| threads / batch | FP8 visits/s | FP16 visits/s | CUDA visits/s | FP8 nnEvals/s | FP16 nnEvals/s |
|---|---|---|---|---|---|
| 32 / 16 | **1922** | 1292 | 1193 | 1342 | 925 |
| 64 / 32 | **1856** | 1381 | 1320 | 1361 | 991 |
| 128 / 63 | **1625** | 1188 | 1155 | 1300 | 961 |

FP8 is **+34–49% on visits/s over FP16** and **+35–45% on raw nnEvals/s** — and ~+40–60% vs the
CUDA FP16 baseline.

**Accuracy — calibration-free per-conv scaling (done).** Weights use a per-output-channel static
scale (from the weights); the activation uses a per-tensor scalar sized analytically from the
preceding folded-BatchNorm's running stats (`|mergedScale·mean + mergedBias| + 6·|mergedScale|·√var`,
conservative). No calibration data or extra forward pass, and it stays on the fast per-tensor FP8
path (a per-channel activation scale is more accurate but blocks the fast conv fusion — measured
~2× slower, so avoided).

| FP8 activation scale | nnEvals/s @ t=64 | winrate Δ | score Δ | policy Δ |
|---|---|---|---|---|
| untuned global constant | 1361 | 6.6e-3 | 0.184 | 2.2e-2 |
| **analytic per-conv (shipped)** | **1372** | 9.8e-3 | **0.077** | 2.3e-2 |
| FP16 reference | 991 | 5.0e-3 | 0.063 | 3.2e-2 |

The scoreLead error is **more than halved to near-FP16 levels** with full speed retained; winrate
and policy stay in tolerance. Final play-readiness is a strength gauntlet (net-eval agreement is
now solid). Enabled with `trtTrunkPrecision = fp8`.

**FP4 (NVFP4) — now EXECUTES correctly (crash fixed), but not a practical win on this net.**
The working recipe, in order of the walls cleared:

1. **Layout.** NVFP4 has no per-tensor path — block-16 microscaling only, and TensorRT-RTX requires
   the blocked dim last/2nd-to-last. Reshaping the activation `[N,C,H,W] → [N,C,H·W]` puts C at the
   2nd-to-last position, clearing the layout check **without an NHWC trunk**.
2. **Activation.** Dynamic block-quantize (`addDynamicQuantize`, runtime FP8 scales) — fine at
   inference.
3. **Weights — the crash fix.** Dynamic-quantizing the weights stays a runtime op and **segfaults
   inside `libnvinfer`**. Instead we **bake** them: pre-quantize offline to FP4 (E2M1) values + FP8
   (E4M3) per-block scales (via CUDA's `__nv_cvt_float_to_fp4/_fp8`, each element quantized against
   its block's *decoded* FP8 scale), feed both as constants straight into `IDequantize` (no
   `IQuantize` — an FP8 scale can't feed one). Constants fold at build time and run cleanly.

That runs end-to-end (any kernel size — 1×1 *and* 3×3), no crash. But two hard limits remain, and
both are now **measured**, not guessed:

- **Slower, not faster — no NCHW shortcut exists.** The reshapes/transposes that satisfy the
  block-layout rule stop TensorRT from fusing a *native* FP4 conv: it dequantizes to FP16, runs FP16,
  and *also* pays the reshape overhead. Quantizing the whole trunk (3×3 + 1×1) gives **836 nnEvals/s
  at t=64 — slower than FP16 (~991), FP8 (~1360), and even CUDA (~961).** Native FP4 tensor-core
  convs need a channels-last (NHWC) trunk so no reshape breaks the fusion.
- **FP4 PTQ is far too lossy.** Post-training 4-bit on the trunk costs **scoreLead ~0.54, policy
  ~0.18** vs FP32 — unplayable (E2M1 is 3 bits of precision). Verified not an encoding bug (flipping
  the nibble order gives score ~163). A usable FP4 net needs **quantization-aware training**.

So FP4 is a solved *execution* problem (`buildFP4Conv` handles any kernel), but making it a practical
win needs **both** an NHWC trunk (for the speed — the same NHWC groundwork KataGo's ONNX path already
has) **and** QAT (for the accuracy — a training-pipeline effort, not a backend one). Neither is a
bounded backend change. `trtTrunkPrecision=fp4` rejects with this summary;
`trtTrunkPrecisionExperimentalFp4=true` runs it.

## Precision ladder, measured

| precision | NN throughput vs FP16 | status |
|---|---|---|
| FP32/TF32 | 0.5× | correct, slow (no FP16 on RTX weakly-typed) |
| FP16 (mixed) | 1.0× | ✅ shipped, ≈/> CUDA at batch ≥ 32 |
| FP8 (per-tensor) | ~1.4× | ✅ +35–45% nnEvals, analytic per-conv activation scale (near-FP16 accuracy) |
| FP4 (NVFP4) | 0.6× here | ⚠️ executes any-kernel (crash fixed), but reshapes block FP4 fusion → *slower* than FP16, and PTQ unplayable → needs NHWC + QAT |

## Caveats / future work

- **FP8 strength gauntlet** — net-eval accuracy is now near-FP16 (scoreLead 0.077); the final
  play-readiness check is Elo vs the FP16 build. (The activation scale is analytic per-conv, done.)
- **NHWC trunk** — the real unlock for NVFP4 (GB10 peak) and MXFP8 block microscaling. The 3D-reshape
  trick beats the *layout* wall but RTX crashes executing the fused FP4 conv; a channels-last trunk
  with baked NVFP4 weight constants avoids the setInput+dynamic-quant path that crashes, and covers
  the 3×3 convs (the FLOP bulk) that the reshape trick misses.
- **FP16/FP8 are ModelParser-only** (convnets). Transformer nets go through the ONNX emitter, which
  still builds FP32/TF32 on RTX; a strongly-typed FP16/FP8 ONNX in `onnxmodelbuilder.cpp` covers them.
- Numbers are single-GPU b28 at 19×19; other nets/board sizes may shift the crossovers.
