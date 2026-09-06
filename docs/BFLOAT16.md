# BFloat16 in Nivara

`System.Numerics.BFloat16` is a brain-floating-point format: an 8-bit exponent
(wide dynamic range, the same as `float`) with a 7-bit mantissa (low
precision). On **.NET 11** it implements `IBinaryFloatingPointIeee754<BFloat16>`,
so it natively satisfies the `IFloatingPointIeee754<T>` constraint that Nivara's
AutoDiff and numeric kernels are built around, and the BCL `TensorPrimitives`
exposes `BFloat16` arithmetic overloads.

> **Vectorization note — read this first.** `BFloat16` (like `Half`) is **first-class
> and correct** in Nivara, but **not SIMD-accelerated** on .NET 11. The phrase
> "TensorPrimitives supports BFloat16" is true in one sense and false in another, and that
> ambiguity is the usual source of confusion:
>
> - **Type support (TRUE).** `TensorPrimitives` exposes *generic* overloads such as
>   `Add<T>`, `Multiply<T>`, `Dot<T>`, `Sum<T>` constrained only to
>   `IAdditionOperators<T,T,T>` + `IAdditiveIdentity<T,T>`. `BFloat16` satisfies those
>   interfaces (it implements `IBinaryFloatingPointIeee754<BFloat16>`), so
>   `TensorPrimitives.Add<BFloat16>` **compiles and returns correct results** — no
>   `NotSupportedException`.
> - **SIMD acceleration (FALSE).** Microsoft documents that TensorPrimitives is
>   *"hardware-accelerated for the element types that `Vector<T>` supports
>   (`Vector<T>.IsSupported`)."* Because `Vector<BFloat16>.IsSupported` is `false` at every
>   width (64/128/256), the BCL cannot build a `Vector<BFloat16>` and silently drops to its
>   **scalar** fallback loop. The operation is correct — just not vectorized.
>
> So "BFloat16 is supported in Nivara" means: *the same generic code paths run end-to-end with
> correct results*, **not** that the kernel is vectorized. `Half` is in the identical situation.
> The measured cost (matmul ~26× slower than F32) is in *SIMD / Vector Lane Support* under
> [Precision & limitations](#precision--limitations) below.

| `TensorPrimitives` aspect | `BFloat16` / `Half` | `float` / `double` |
| --- | --- | --- |
| Generic overloads compile & run (`Add<T>`, `Dot<T>`, …) | Yes | Yes |
| Result correctness | Correct | Correct |
| SIMD / `Vector<T>` fast path | No — scalar BCL fallback | Yes |
| Relative throughput | ≈ hand-written scalar loop | much faster |

> **Hints for .NET `Numerics` engineers.**
> - `TensorPrimitives` is an *encapsulated* dispatcher: per element type `T` it picks the
>   SIMD kernel when `Vector<T>.IsSupported` is `true`, otherwise a scalar loop. **You don't
>   choose — the runtime does, per `T`.** It always "tries," but the best available path for
>   `BFloat16`/`Half` *is* the scalar fallback, so "supported" there means *correct*, not
>   *accelerated*.
> - `BFloat16`/`Half` implement the generic-math interfaces (`IAdditionOperators`,
>   `IBinaryFloatingPointIeee754`, …), so every **generic** `TensorPrimitives` overload
>   (`Add<T>`, `Dot<T>`, `Sum<T>`, …) compiles and returns correct results. Type-level
>   support ≠ acceleration.
> - SIMD in .NET is `Vector<T>`-gated. `Vector<BFloat16>`/`Vector<Half>` report
>   `IsSupported == false` on .NET 11 at every width. No `Vector<T>` support ⇒ no SIMD kernel
>   in `TensorPrimitives`, however "native" the type is — `BFloat16` is a first-class runtime
>   type, just not a *vectorizable* one.
> - The BCL does **not** auto-widen narrow floats to `float` to gain SIMD.
>   `TensorPrimitives.Add<BFloat16>` operates on `BFloat16` *as* `BFloat16` and falls back to
>   scalar. (Only the explicit `ConvertToSingle` / `ConvertChecked` helpers widen.)
> - To actually vectorize BF16/Half you must do it yourself: reinterpret the 16-bit values into
>   `float` lanes — `BFloat16` is losslessly the top half of `float32`; `Half` via the standard
>   conversion — run the genuinely-SIMD `TensorPrimitives<float>`, then narrow back. The BCL
>   won't do this for you; it's the manual path a library like Nivara can add (see the
>   *SIMD / Vector Lane Support* note under [Precision & limitations](#precision--limitations)).
> - Watch `Vector<BFloat16>` in future .NET: if it ever flips to supported, the BCL scalar
>   fallback disappears with **no Nivara code change**.

Nivara supports `BFloat16` in **two layers**:

1. **The column / query-analytics layer** — `BFloat16` is now a first-class numeric column
   type: element-wise arithmetic (scalar, see note), window functions, sorting, aggregation, and
   fused query expressions.
2. **The AutoDiff domain** — `BFloat16` is a first-class gradient type: it flows through every
   op, optimizer, and module.

This document covers both. Related references:
[`AUTODIFF.md`](AUTODIFF.md),
[`TENSORS.md`](TENSORS.md) (type-support note),
[`INTEGERS.md`](INTEGERS.md)

---

## Why BFloat16

- **Memory**: half the storage of `float` — attractive for large columnar
  datasets and model activations/gradients.
- **Range**: the same exponent range as `float32`, so it survives the wide
  dynamic range of gradients and normalized activations without under/overflow
  (where `Half` can be marginal).
- **.NET 11**: the type is built in and `TensorPrimitives` exposes `BFloat16`
  overloads, so Nivara's generic code paths apply with **no `NotSupportedException`**.
  They are **not** SIMD-accelerated: because `Vector<BFloat16>` is unsupported, the BCL
  executes a scalar loop for `BFloat16`/`Half` arithmetic and reductions (see the
  *Vectorization note* above and *SIMD / Vector Lane Support* below).

For the concrete end-to-end numbers (weight memory F32 vs FP16/BF16 and the
accuracy-vs-reference table), see the *Narrow-precision inference* section of
`samples/NivaraInference/README.md`. Both FP16 (`Half`) and BF16 halve weight
memory (2 B/param vs `float`'s 4): the sample measures ~91→~45.5 MB (MiniLM) and
~255→~128 MB (DistilBERT / SST-2).

---

## Column / query-analytics layer

### Typed columns

```csharp
var col  = NivaraColumn<BFloat16>.Create(new BFloat16[] { (BFloat16)1.5f, (BFloat16)2.5f, (BFloat16)3.5f });
var ncol = NivaraColumn.CreateFromNullable(new BFloat16?[] { (BFloat16)1.5f, null, (BFloat16)3.5f });
```

`BFloat16` is recognized as a numeric type everywhere the type system dispatches:
`NumericKernelDispatcher.arithmeticDomain`, `NumericPromoter`
(`BFloat16` promotes to `float`/`double` like `Half` — `BFloat16 + int` →
`double`, since there is no implicit `BFloat16`↔integral conversion),
`TypeCompatibilityValidator.GetNumericTypes`, and `TypeExtensions.IsNumericType`.

### Element-wise arithmetic (null-mask preserved)

```csharp
var scaled = col.Multiply((BFloat16)2.0f);   // [3.0, 5.0, 7.0] via the generic TensorPrimitives path (scalar for BF16)
var ratio  = col.Divide((BFloat16)2.0f);      // [0.75, 1.25, 1.75]
var added  = col.Add(otherColumn);            // column-on-column
```

Arithmetic runs through the same generic `TensorPrimitives` path as `Half`. On
.NET 11 `Vector<BFloat16>` is unsupported, so the BCL runs a **scalar** loop rather
than a SIMD kernel — the result is correct and first-class, just not vectorized
(see the *Vectorization note*). **Null masks are preserved** (a null input position
yields a null output, like every other numeric type).

### Window functions

```csharp
var frame  = NivaraFrame.Create(("v", NivaraColumn<BFloat16>.Create(new BFloat16[] { (BFloat16)1, (BFloat16)2, (BFloat16)3 })));
var rolled = frame.RollingSum("v", "sum", 3);   // typed BFloat16 output column
var cum    = frame.CumulativeSum("v", "cum");
```

`Rolling*` / `Cumulative*` / `Shift` / `Lead` all accept `NivaraColumn<BFloat16>`
via `Over()`/`WindowSpec`.

### Sorting

```csharp
var sorted = frame.OrderBy("v");             // multi-column sort + comparers support BFloat16
```

### Aggregation (precision-promoted to `double`)

`Sum`, `Mean`, `Quantile`, and `Median` over a `BFloat16` column all produce
`double` (the same precision-preserving promotion `Half` uses):

```csharp
var allRows = Enumerable.Range(0, col.Length).ToList();
AggregationFunctions.Sum().Apply(col, allRows);          // double
AggregationFunctions.Mean().Apply(col, allRows);         // double
AggregationFunctions.Quantile(0.5).Apply(col, allRows);  // double (median)
AggregationFunctions.Median().Apply(col, allRows);       // double
```

### Fused query expressions

```csharp
using Nivara.Expressions;

var input = new Dictionary<string, IColumn> { ["A"] = col };
var fused = new FusedExpressionEvaluator();

// Same-type expression runs through the fused evaluator
var sameType = fused.Evaluate(ColumnExpressions.Col("A") + ColumnExpressions.Col("A"), input);
// sameType.ElementType == typeof(BFloat16)

// Mixed BFloat16 + int promotes to double (safe superset, like a C# error pair)
var promoted = fused.Evaluate(ColumnExpressions.Col("A") + 1, input);
// promoted is NivaraColumn<double>
```

### NivaraSeries

`NivaraSeries` gains a `BFloat16` conversion arm so quantile/aggregation over
series works as well.

> **Comparisons** (`>`, `<`, `==`) on `BFloat16` columns fall back to
> `Comparer<T>.Default` (scalar, not SIMD) — identical to `Half`, which is also
> absent from the comparison fast-path domain. They are correct, just not
> vectorized.

---

## AutoDiff domain

### What works

`BFloat16` is admitted into `TypeValidator`'s supported set, so it is treated
exactly like `float`/`double`/`Half` across the autograd engine:

- **All operations** — element-wise, `MatMul` (runs through the BCL
  `TensorPrimitives.Dot` row dot-product; the old hand-rolled `Vector<T>` SIMD
  branch that threw `NotSupportedException` for `BFloat16` was removed — note this
  `Dot` is the **scalar** BCL path for `BFloat16`/`Half`, since `Vector<BFloat16>`
  is unsupported), reductions, normalization, activations, attention, convolutions, VAE/Transformer
  modules.
- **Optimizers** — `SGD<BFloat16>`, `Adam<BFloat16>`, `AdamW<BFloat16>` with
  their `TensorPrimitives`-based state buffers.
- **Modules** — `Linear<BFloat16>`, `Sequential<BFloat16>`, `Embedding`,
  `Conv1d/2d`, `BatchNorm`, `LayerNorm`, `TransformerBlock`, `VAE`, etc., since
  they are all generic over `T : struct, IFloatingPointIeee754<T>`.
- **Transformer token-ID correctness** — `Embedding<T>` (and `BertEncoder<T>`,
  `MiniLMDistilled<T>`, `DistilBertForSequenceClassification<T>`) take token IDs
  as **exact `int[]`** via `Forward(int[] tokenIds, ...)` overloads. BFloat16 (and
  `Half`) cannot represent vocabulary indices (~30k) exactly — only integers up to
  256 — so passing token IDs as a `T` tensor before the embedding lookup corrupts
  them (e.g. `30522 → 30512`) and produces garbage output (~7 logit diff vs the
  F32 reference). Keeping the indices as `int` (independent of the compute dtype)
  makes BFloat16/Half transformer inference correct; the existing
  `ReverseGradTensor<T>` overloads remain for F32/F64. End-to-end,
  `DistilBertForSequenceClassification<BFloat16>` matches the F32 HuggingFace
  reference at **8/8 argmax** with a **~0.33 max logit diff**.
- **Frame → tensor batch** — `ToReverseGradTensorsAuto` now converts `BFloat16`
  frame columns (it previously skipped them).
- **Model serialization** — state dicts persist `BFloat16` weights via
  base64-encoded binaries.

### SafeTensors

`SafeTensorsLoader` is **dtype-aware**: it reads each tensor's `dtype` from the
header and converts to the requested result type `T` (via `T.CreateChecked`).
The non-generic `Read()` returns `float[]` and feeds the `Module<float>`
pipeline directly. What actually happens depends on the *on-disk* dtype:

| On-disk `dtype` | Default `Read()` → `float[]` | `Read<BFloat16>()` → `BFloat16[]` |
|---|---|---|
| `F32` | no-op reinterpret (`ConvertF32<float>`) | **F32 → BF16 truncation** (23-bit → 7-bit mantissa) — the `bf16` mode |
| `BF16` | **BF16 → F32 widening** (lossless — BF16 *is* the top 16 bits of float32) | **lossless, zero-hop** — raw bytes reinterpreted via `MemoryMarshal.Cast<byte, BFloat16>` (no F32 intermediate) |
| `F16` (`Half`) | **F16 → F32 widening** (lossless) | F16 → F32 → BF16 (two conversions) |

- **BF16 checkpoint + default `Read()`** → `ConvertBF16<float>` widens each
  `BFloat16` to `float` by placing its 16 bits in the high half of a `float32`
  (lossless — BF16 *is* the top 16 bits of float32).
- **F32 checkpoint + default `Read()`** → `ConvertF32<float>` is a no-op
  reinterpret (no widening at all).
- **`Read<BFloat16>()` on a BF16 checkpoint** → the raw bytes are reinterpreted
  directly as `BFloat16` via `MemoryMarshal.Cast<byte, BFloat16>` (the
  `ConvertBF16ToBFloat16` fast path). No F32 hop is needed because the on-disk
  16-bit pattern already *is* the `BFloat16` memory layout — bit-for-bit
  identical to the old BF16 → F32 → BF16 route, but without the redundant
  shift/reinterpret/`CreateChecked` per element.
- **`Read<BFloat16>()` on an F32 checkpoint** → `ConvertF32<BFloat16>` performs
  genuine **F32 → BF16 truncation** (23-bit → 7-bit mantissa) — exactly what the
  `NivaraInference` sample's `bf16` mode does to run real BFloat16 inference.

Why the F32 default: float32 is a strict superset of BF16's value set (same
8-bit exponent), so widening never loses information and feeds full-precision
F32 compute. This is a **deliberate precision/compatibility default, not a
workaround** — on .NET 11 `BFloat16` is a native type with `TensorPrimitives`
support, so the non-widening path (`Read<BFloat16>` + `Module<BFloat16>`) is
fully available and exercised by the sample's `bf16` mode. Flipping the default
to track the on-disk dtype would break the `float[]` API contract the F32 model
builders depend on.

### Fused BF16 read path (`Read<float>`) + SIMD widening

For BF16 checkpoints that ask for F32 inference, the loader fuses the lossless
BF16→F32 widen directly into the read — never materializing an interim `ushort[]`:

- `SafeTensorsLoader.Read<float>(path)` parses the header and, for each BF16
  tensor, widens the raw 16-bit patterns straight into the destination `float[]`
  via `WidenBf16ToF32(ReadOnlySpan<ushort>, Span<float>)` — a `Vector<ushort>`
  SIMD chain (`Vector.Widen` → `<<16` → bit-reinterpret) with a scalar tail. It
  property-matches the scalar reference for **all 65,536** BF16 patterns;
  `ConvertBF16<float>` routes through the same kernel, so the F32 read path is
  SIMD for free (one pass, no interim `ushort[]`, ~1 GB less peak memory than a
  two-step raw read + widen).
- The Qwen2.5-0.5B checkpoint (988 MB BF16, F32-target load) loads in roughly
  **0.7–2.2 s** on this machine (Release; the OS file cache drives the spread) —
  **no regression** vs the earlier two-step, with identical F32 inference numerics
  (widening is lossless — BF16 is the high 16 bits of float32). The two-step's
  documented "~2.5× faster / half the memory" claim was an apples-to-oranges
  comparison (half-size `ushort[]` output with no widen vs full-size `float[]`
  with widen); at equal F32 output the two-step offered nothing and was removed
  (#388).

See `docs/QWEN.md` (Phase 2.5) and
`SafeTensorsLoaderBf16Tests` for the full context.

### Example — train a `BFloat16` linear model

```csharp
using System.Numerics;
using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;

// BFloat16 is a supported AutoDiff type
TypeValidator.IsSupportedType(typeof(BFloat16));   // true
NivaraAutoGradExtensions.IsAutoGradSupported<BFloat16>(); // true

// Trainable weights in BFloat16
var w = new ReverseGradTensor<BFloat16>(
    NivaraColumn<BFloat16>.Create(new BFloat16[] { (BFloat16)0.5f, (BFloat16)0.5f }),
    requiresGrad: true);

var x = ReverseGradTensor<BFloat16>.FromArray(new BFloat16[] { (BFloat16)1.0f, (BFloat16)2.0f });
var y = ReverseGradTensor<BFloat16>.FromArray(new BFloat16[] { (BFloat16)3.0f, (BFloat16)5.0f });

using (GradientUtils.Grad())
{
    var pred = ReverseGradOperations.Multiply(x, w);            // [0.5, 1.0]
    var diff = pred - y;                                        // [-2.5, -4.0]
    var loss = ReverseGradOperations.Mean(ReverseGradOperations.Multiply(diff, diff));
    loss.Backward();                                            // fills w.Grad
}
// w.Grad now holds BFloat16 gradients
```

The `BFloat16Tests` suite verifies forward/backward parity with `float`
references, `Linear<BFloat16>` training under `SGD`/`Adam`, and the
inference-default graph guard.

---


## What you could not do before

- **Column / query:** `NivaraColumn<BFloat16>` arithmetic threw
  `NotSupportedException` (it was absent from `NumericKernelDispatcher.arithmeticDomain`);
  `ExpressionTypeInferer` excluded `BFloat16` from the fused evaluator; and
  aggregation, quantile, window functions, and sorting had no `BFloat16` arm.
  All of those are now wired (mirroring `Half`).
- **AutoDiff:** every `BFloat16` autograd operation threw
  `NotSupportedException`. The hand-rolled `Vector<T>` matmul SIMD branch
  rejected `BFloat16`, and `TypeValidator` excluded it. Now it is admitted at
  runtime and exercises the BCL `TensorPrimitives.Dot` path.

---

## Precision & limitations

- **Low precision (like `Half`).** Aggregation promotes to `double` to avoid
  loss; keep this in mind for cumulative sums over many rows.
- **Comparisons are scalar**, not SIMD — same as `Half`.
- **Frame convenience ops with their own type switches** (e.g. the data-prep
  `Normalize` / `Standardize` helpers) may still not support `BFloat16`. This is
  consistent with how `Half` is treated and is a known follow-up, not a
  regression.
- **Non-nullable at the AutoDiff boundary (ADR-001).** Resolve nulls
  (`FillNull` / `DropNulls`) before converting a `BFloat16` column to a gradient
  tensor.
- **SIMD / Vector Lane Support (empirical, .NET 11, net11 `System.Numerics.Tensors` 11.0.0-preview.7)**: `BFloat16` and `Half` are scalar-first-class but SIMD-second-class. `Vector<BFloat16>`.IsSupported = false (all widths 64/128/256); `Vector.Create<BFloat16>` throws `NotSupportedException`. Matmul (`TensorsHelper.MultiplyCore<T>`) routes only `float`/`double` to SIMD row-dot kernels; by default `BFloat16`/`Half` fall to scalar `MultiplyRowScalar` (~26× slower, verified via `NivaraInference` benchmark). **Auto-widened SIMD path (Phase 1, branch `khurram/smollm-1`):** `src/Nivara/Primitives/` provides `WidenPrimitives` + `NarrowFloatKernels` that bit-reinterpret `ushort` lanes, widen to `float`, run in `float`, and narrow back for element-wise `Add/Sub/Mul/Div` (`NumericTensorKernels`) and matmul row-dot (`TensorsHelper.MultiplyCore`, lifts AutoDiff). Gated behind `NivaraPrimitives.UseWidenSimd` (default **off**), so default behavior is unchanged until flipped; `ShouldWiden<T>` also applies a length gate + `Vector128.IsHardwareAccelerated` check (per-op, no temp `float` buffer — fused per-vector). **Phase 3 A/B (SmolLM-135M, 32 greedy tokens, `samples/NivaraInference smollm ab`):** BF16 scalar ~225 s vs BF16 widen **median 22.6 s** (~10× faster) and vs F32 native ~10.7 s — so BF16+widen is roughly 1.5–2× slower than F32 native (widen does F32 compute plus widen/narrow conversion; the exact ratio varies with machine load), making BF16 a memory-halving convenience rather than a compute win on this workload. The toggle is transparent to F32 (identical token streams). Correctness unchanged: SmolLM BF16 22/32 argmax + 0.937 logits cosine vs PyTorch; `distilbert_sst --precision bf16` argmax 8/8 with and without `--simd-widen`.

---
