# Nivara Automatic Differentiation (AutoDiff)

Nivara provides a PyTorch-inspired reverse-mode automatic differentiation engine built on top of its columnar DataFrame types. Tensors wrap `NivaraColumn<T>` and the computation graph is a DAG of operation nodes built implicitly during forward operations.

Beyond the core autograd engine, Nivara delivers a full training stack: module system, loss functions, stateful optimizers, data loading, training loops, serialization, and data-parallel training — all implemented in ~50 files under `src/Nivara/AutoDiff/`.

---

## Icebreaker: Honest comparison with PyTorch

A cross-framework parity exercise (see `samples/README.md`) trained an
identical 3-layer MLP in both Nivara and PyTorch and compared results.

**Correctness: proven.** Loss curves match within 0.04% relative diff across
50 epochs. The gradient math, optimizer (Adam), and training loop are correct.

**Developer experience: PyTorch still wins comfortably.** The gaps hit during
the exercise:

| Dimension | PyTorch | Nivara | Impact |
|-----------|---------|--------|--------|
| Model registration | `self.l1 = nn.Linear(8, 64)` — auto-registered | `L1 = new Linear<float>(8, 64)` + manual `RegisterModules(...)` | Easy to forget; no compiler error if you do |
| Forward pass | `torch.relu(self.l1(x))` — direct | `ReverseGradOperations.Relu(L1.Forward(x))` — extra ceremony | More typing, harder to read |
| Generics | None (dynamic) | `Module<float>`, `Linear<float>`, `ReverseGradTensor<float>` everywhere | Type pollution propagates through all code |
| Weight loading | `param.data.copy_(tensor)` — one-liner | Custom JSON flatten + manual name mapping (PyTorch name → Nivara key) | ~40 lines of fragile boilerplate, easy to mismatch |
| Optimizer API | `optimizer = Adam(model.parameters())` — unambiguous | Optimizers now register owning `Parameter<T>` objects via `model.GetParameters().Values`; tensor dictionaries are for inspection/initialization, not training | Safer than the old API, still more verbose than PyTorch |
| Error messages | Polished after a decade | Generic operation-node stack traces | Harder to debug autograd failures |
| Ecosystem | TensorBoard, torchinfo, etc. | `result.PrintSummary()` — minimal | Fine for small models, insufficient at scale |

**Resolved optimizer trap:** The old tensor-dictionary optimizer overload was
removed. Training registration should use owning parameter wrappers:
`optimizer.AddParameterGroup(model.GetParameters().Values)`. `Parameters()`
continues to return tensor dictionaries for inspection, initialization, and
serialization-oriented workflows.

**Weight divergence is expected but disorienting.** Even with identical
seeded initialization, SGD trajectories diverge between frameworks (different
BLAS kernels, FP accumulation order). The loss curves stay aligned, proving
the gradient computation is correct, but a first-time user comparing weights
directly will see large differences (e.g., L2.Weight 180% relative diff) and
may incorrectly conclude something is broken.

**Verdict for this example:** A for correctness, B for usability. The hard
part (backprop, optimizer, training infrastructure) works. The easy part
(ergonomics, discoverability, documentation) shows PyTorch's decade of
iteration.

---

## Architecture

```
Core Engine
───────────
GradTensor<T>                         ← Base: Data, Shape, Reshape, ToColumn/ToSeries/AsTensor
├── ReverseGradTensor<T>              ← Adds: Grad, RequiresGrad, GradFn, Backward(), Detach()
└── ForwardGradTensor<T>              ← Adds: Tangent, RequiresTangent; operator overloads (+,-,*,/)

OpNode<T>                             ← One operation node in the graph
├── OperationName                     ← "Add", "MatMul", "Relu", etc.
├── Inputs                            ← Parent tensors (object[])
├── BackwardFunction                  ← Action<NivaraColumn<T>> (local gradient rule)
├── ShouldSaveForBackward             ← Whether to save values for backward pass
├── SavedValues                       ← Dictionary of saved forward values
└── Apply(gradOutput)                 ← Invokes the backward function

ComputationGraph                      ← Graph traversal engine (internal)
├── AddNode(output, opNode)           ← Attaches GradFn to output tensor
├── Backward(tensor, gradient)        ← Topological sort + reverse traversal
├── TopologicalSort(root)             ← DFS with cycle detection
├── ZeroGrad(tensor)                  ← Recursively clears gradients
└── GetGraphInfo(root)                ← Returns diagnostic summary

ReverseGradOperations                 ← Reverse-mode forward+backward ops (static methods)
├── Element-wise: Add, Subtract, Multiply, Divide, DivideScalar, Clip, Pow
├── Matrix: MatMul, Transpose, TransposeAxes, Slice, Concat, Gather, MultiHeadAttention (fused kernel)
├── Reductions: Sum, Mean, MeanPool
├── Normalization: RMSNorm, PerRowRMSNorm, PerRowLayerNorm (RMSNormKernel, LayerNormKernel)
├── Activations: Relu, LeakyRelu, Sigmoid, Tanh, Gelu (tanh approximation)
├── Unary: Negate, Abs, Exp, Log
├── Regularization: Dropout
├── Embedding: SparseEmbeddingBag
├── Probability: Softmax, LogSoftmax, KlDivergence, SampleNormal
├── Broadcast: BroadcastMultiply, BroadcastAdd
└── ...vectorized via TensorPrimitives where available

ForwardGradOperations                 ← Forward-mode JVP ops (static methods, mirrors ReverseGradOperations)
├── Element-wise: Add, Subtract, Multiply, Divide, DivideScalar, Clip, LeakyRelu
├── Matrix: MatMul, Transpose
├── Reductions: Sum, Mean
├── Activations: Relu, Sigmoid, Tanh
├── Unary: Negate, Abs, Exp, Log
├── Regularization: Dropout
├── Probability: Softmax, LogSoftmax, KlDivergence, SampleNormal
└── ...tangent propagation computed alongside primal

Module System (Nn)
──────────────────
Parameter<T>                          ← Named ReverseGradTensor<T> with requiresGrad=true
Module<T>                             ← Abstract base: Forward(), Parameters(), StateDict(), Train()/Eval()
├── Linear<T>                         ← y = x @ Wᵀ + b (Kaiming-uniform init, bias optional)
├── Sequential<T>                     ← Ordered module chain
├── Activation<T>                     ← ReLU, Sigmoid, Tanh, LeakyReLU wrappers
├── Dropout<T>                        ← Inverted dropout with train/eval toggle
├── Embedding<T>                      ← Lookup embedding (token IDs → vectors)
├── SparseEmbedding<T>                ← Sparse embedding bag (fixed-width active features)
├── Conv1d<T>                         ← 1D convolution: im2col → TensorPrimitives.Dot kernel
├── Conv2d<T>                         ← 2D convolution: tiled im2col → Dot, grouped conv, 1×1 fast path
├── ConvTranspose2d<T>                ← 2D transposed convolution: scatter kernel
├── BatchNorm1d<T>                    ← 1D batch normalization (train/eval modes, running stats; 2D [N,C] + 3D [B,C,L])
├── BatchNorm2d<T>                    ← 2D batch normalization (same pattern as 1D)
├── LayerNorm<T>                      ← Layer normalization (TensorPrimitives.Dot kernel)
├── DepthwiseSeparableConv2d<T>       ← MobileNet-style: depthwise conv + pointwise 1×1
├── MaxPool2d<T>                      ← 2D max pooling
├── AdaptiveAvgPool2d<T>              ← Adaptive average pooling (global pooling for classifier heads)
├── VAE<T>                            ← Variational Autoencoder (encoder → reparameterize → decoder; optional conditioning)
├── ConvVAE<T>                        ← Fully convolutional VAE (Conv2d encoder, 1×1 Conv heads)
├── TransformerBlock<T>               ← Multi-head causal self-attention + GELU MLP (pre-norm)
│   └── NormType enum: RMSNorm (default) | LayerNorm
├── MultiheadAttention<T>             ← Standalone Q/K/V/O attention (self/cross), delegates to fused ReverseGradOperations.MultiHeadAttention
├── LlamaCausalAttention<T>           ← Causal GQA attention + RoPE for Llama/Qwen; optional Q/K/V projection bias (`qkvBias: false`)
├── LlamaDecoderBlock<T>              ← Llama block: RMSNorm → LlamaCausalAttention → SiLU gated FFN; forwards `qkvBias`
├── Sampler<T>                        ← Temperature/top-k categorical sampling
├── TextTokenizer                     ← Vocabulary builder with special tokens
└── Initializers/                     ← Kaiming, Xavier, Uniform, Normal, PyTorchDefault

Pre-built application-level classifiers (`TextClassifierModel<T>`, `TokenClassifierModel<T>`) moved out of core in 1.2.0 to `samples/Nivara.Samples/` (namespace `Nivara.AutoDiff.Nn`). Core provides composable primitives only.

Loss Functions (Nn.Functional; common Loss<T> base + Reduction enum)
─────────────────────────────────────────────────────────────────────
Loss<T>                               ← Abstract base: ctor-defaulted Reduction, Forward(p, t, Reduction) override
Reduction                             ← Sum, Mean (default, PyTorch parity), None
MSELoss<T>                            ← Σ(pred - target)²
L1Loss<T>                             ← Σ|pred - target|
BCELoss<T>                            ← -(y·log(p) + (1-y)·log(1-p)); eps via ctor
BCEWithLogitsLoss<T>                  ← Fused sigmoid + BCE (numerically stable); fused backward via single OpNode
CrossEntropyLoss<T>                   ← Fused log-softmax + NLL; soft or int[] targets; None → per-sample NLL
Activation.Softmax<T> / LogSoftmax<T> ← Dim-aware wrappers (Functional classes merged into Activation)

Optimizers
──────────
Optimizer<T> (abstract)              ← Step(), ZeroGrad(), AddParameterGroup()
├── SGD<T>                           ← Momentum + weight decay + TensorPrimitives fast path
├── Adam<T>                          ← Bias-corrected, β₁/β₂ defaults 0.9/0.999
├── AdamW<T>                         ← Decoupled weight decay (Loshchilov & Hutter 2019)
└── SgdUpdate (static)              ← Single-tensor SgdUpdate helper

Training
────────
TensorDataset<T>                      ← Wraps NivaraFrame, exposes feature/label tensor slices
DataLoader<T>                         ← Batch iteration with shuffle
Batch<T>                              ← { Features, Labels }
TrainingLoop<T>                       ← ForEach-epoch/batch: forward → backward → step → zero_grad
DataParallelTrainer<T>                ← Parallel.For over chunks + gradient sum + optimizer step

Serialization
─────────────
ModelSerializer                       ← Save/Load model state dicts (JSON + base64 binary)
Checkpoint<T>                         ← Epoch + loss + optimizer state + model params

Utilities
─────────
GradientUtils                         ← ZeroGrad, Detach, ClipGradValue/Norm, creators, diagnostics
TypeValidator                         ← Runtime type checking (float/double/Half/BFloat16)
TypeConverter                         ← Cross-type tensor conversion (float ↔ double)

NivaraAutoGradExtensions              ← NivaraColumn/NivaraSeries/NivaraFrame ↔ ReverseGradTensor
```

---

## Key Design Principles

- **`IFloatingPointIeee754<T>` type constraint** — `float`, `double`, `Half`, and `BFloat16` are supported, enforced at compile time by the generic constraint (and at runtime by `TypeValidator.IsSupportedType`). Other numeric types (int, long, etc.) do not satisfy the constraint. For the full BFloat16 capability matrix (AutoDiff **and** the column/query layer), see [`BFLOAT16.md`](BFLOAT16.md).
- **1D storage, shape metadata** — data is always stored as a flat `NivaraColumn<T>`. Shape is metadata (`int[] shape`) with `Reshape()` validation. Default shape is `[Length]`.
- **Inference is the default** — normal `Forward` and `ReverseGradOperations` calls compute values without building a computation graph. `ComputationGraph.AddNode()` asserts graph construction only occurs inside `GradientUtils.Grad()` scope.
- **Training is explicit** — wrap manual training code in `using (GradientUtils.Grad())`; inside that scope, operations check trainable inputs (`requiresGrad`) and attach `OpNode` history to results.
- **Gradient accumulation** — `AccumulateGradient()` either sets or adds to `Grad` (supports fan-in from multiple paths). Uses `ArrayPool<T>.Shared` to minimize GC pressure.
- **Non-nullable domain (ADR-001)** — AutoDiff operates on non-null data only. The null boundary is enforced at `NivaraColumn<T>` → `ReverseGradTensor<T>` conversion. All AutoDiff ops are span-ified (`Span<T>` + `TensorPrimitives`), with no null-mask branches on hot paths.
- **IDisposable** — `GradTensor<T>`, `Parameter<T>`, `Module<T>`, `Optimizer<T>`, and `TrainingLoop<T>` all implement `IDisposable`.

---

## Tensor Classes

### GradTensor\<T\>

The base tensor class holding data and shape metadata:

```csharp
public class GradTensor<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
```

| Member | Description |
|--------|-------------|
| `Data` | The underlying `NivaraColumn<T>` |
| `Length` | Number of elements |
| `Shape` | Read-only copy of dimension sizes |
| `Rank` | Number of dimensions |
| `this[int index]` | Element accessor |
| `Reshape(params int[] dims)` | Sets shape metadata (product must equal Length) |
| `AsTensor()` | Zero-copy `Tensor<T>` view sharing the backing array (throws if the ADR-001 non-nullable domain is violated) |
| `ToColumn()` | Returns `NivaraColumn<T>` |
| `ToSeries()` | Returns `NivaraSeries<T>` |

Type support is enforced at compile time by the `IFloatingPointIeee754<T>`
generic constraint; the constructors retain a `TypeValidator.ValidateNumericType<T>()`
call (now a no-op) for compatibility.

### ReverseGradTensor\<T\>

Extends `GradTensor<T>` with gradient tracking and backward pass:

```csharp
public sealed class ReverseGradTensor<T> : GradTensor<T> where T : struct, IFloatingPointIeee754<T>
```

| Member | Description |
|--------|-------------|
| `Grad` | `NivaraColumn<T>?` — accumulated gradient (null before backward) |
| `RequiresGrad` | Whether this tensor tracks gradients |
| `GradFn` | `OpNode<T>?` — computation graph node (null for leaf tensors) |
| `IsLeaf` | `true` if `GradFn == null` |
| `Backward(gradient?)` | Initiates reverse-mode gradient computation |
| `Detach()` | Returns new tensor without gradient tracking |
| `ZeroGrad()` | Clears gradient |
| `ConvertTo<TTarget>()` | Converts to different numeric type |
| `ToFloat()` / `ToDouble()` / `ToHalf()` | Convenience conversion methods |

**Factory methods:**

| Method | Description |
|--------|-------------|
| `FromColumn(column, requiresGrad)` | Wraps a `NivaraColumn<T>` |
| `FromSeries(series, requiresGrad)` | Wraps a `NivaraSeries<T>` |
| `FromArray(array, requiresGrad)` | Creates from `T[]` |
| `FromMatrix(data, rows, cols, requiresGrad)` | Creates 2D tensor with shape [rows, cols] |

**Backward behavior:**
- Scalar tensors (length 1): `Backward()` with no argument initializes gradient to `[1.0]`
- Non-scalar tensors: `Backward(gradient)` requires an explicit gradient tensor of matching shape
- Throws `InvalidOperationException` if called on tensors without `requiresGrad`
- Wraps graph errors (circular dependency, missing output) in descriptive messages

---

## Computation Graph (OpNode / ComputationGraph)

### OpNode\<T\>

Represents a single operation in the computation graph:

```csharp
sealed class OpNode<T> where T : struct, IFloatingPointIeee754<T>
{
    string OperationName { get; }           // "Add", "MatMul", "Relu", etc.
    IReadOnlyList<object> Inputs { get; }   // parent tensors
    Action<NivaraColumn<T>, bool> BackwardFunction { get; }
    bool ShouldSaveForBackward { get; }
    Dictionary<string, object>? SavedValues { get; }
    void Apply(NivaraColumn<T> gradOutput);
}
```

The `BackwardFunction` closure captures references to input tensors and any saved forward values (e.g., sigmoid output for sigmoid gradient). It computes the local gradient contribution and calls `AccumulateGradient` on each input that requires `grad`.

### ComputationGraph

Internal graph traversal engine (not public API). Public graph inspection is
exposed through `GradientUtils`: `ZeroGrad`, `GetGraphInfo`, `PrintGraphSummary`,
`DescribeTensor`.

| Method | Description |
|--------|-------------|
| `AddNode(output, node)` | Attaches GradFn to the output tensor |
| `Backward(tensor, gradient)` | Topological sort + reverse-topological traversal, calling each node's `Apply(gradOutput)` |
| `TopologicalSort(root)` | DFS with cycle detection via visiting/visited sets |
| `ValidateGraph(root)` | Validates no circular dependencies |
| `ZeroGrad(tensor)` | Recursively clears gradients from reachable tensors |
| `GetGraphInfo(root)` | Returns a typed `GraphInfo` record `{ TotalNodes, IsLeaf, RequiresGrad, OperationCounts }` |

**Backward algorithm:**

1. `BuildNodeToOutputMap(tensor)` — maps OpNode → output tensor
2. `TopologicalSort(tensor)` — DFS producing a linear order
3. Iterate nodes **in reverse** (reverse topological order)
4. For each node, look up its output tensor, get `outputTensor.Grad`
5. Call `node.Apply(outputGrad)` which invokes the backward function
6. Each backward function computes local gradients and accumulates them via `AccumulateGradient`

---

## Supported Operations

### Element-wise

| Op | Forward | Backward Rule | Null Semantics |
|----|---------|---------------|----------------|
| `Add(a, b)` | `a + b` | `∂/∂a = grad`, `∂/∂b = grad` | n/a — non-nullable (ADR-001) |
| `Subtract(a, b)` | `a - b` | `∂/∂a = grad`, `∂/∂b = -grad` | n/a — non-nullable (ADR-001) |
| `Multiply(a, b)` | `a * b` | `∂/∂a = grad * b`, `∂/∂b = grad * a` | n/a — non-nullable (ADR-001) |
| `Divide(a, b)` | `a / b` | `∂/∂a = grad / b`, `∂/∂b = -(a/b²) * grad` | throws on zero division |
| `DivideScalar(a, scalar)` | `a / scalar` | `∂/∂a = grad / scalar` | Scalar divisor; no divisor tensor or node is created (issue #207) |
| `Clip(a, min, max)` | `clamp(a, min, max)` | 1 if in-range, 0 outside | n/a — non-nullable (ADR-001) |
| `Pow(a, exponent)` | `a^exponent` | `exponent * a^(exponent-1) * grad` | Scalar exponent |

### Matrix / Tensor Manipulation

| Op | Forward | Backward Rule | Requirements |
|----|---------|---------------|--------------|
| `MatMul(a, b)` | `a @ b` | `∂/∂a = grad @ bᵀ`, `∂/∂b = aᵀ @ grad` | Both tensors rank 2; `a.Cols == b.Rows` |
| `Transpose(a)` | `aᵀ` | `∂/∂a = gradᵀ` | Rank 2 |
| `TransposeAxes(a, axis1, axis2)` | Swap two axes | `∂/∂a = TransposeAxes(grad, axis1, axis2)` | Rank 2-3 |
| `Slice(a, start, length)` | `a[start:start+length]` | Gradient scattered back to original positions | start + length ≤ a.Length |
| `Concat(tensors, axis)` | Join along axis | Gradient split back to original tensors | All non-axis dims must match |
| `Gather(source, indices, axis)` | Select indices along axis | Scattered back via `SegmentSum` | indices valid for source shape |
| `MultiHeadAttention(query, key, value, numHeads, scale, mask?)` | Packed per-head scaled dot-product attention | Single fused VJP producing dQ, dK, dV | `query/key/value` rank 2 `[len, dim]`; `mask` additive `[qLen, kvLen]` |

MatMul is implemented in `ReverseGradOperations.MatMul` as a single `OpNode`
over the shared span kernels `TensorsHelper.MultiplyCore` (forward produces
`result[aRows × bCols]` from `TensorPrimitives.Dot` rows) and
`TensorsHelper.Transpose` (backward computes `grad @ bᵀ` and `aᵀ @ grad`).
Operand data comes from zero-copy column spans via `TryGetSpan`; results wrap
once with `NivaraColumn<T>.CreateFromOwnedArray`. Outside
`GradientUtils.Grad()`, the forward pass runs without creating the `OpNode`.

`MultiHeadAttention` (issue #86) is a fused attention kernel in `src/Nivara/AutoDiff/Operations/AttentionKernels.cs`: heads are gathered into contiguous column groups once, QK^T/softmax/PV run per head over `TensorPrimitives` row kernels (`SoftmaxRows`, `SoftmaxBackwardRows`), and results are packed back via `ScatterHead`. It executes as a single `OpNode` producing dQ/dK/dV in one backward pass, replacing the per-head `Slice`/`Transpose`/`MatMul`/`Softmax` decomposition. Inference outside `GradientUtils.Grad()` runs the forward pass without building any graph nodes. `TransformerBlock`, `MultiheadAttention<T>`, and the `BertModel` sample all route through this kernel.

### Reductions

| Op | Forward | Backward Rule | Notes |
|----|---------|---------------|-------|
| `Sum(a)` | `∑a` | `broadcast(grad, n)` — fills gradient value to all positions | Expects scalar output |
| `Mean(a)` | `(∑a)/n` | `broadcast(grad/n, n)` — fills gradient/n to all positions | Expects scalar output |
| `MeanPool(a, poolSize, embedDim)` | Mean over `poolSize` tokens per embedding dim | Gradient divided by `poolSize` and scattered back | Used by TextClassifierModel (in samples since 1.2.0) |

### Normalization

| Op | Forward | Backward Rule | Notes |
|----|---------|---------------|-------|
| `RMSNorm(a, eps)` | `a / √(mean(a²) + eps)` | Custom backward with saved input | Row-level normalization |
| `PerRowRMSNorm(a, rows, cols, eps)` | Per-row RMS normalization | Custom backward with saved input | Used by TransformerBlock |
| `PerRowLayerNorm(a, rows, cols, eps)` | Per-row standard LayerNorm (mean + variance) | Delegates to LayerNormKernel.Backward | Used by TransformerBlock with NormType.LayerNorm |

### Activations

| Op | Forward | Backward Rule | Vectorization |
|----|---------|---------------|---------------|
| `Relu(a)` | `max(a, 0)` | `grad * (1 if a > 0 else 0)` | `TensorPrimitives.Max` for forward; manual loop for grad |
| `LeakyRelu(a, slope)` | `max(a, 0) + slope * min(a, 0)` | `grad * (1 if a > 0 else slope)` | Default slope is 0.01 (not `default(T)` which is 0). Manual loop |
| `Sigmoid(a)` | `σ(a) = 1/(1+e⁻ᵃ)` | `σ(a) * (1-σ(a)) * grad` | Manual loop via `Math.Exp` |
| `Tanh(a)` | `tanh(a)` | `(1 - tanh²(a)) * grad` | Manual loop via `Math.Tanh` |
| `Gelu(a)` | `0.5·a·(1 + tanh(√(2/π)·(a + 0.044715·a³)))` | `0.5·(1+erf(a/√2)) + a·φ(a)` via the tanh-approximation CDF/PDF | Manual `Math.Tanh` loop (forward + backward) |
| `Negate(a)` | `-a` | `-grad` | `TensorPrimitives.Negate` |
| `Abs(a)` | `\|a\|` | `sign(a) * grad` | `TensorPrimitives.Abs` for forward; manual loop for grad |
| `Exp(a)` | `eᵃ` | `eᵃ * grad` | Manual loop via `Math.Exp` |
| `Log(a)` | `ln(a)` | `grad / a` | Manual loop; throws on non-positive |

### Probability

| Op | Forward | Backward Rule | Notes |
|----|---------|---------------|-------|
| `Softmax(a)` | `e^(a - max) / Σe^(a - max)` | Full Jacobian via `diag(s) - s·sᵀ` | Numerically stable subtract-max |
| `LogSoftmax(a)` | `log(softmax(a))` | `grad - Σgrad · softmax(a)` | Fused for CrossEntropy efficiency |
| `KlDivergence(mu, logVar)` | `-0.5 * Σ(1 + logVar - mu² - e^logVar)` | Analytic gradient via ELBO | Used by VAE |
| `SampleNormal(mu, logVar, seed?)` | `mu + ε * √(e^logVar)` (reparameterization trick) | Passes gradient through mu and std | ε is sampled noise |

### Regularization

| Op | Forward | Backward Rule | Notes |
|----|---------|---------------|-------|
| `Dropout(input, prob, isTraining)` | Zero out `prob` fraction, scale by `1/(1-prob)` | Gradient passes through kept positions only | Inverted dropout; identity in eval mode |

### Embedding

| Op | Forward | Backward Rule | Notes |
|----|---------|---------------|-------|
| `SparseEmbeddingBag(weight, input, paddingIndex)` | Sum of embedding rows for active features per batch | Scattered gradient to weight rows | Padding indices ignored |

### Vectorization Strategy

All AutoDiff operations are span-ified: they extract `ReadOnlySpan<T>` via `TryGetSpan()` and call `TensorPrimitives` kernels directly, then return `NivaraColumn<T>.Create(result)`. No `NivaraColumn.Data` access and no null-mask branches remain on hot paths (per ADR-001 non-nullable domain), so each operation is a single-path SIMD kernel.

Operations that lack `TensorPrimitives` support (e.g., Sigmoid, Tanh, Exp, Log) use manual span loops in the same single-path style. MatMul uses `NivaraColumn<T>.MatMul()` with `TensorPrimitives.Dot` + `Parallel.For`.

---

## Forward-Mode Automatic Differentiation

Nivara supports both reverse-mode (backpropagation) and forward-mode automatic differentiation. Forward-mode propagates a **tangent** (directional derivative) alongside the primal value during forward evaluation, computing Jacobian-Vector Products (JVPs) without storing a computation graph.

### ForwardGradTensor\<T\>

```csharp
public sealed class ForwardGradTensor<T> : GradTensor<T> where T : struct, IFloatingPointIeee754<T>
```

| Member | Description |
|--------|-------------|
| `Tangent` | `NivaraColumn<T>?` — directional derivative data (null if not tracking) |
| `RequiresTangent` | Whether this tensor carries a tangent |

**Factory methods:** `FromColumn`, `FromSeries`, `FromArray`, `FromMatrix` — mirror `ReverseGradTensor` factories.

**Operator overloads:** `+`, `-`, `*`, `/`, unary `-` — delegate to `ForwardGradOperations`.

### ForwardGradOperations

```csharp
public static class ForwardGradOperations
```

Mirrors `ReverseGradOperations` in structure. Each method computes the primal value and propagates the tangent via the JVP rule:

| Op | JVP Rule |
|----|----------|
| `Add(a, b)` | `t_out = t_a + t_b` |
| `Subtract(a, b)` | `t_out = t_a - t_b` |
| `Multiply(a, b)` | `t_out = t_a * b + a * t_b` |
| `Divide(a, b)` | `t_out = (t_a * b - a * t_b) / b²` |
| `DivideScalar(a, scalar)` | `t_out = t_a / scalar` |
| `MatMul(a, b)` | `t_out = t_a @ b + a @ t_b` |
| `Transpose(a)` | `t_out = t_aᵀ` |
| `Relu(a)` | `t_out = t_a * (a > 0 ? 1 : 0)` |
| `Sigmoid(a)` | `t_out = t_a * s * (1 - s)` where `s = sigmoid(a)` |
| `Tanh(a)` | `t_out = t_a * (1 - t²)` |
| `Clip(a, min, max)` | `t_out = t_a * (min ≤ a ≤ max ? 1 : 0)` |
| `Exp(a)` | `t_out = t_a * exp(a)` |
| `Log(a)` | `t_out = t_a / a` |
| `Softmax(a)` | Full Jacobian-vector product |
| `Sum(a)` | `t_out = Σt_a` |
| `Mean(a)` | `t_out = Σt_a / n` |
| `Dropout(...)` | Tangent passes through kept positions, zeroed at dropped positions |
| `MatMulTransposedB(a, b)` | `t_out = t_a @ bᵀ + a @ t_bᵀ` |
| `TransposeAxes(a, axis1, axis2)` | `t_out = t_a transposed along the same axes` |
| `Slice(a, start, length)` | `t_out = t_a[start .. start+length]` |
| `Concat(tensors, axis)` | `t_out = concat of tangents along axis (zero-filled where a tangent is absent)` |
| `Gather(source, indices, axis)` | `t_out = t_source[indices]` (indices are non-differentiable) |
| `SparseEmbeddingBag(weight, indices, paddingIndex)` | `t_out = bag-mean/agg of t_weight at the selected rows` (indices non-differentiable) |
| `GeluExact(a)` | `t_out = t_a * gelu_exact'(a)` |
| `Pow(a, exponent)` | `t_out = t_a * exponent * a^(exponent-1)` |
| `MeanPool(a, poolSize, embedDim)` | `t_out = pooled t_a / poolSize` (linear) |
| `RMSNorm(a)` | `t_out = J·t_a` reusing the reverse backward kernel (symmetric Jacobian) |
| `PerRowRMSNorm(a, rows, cols)` | `t_out = per-row J·t_a` reusing the reverse backward kernel |
| `AddBias(a, bias)` | `t_out = t_a + broadcast(t_bias)` |
| `BroadcastMultiply(input, scale)` | `t_out = t_input * scale + input * broadcast(t_scale)` |
| `BroadcastAdd(input, bias)` | `t_out = t_input + broadcast(t_bias)` |
| `MultiHeadAttention(q, k, v, numHeads, scale, mask)` | per-head `t_scores = scale·(t_Q @ Kᵀ + Q @ t_Kᵀ)`, `t_out = t_P @ V + P @ t_V` with an in-place softmax JVP (`SoftmaxBackwardRows`); `mask` is a non-differentiable constant |
| `BatchedMultiHeadAttention(...)` | same rule as `MultiHeadAttention`, applied per batch element |

All 40 operations in `ForwardGradOperations` also include `KlDivergence` and `SampleNormal` for VAE forward-mode workflows.

### When to use which

| Mode | Best for | Storage | Typical use |
|------|----------|---------|-------------|
| **Reverse-mode** | Scalar loss → many parameters | Computation graph (OpNodes) | Standard training loops |
| **Forward-mode** | Few outputs, directional derivatives | Tangent vector (no graph) | Sensitivity analysis, Jacobian columns |

Reverse-mode is the default and is used by all training infrastructure (`TrainingLoop`, `DataParallelTrainer`, optimizers). Forward-mode is available for specialized workflows where storing the full graph is impractical.

---

## Null Handling

Per [ADR-001](../adr/001-autodiff-nonnullable-domain.md), AutoDiff is a **non-nullable domain**. The null boundary is enforced at domain entry points (`NivaraColumn<T>` → `ReverseGradTensor<T>` conversion): all AutoDiff ops assume non-null data, and `Debug.Assert` guards on the `ReverseGradTensor<T>` constructors and `ComputationGraph.AddNode()` enforce the boundary.

Null-mask branches have been removed from all hot paths — `AccumulateGradient`, `BroadcastGradient`, KL/sample ops, `SGD`, `Adam`, and `AdamW` — so gradient accumulation and optimizer updates run as single-path `TensorPrimitives` kernels with no null-merge logic. The `MergeNullMasks` helper and per-position null semantics documented in earlier releases no longer exist.

To use nullable DataFrame columns with AutoDiff, resolve nulls at the boundary first (`FillNull` / `DropNulls` / `WithoutNulls`).

---

## Module System (Nn)

All leaf modules that own learnable weights expose them through a **single uniform
contract**: `Weight` / `Bias` of type `Parameter<T>?`, where `null` means the
parameter was omitted (`bias: false` / `affine: false`). There are no
`WeightParam` / `BiasParam` accessors and no tensor-typed `Weight` members;
consumers reach the underlying tensor via `Weight!.Tensor` / `Bias!.Tensor`.
This matches PyTorch's `module.weight` / `module.bias` mental model.

### Parameter\<T\>

```csharp
public sealed class Parameter<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
```

Wraps a `ReverseGradTensor<T>` with a name. Constructors accept array, size, or an existing tensor. `requiresGrad` defaults to `true`. Disposing a parameter disposes its current tensor; module-registered parameters are disposed by their owning module.

### Module\<T\>

```csharp
public abstract class Module<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
```

| Member | Description |
|--------|-------------|
| `IsTraining` | Current train/eval state |
| `Forward(input)` | Abstract — define model logic |
| Multi-input forward | Opt-in via `IMultipleInputModule<T>` — implemented only by `MultiheadAttention<T>` (`Forward(input, paddingMask)`) and `VAE<T>` (`Forward(x, condition)`) |
| `Train()` | Sets training mode (recursive) |
| `Eval()` | Sets evaluation mode (recursive) |
| `RegisterModules(...)` | Register child modules for parameter discovery |
| `RegisterParameters(...)` | Register standalone parameters |
| `Parameters()` | Returns flat `Dictionary<string, ReverseGradTensor<T>>` |
| `GetParameters()` | Returns `Dictionary<string, Parameter<T>>` with metadata |
| `StateDict()` | Returns a snapshot `Dictionary<string, ReverseGradTensor<T>>` for save/transfer/fine-tune workflows |
| `LoadStateDict(state, strict: false)` | Loads matching parameter tensors with shape validation; missing keys are allowed unless `strict` is true |
| `NamedModules()` | Returns registered child modules |

### Linear\<T\>

```csharp
public sealed class Linear<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
```

`y = x @ Wᵀ + b` with shape `[batch, inFeatures] → [batch, outFeatures]`.

- Registers weight (shape `[outFeatures, inFeatures]`) and optional bias (shape `[1, outFeatures]`) as parameters
- Initializes weights with Kaiming-Uniform: `U(-√(6/fanIn), √(6/fanIn))`
- Forward transposes weight, applies MatMul, then broadcasts bias via `ones @ bias`
- Inference (outside `GradientUtils.Grad()`) passes the raw weight straight to the kernel's transposed-B matmul (`MatMulTransposedB`) — zero transposes
- Training (inside `GradientUtils.Grad()`) records a grad-tracking `MatMulTransposedB` op: `Linear` passes its raw `[outFeatures, inFeatures]` weight to the transposed-B matmul, so weights are never transposed per forward or per backward (the earlier version-stamped transpose cache, issue #87, was removed)

### Sequential\<T\>

```csharp
public sealed class Sequential<T> : Module<T>
```

Pipes forward pass through an ordered list of modules. Supports `Append()` for dynamic construction.

### Activation\<T\>

Wraps a single activation function as a module: `Relu`, `Sigmoid`, `Tanh`, `LeakyRelu`.

### Dropout\<T\>

Inverted dropout — scales by `1/(1-p)` during training, identity during eval. Training-mode dropout is differentiable: the sampled keep mask is reused during backward so gradients before dropout receive `gradOutput * keepMask * scale`.

### Embedding\<T\>

```csharp
public sealed class Embedding<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
// new Embedding<T>(numEmbeddings, embeddingDim)
```

Lookup embedding: token IDs → dense vectors. Weight matrix shape `[numEmbeddings, embeddingDim]`, initialized with `Normal(0, 0.02)`.

- `Forward(tokenIds)` — single token or batched input; uses a zero-copy `Gather` (replaces the old one-hot + MatMul path)
- `Weight` — `Parameter<T>?` accessor; tensor via `Weight!.Tensor`

### SparseEmbedding\<T\>

```csharp
public sealed class SparseEmbedding<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
// new SparseEmbedding<T>(numEmbeddings, embeddingDim, paddingIndex: -1)
```

Sparse embedding bag for fixed-width batches of active feature indices. Input shape `[batchSize, maxActiveFeatures]`, output `[batchSize, embeddingDim]`. Entries matching `PaddingIndex` are ignored. Useful for feature hashing / sparse feature sets.

### TextClassifierModel\<T\> *(samples)*

> Moved out of core in 1.2.0 — lives in `samples/Nivara.Samples/TextClassifierModel.cs` (namespace `Nivara.AutoDiff.Nn`). Reference the `Nivara.Samples` project to use it.

```csharp
public sealed class TextClassifierModel<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
// new TextClassifierModel<T>(vocabSize, embeddingDim, hiddenDim, numClasses, maxSeqLen)
```

Pre-built text classification pipeline: `Embedding → MeanPool → Linear(hidden) → ReLU → Linear(numClasses)`.

- `Forward(input)` — logits for each class (mean-pooled over sequence)
- `Predict(int[] tokenIds)` — returns `int[]` of predicted class indices per batch

### TokenClassifierModel\<T\> *(samples)*

> Moved out of core in 1.2.0 — lives in `samples/Nivara.Samples/TokenClassifierModel.cs` (namespace `Nivara.AutoDiff.Nn`). Reference the `Nivara.Samples` project to use it.

```csharp
public sealed class TokenClassifierModel<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
// new TokenClassifierModel<T>(vocabSize, embeddingDim, hiddenDim, numClasses, maxSeqLen)
```

Pre-built per-token classification pipeline: `Embedding → Linear(hidden) → ReLU → Linear(numClasses)` — no pooling, output per token.

- `Forward(input)` — logits per token `[batchSize * seqLen, numClasses]`
- `Predict(int[] tokenIds)` — returns `int[]` of predicted class per token

### VAE\<T\>

```csharp
public sealed class VAE<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
// new VAE<T>(inputDim, latentDim, hiddenDim, decoderHiddenDim?, activation?, beta: 1.0f)
```

Variational Autoencoder with encoder → reparameterization trick → decoder.

| Method | Description |
|--------|-------------|
| `Encode(x)` | Returns `(Mu, LogVar)` latent distribution parameters |
| `Reparameterize(mu, logVar, seed?)` | Samples `z = mu + ε * std` (identity in eval mode) |
| `Decode(z)` | Reconstructs from latent vector |
| `Forward(x)` | Encode → reparameterize → decode |
| `ElboLoss(recon, original, mu, logVar, lossType)` | Reconstruction + KL divergence loss |

`ElboLossType`: `KldBeta` (weighted KL via learned `Beta` parameter) or `KldAnnealing` (unweighted KL for annealing schedules).

### TransformerBlock\<T\>

```csharp
public sealed class TransformerBlock<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
// new TransformerBlock<T>(embedDim, numHeads, hiddenDim, dropout: 0.0, residualDropout: 0.0, normType: NormType.RMSNorm)
```

Pre-norm transformer block with configurable normalization:

```
NormType { RMSNorm, LayerNorm }
PerRowRMSNorm(x) → TensorPrimitives-backed per-row normalization (no mean centering)
PerRowLayerNorm(x) → LayerNormKernel.Forward with affine=false (mean + variance normalization)
```

- Multi-head self-attention (Q/K/V/O projections, scaled dot-product, output projection)
- Configurable normalization: `RMSNorm` (default, fused per-row `TensorPrimitives.Dot`) or `LayerNorm` (delegates to `LayerNormKernel<T>` with affine=false)
- GELU FFN (fc1 → GELU → fc2)
- Residual connections with optional attention and residual dropout
- `embedDim` must be divisible by `numHeads`

### Conv1d\<T\>

```csharp
public sealed class Conv1d<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
// new Conv1d<T>(inChannels, outChannels, kernelSize, stride: 1, padding: 0, bias: true)
```

1D convolution with tiled im2col-based kernel and full autograd support. Weight layout is PyTorch-compatible `[outChannels, inChannels, kernelSize]`.

```
Forward:   Im2Col1DTile → TensorPrimitives.Dot per output channel (1×1 fast path skips im2col)
InputGrad: Conv1dInputGradKernel (scatter-add of weight × gradOut patches)
WeightGrad: Im2Col1DTile → TensorPrimitives.MultiplyAdd per output channel
BiasGrad:  TensorPrimitives.Sum over batch and length per output channel
```

- Input shape: `[N, C, L]`, output: `[N, outChannels, oL]` where `oL = (L + 2*padding - kernelSize) / stride + 1`
- Kaiming-Uniform initialization: `U(-√(6/fanIn), √(6/fanIn))`
- All kernel methods accept `Span<T>`/`ReadOnlySpan<T>` for composability

### Conv2d\<T\>

```csharp
public sealed class Conv2d<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
// new Conv2d<T>(inChannels, outChannels, kernelSize, stride: 1, padding: 0, groups: 1, bias: true)
```

2D convolution with tiled im2col → `TensorPrimitives.Dot` per output channel.

```
Forward:   Im2ColTile → Dot per output channel (groups=1: direct, groups>1: gather/scatter)
InputGrad: InputGrad1x1 | InputGrad3x3 | InputGradGeneric
WeightGrad: Im2ColTile → MultiplyAdd per output channel
BiasGrad:  TensorPrimitives.Sum per channel
```

Key optimizations:
- **PatchLocation lookup table**: precomputes `(Batch, OH, OW)` per-tile, eliminates 4 integer divisions per position
- **ConvForward1x1**: bypasses im2col entirely for 1×1 kernels (stride=1, padding=0)
- **InputGrad specializations**: `InputGrad1x1` (direct MultiplyAdd), `InputGrad3x3` (bounds-checked 9-tap scatter), `InputGradGeneric` (nested loops)
- **Zero-copy via TryGetSpan**: eliminates tensor copy when storage is contiguous
- **Grouped convolution**: `groups` parameter splits input/output channels into independent groups. For `groups=1` (common path), zero overhead

### ConvTranspose2d\<T\>

```csharp
public sealed class ConvTranspose2d<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
// new ConvTranspose2d<T>(inChannels, outChannels, kernelSize, stride: 1, padding: 0, bias: true)
```

2D transposed convolution using direct scatter kernel (not im2col-based).

```
Forward:     Col2ImForward (scatter) + bias
InputGrad:   ConvTransposeInputGradKernel (scatter with stride check)
WeightGrad:  ConvTransposeWeightGradKernel (reduction over ih, iw)
```

- No grouped convolution support (Conv2d has it)
- Used by ConvVAE decoder for stride-upsampling

### BatchNorm1d\<T\> / BatchNorm2d\<T\>

```csharp
public sealed class BatchNorm1d<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
// new BatchNorm1d<T>(numFeatures, eps: 1e-5, momentum: 0.1, affine: true)

public sealed class BatchNorm2d<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
// new BatchNorm2d<T>(numFeatures, eps: 1e-5, momentum: 0.1, affine: true)
```

Fused span-based kernel with `TensorPrimitives` — single `OpNode` per call.

```
BatchNormKernel<T>
├── Forward(input, n, c, hw, gamma, beta, eps, affine) → (Output, XHat, InvStd, Mean)
├── BackwardInput(gradOut, xHat, gamma, invStd, n, c, hw) → gradInput
├── BackwardWeight(gradOut, xHat, n, c, hw) → gradGamma
└── BackwardBias(gradOut, n, c, hw) → gradBeta
```

`BatchNorm1d<T>` accepts both 2D `[N, C]` and 3D `[B, C, L]` input. The 3D path normalizes each of the L positions independently (per-channel statistics across the batch and length dimensions), enabling direct use in Conv1d pipelines where intermediate activations have shape `[B, C, L]`.

- Train mode: computes batch statistics, updates running stats via direct span arithmetic
- Eval mode: uses cached running mean/var
- StateDict/LoadStateDict includes `running_mean`, `running_var`, `num_batches_tracked`

### LayerNorm\<T\>

```csharp
public sealed class LayerNorm<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
// new LayerNorm<T>(normalizedShape, eps: 1e-5, affine: true)
```

Span-based kernel with `TensorPrimitives`. Normalizes over the last dimension per instance (no running stats, unlike BatchNorm). Uses `TensorPrimitives.Dot` for SIMD-accelerated sum-of-squares computation.

```
LayerNormKernel<T>
├── Forward(input, rows, normalizedShape, gamma, beta, eps, affine) → (Output, Mean, InvStd, XHat)
│   └── Uses TensorPrimitives.Dot for sum-of-squares (SIMD-accelerated)
├── BackwardInput(gradOut, xHat, gamma, invStd, rows, normalizedShape, affine) → gradInput
├── BackwardWeight(gradOut, xHat, rows, normalizedShape) → gradGamma
└── BackwardBias(gradOut, rows, normalizedShape) → gradBeta
```

### DepthwiseSeparableConv2d\<T\>

```csharp
public sealed class DepthwiseSeparableConv2d<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
// new DepthwiseSeparableConv2d<T>(inChannels, outChannels, kernelSize, stride: 1, padding: 0, bias: true)
```

Efficient depthwise separable convolution (MobileNet-style): depthwise conv (`groups=inChannels`) + pointwise 1×1 conv. Reuses existing `Conv2d` grouped kernel and `ConvForward1x1`.

```
DepthwiseSeparableConv2d<T> : Module<T>
└── Forward(input) → Conv2d(groups=inChannels) → ReLU → Conv2d(1×1)
```

### ConvVAE\<T\>

```csharp
public sealed class ConvVAE<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
// new ConvVAE<T>(inputChannels, encoderChannels, latentChannels, spatialSize, kernelSize, stride, padding)
```

Fully convolutional VAE with 1×1 Conv2d heads for spatial latent representations.

```
ConvVAE<T> : Module<T>
├── Forward(x) → recon
├── Encode(x) → (Mu, LogVar)       [both spatial, e.g. B×C'×H'×W']
├── Reparameterize(mu, logVar) → z  [spatial reparameterization trick]
├── Decode(z) → recon               [ConvTranspose stack]
└── ElboLoss(recon, x, mu, logVar) → scalar  [MSE + KL divergence]
```

- Configurable encoder channel list, latent channels, kernel/stride/padding
- 1×1 Conv heads preserve spatial structure in latent space

### Conditional VAE

```csharp
// VAE<T> with conditionDim parameter
// new VAE<T>(inputDim, latentDim, hiddenDim, decoderHiddenDim?, conditionDim: 0, ...)
```

Extended `VAE<T>` with optional conditioning. Encoder/decoder accept condition tensor via `Concat`.

```
VAE<T> : Module<T>
├── Forward(x) → recon
├── Forward(x, condition) → recon
├── Encode(x) → (Mu, LogVar)
├── Encode(x, condition) → (Mu, LogVar)
├── Reparameterize(mu, logVar) → z
├── Decode(z) → recon
├── Decode(z, condition) → recon
└── ElboLoss(recon, x, mu, logVar, lossType) → scalar
```

### Sampler\<T\>

```csharp
public sealed class Sampler<T> where T : struct, IFloatingPointIeee754<T>
// new Sampler<T>(seed?)
```

Temperature-scaled, top-k filtered categorical sampling from logits.

| Method | Description |
|--------|-------------|
| `Sample(logits, temperature, topK)` | Returns sampled token index |

Temperature < 1 sharpens distribution, > 1 softens. Top-k filters to k most probable tokens before sampling.

### TextTokenizer

```csharp
public sealed class TextTokenizer
```

Vocabulary builder with special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`).

| Method | Description |
|--------|-------------|
| `FromDocuments(docs, maxVocabSize, minFreq)` | Builds vocabulary from corpus |
| `Encode(text)` | Tokenizes and maps to `int[]` of token IDs |
| `Decode(ids)` | Converts token IDs back to string |
| `Save(jsonPath)` / `Load(jsonPath)` | JSON serialization |

### Initializers

| Class | Formula | Use |
|-------|---------|-----|
| `KaimingUniformInitializer<T>` | `U(-√(6/fanIn), √(6/fanIn))` | ReLU layers |
| `KaimingNormalInitializer<T>` | `N(0, √(2/fanIn))` | ReLU layers |
| `XavierUniformInitializer<T>` | `U(-√(6/(fanIn+fanOut)), √(6/(fanIn+fanOut)))` | Tanh/Sigmoid layers |
| `XavierNormalInitializer<T>` | `N(0, √(2/(fanIn+fanOut)))` | Tanh/Sigmoid layers |
| `UniformInitializer<T>` | `U(-bound, bound)` | Generic |
| `NormalInitializer<T>` | `N(mean, std)` | Generic |
| `PyTorchDefaultInitializer<T>` | `U(-1/√(fanIn), 1/√(fanIn))` | PyTorch-compatible default init |

Each implements `IInitializer<T>` with `Initialize(Parameter<T>)`. Pass an instance to a
module constructor (`weightInitializer:`, `biasInitializer:`), or call
`initializer.Initialize(parameter)` directly after construction.

### Example

```csharp
class MLP : Module<float>
{
    Linear<float> L1, L2, L3;

    public MLP()
    {
        L1 = new Linear<float>(784, 256);
        L2 = new Linear<float>(256, 64);
        L3 = new Linear<float>(64, 10);
        RegisterModules(L1, L2, L3);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> x)
    {
        var h = ReverseGradOperations.Relu(L1.Forward(x));
        h = ReverseGradOperations.Relu(L2.Forward(h));
        return L3.Forward(h);
    }
}
```

---

## Loss Functions

All loss functions live in `Nivara.AutoDiff.Nn.Functional`. Every loss inherits the abstract `Loss<T>` base, which stores a `Reduction` (default `Reduction.Mean` for PyTorch parity) set via the constructor. `Forward(predictions, targets)` applies the stored reduction; the three-argument `Forward(predictions, targets, Reduction)` overrides it per call. `Reduce(elementwiseLoss, reduction, divisor)` centralizes reduction: `None` returns the elementwise loss, `Sum` reduces with `ReverseGradOperations.Sum`, and `Mean` divides the sum by the divisor (element count unless the loss supplies a batch divisor).

| Loss | Forward Formula | Notes |
|------|----------------|-------|
| `MSELoss<T>` | `Σ(pred - target)²` | Mean Squared Error. Default `Reduction.Mean` divides by element count; `Sum`/`None` available. |
| `L1Loss<T>` | `Σ\|pred - target\|` | Mean Absolute Error. Default `Reduction.Mean`. |
| `BCELoss<T>` | `-Σ(y·log(p) + (1-y)·log(1-p))` | Inputs clamped to `[eps, 1-eps]` for numerical stability; `eps` is a ctor argument. |
| `BCEWithLogitsLoss<T>` | Fused sigmoid + BCE | Numerically stable — no clamp needed. Backward uses fused `sigmoid(x) - z` via custom OpNode (fixes subgradient error at x=0). |
| `CrossEntropyLoss<T>` | LogSoftmax + NLL ÷ batchSize | Expects logits + soft targets, or logits + `int[]` labels (one-hot built internally). `Reduction.None` returns per-sample NLL (`[N]`, summing class-weighted NLL within each row) matching PyTorch `reduction='none'`. |
| `Activation.Softmax<T>` | dim-aware softmax | `Activation.Softmax(input, dim = -1)` wrapping `ReverseGradOperations.Softmax`. |
| `Activation.LogSoftmax<T>` | dim-aware log-softmax | `Activation.LogSoftmax(input, dim = -1)` wrapping `ReverseGradOperations.LogSoftmax`. |

---

## Optimizers

### Optimizer\<T\> (abstract base)

```csharp
public abstract class Optimizer<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
```

**Parameter groups** — optimizers can manage multiple groups with different learning rates and weight decays:

```csharp
optimizer.AddParameterGroup(parameter);                      // uses optimizer.LearningRate
optimizer.AddParameterGroup(model.GetParameters().Values);   // uses optimizer.LearningRate
optimizer.AddParameterGroup(parameter, learningRate, weightDecay);
optimizer.AddParameterGroup(model.GetParameters().Values, learningRate, weightDecay);
```

| Member | Description |
|--------|-------------|
| `LearningRate` | Default learning rate used by parameter groups when no group override is supplied. Settable: assigning forwards to every group created without an explicit override; groups created with an explicit override (or later managed via `SetGroupLearningRate`) are left untouched |
| `SetGroupLearningRate(index, lr)` | Mutates a single group's learning rate and marks it explicitly managed |
| `SetGroupWeightDecay(index, wd)` | Mutates a single group's weight decay |
| `Step()` | Abstract — applies updates to all parameters |
| `ZeroGrad()` | Zeros gradients on all managed parameters |
| `AddParameterGroup(...)` | Registers owning `Parameter<T>` objects; use `model.GetParameters().Values` for modules |
| `Dispose()` | Releases rented state buffers |

`ParameterGroup` exposes its `LearningRate`/`WeightDecay` read-only to consumers
(public get, internal set) — mutate groups only through
`SetGroupLearningRate`/`SetGroupWeightDecay` so the optimizer owns its state.

**In-place steps** — `Step()` writes the update into each parameter's existing
backing array and bumps its version (`Touch()`); the parameter tensor is never
replaced. `Step()` leaves each parameter's `Grad` slot intact — accumulation
happens during `Backward()`, which adds into the existing slot. Consequently a
`Step()` without a subsequent `ZeroGrad()` accumulates stale gradients across
steps (PyTorch semantics). The built-in `TrainingLoop<T>` and
`DataParallelTrainer<T>` call `ZeroGrad()` once per iteration; manual training
code must do the same.

### SGD\<T\>

```csharp
public sealed class SGD<T> : Optimizer<T>
// new SGD<T>(learningRate, momentum: 0.0)
```

- Optional momentum (`[0, 1)`)
- Optional weight decay per parameter group
- No-null fast path uses `TensorPrimitives.Multiply`/`Subtract/Add`
- Static `SgdUpdate(tensor, lr, wd)` helper for single-parameter updates

### Adam\<T\>

```csharp
public sealed class Adam<T> : Optimizer<T>
// new Adam<T>()                         // learningRate = 0.001
// new Adam<T>(learningRate)
// new Adam<T>(beta1: 0.9, beta2: 0.999, eps: 1e-8)
```

- Default learning rate is `0.001` unless overridden by constructor or parameter group
- Bias-corrected first/second moment estimates
- State buffers rented from `ArrayPool<T>.Shared`
- Null-skip: null positions zero momentum buffers (no update)
- Decoupled weight decay via per-group `weightDecay`
- Static `AdamUpdate(tensor, lr, expAvg, expAvgSq, step, ...)` functional helper for single-tensor updates

### AdamW\<T\>

```csharp
public sealed class AdamW<T> : Optimizer<T>
// new AdamW<T>()                         // learningRate = 0.001
// new AdamW<T>(learningRate)
// new AdamW<T>(beta1: 0.9, beta2: 0.999, eps: 1e-8)
```

- Default learning rate is `0.001` unless overridden by constructor or parameter group
- Identical to Adam except weight decay is applied directly to weights (not through gradients) — Loshchilov & Hutter 2019 formulation
- Same null-skip semantics and `ArrayPool` buffer management
- Static `AdamWUpdate(tensor, lr, expAvg, expAvgSq, step, ...)` functional helper for single-tensor updates

### Functional single-tensor updates

```csharp
SGD<T>.SgdUpdate(tensor, lr, wd)                                    // stateless
Adam<T>.AdamUpdate(tensor, lr, expAvg, expAvgSq, step, ...)         // stateful
AdamW<T>.AdamWUpdate(tensor, lr, expAvg, expAvgSq, step, ...)       // stateful
```

Each is available for single-tensor updates outside the module system. The
stateless `SgdUpdate` needs no state; the stateful `AdamUpdate`/`AdamWUpdate`
take caller-owned `expAvg`/`expAvgSq` buffers plus a 1-based `step`, mutate the
buffers in place so consecutive calls accumulate momentum, and return a new
`requiresGrad=false` tensor. All three throw when `Grad` is null or the learning
rate is non-positive.

---

## Training

### TensorDataset\<T\>

```csharp
public sealed class TensorDataset<T> where T : struct, IFloatingPointIeee754<T>
```

Wraps a `NivaraFrame` with named feature and label columns. `GetBatch(indices)` returns a `Batch<T>` with shaped tensors (flat data reshaped to `[batchSize, numCols]`). Uses `ArrayPool<T>.Shared` for batch construction.

### DataLoader\<T\>

```csharp
public sealed class DataLoader<T> : IEnumerable<Batch<T>>
// new DataLoader<T>(dataset, batchSize, shuffle: true, seed: null)
```

Fisher-Yates shuffle (optionally seeded), yields batches of the requested size (final batch may be smaller).

### Batch\<T\>

```csharp
public sealed class Batch<T>
// { Features: ReverseGradTensor<T>, Labels: ReverseGradTensor<T>, Size: int }
```

### TrainingLoop\<T\>

```csharp
public class TrainingLoop<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
```

Standard epoch-per-batch training loop:

```csharp
var optimizer = new SGD<float>(learningRate: 0.01f);
optimizer.AddParameterGroup(model.GetParameters().Values);

var loop = new TrainingLoop<float>(
    model, loader,
    (pred, target) => new MSELoss<float>().Forward(pred, target),
    optimizer,
    epochs: 20);

var result = loop.Run();
result.PrintSummary();
```

| Feature | Description |
|---------|-------------|
| Virtual callbacks | `OnEpochStart(epoch)`, `OnBatchEnd(epoch, batch, loss)`, `OnEpochEnd(epoch, result)` |
| Checkpointing | `SaveCheckpoint(path, epoch, result)` — writes JSON checkpoint |
| Results | `TrainingResult<T>` with `PrintSummary()`, epoch-level loss/timing/batches |

### DataParallelTrainer\<T\>

```csharp
public class DataParallelTrainer<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
```

Multi-core training via `Parallel.For` over data chunks:

```
Split rows into chunks (batchSize per chunk)
  ↓
Parallel.ForEach(chunks):
  ├── GetBatch → Forward → loss → Backward
  └── CloneGradients() → snapshot of per-chunk gradients
  ↓
SumAndApplyGradients(allGradients)     ← TensorPrimitives.Add across chunks
  ↓
Optimizer.Step() + ZeroGrad()
```

| Feature | Description |
|---------|-------------|
| Chunk sizing | Uses `ParallelExecutionHelper` for optimal chunk count |
| Gradient merge | `SumAndApplyGradients` sums per-chunk gradients via `TensorPrimitives.Add` |
| Results | `DataParallelTrainingResult<T>` with `PrintSummary()` |
| Virtual callbacks | `OnEpochStart(epoch)`, `OnEpochEnd(epoch, result)` |

```csharp
var optimizer = new Adam<float>(learningRate: 0.001f);
optimizer.AddParameterGroup(model.GetParameters().Values);

var trainer = new DataParallelTrainer<float>(
    model, loader,
    (pred, target) => new MSELoss<float>().Forward(pred, target),
    optimizer,
    epochs: 10);

var result = trainer.Run();
result.PrintSummary();
// Epoch   1 | Loss:   0.542100 | Workers:  8 | Chunks:   32 | Grad Norm:   1.234500 | Time: 0.42s
```

---

## Serialization

### ModelSerializer

Static class for saving/loading model parameter state dicts:

```csharp
// In-memory state dictionary
var state = model.StateDict();
model.LoadStateDict(state);

// Partial load for fine-tuning/model surgery
state.Remove("Module_1.Weight");
state.Remove("Module_1.Bias");
model.LoadStateDict(state);

// State dictionary JSON
var json = ModelSerializer.StateDictToJson(state);
var restored = ModelSerializer.JsonToStateDict<float>(json, requiresGrad: true);

// Save
ModelSerializer.Save(model, "model.json");

// Load (mutates model parameters in-place)
ModelSerializer.Load(model, "model.json");

// Checkpoint
ModelSerializer.SaveCheckpoint(model, epochResult, "checkpoint.json");
var checkpoint = ModelSerializer.LoadCheckpoint<float>("checkpoint.json");
```

`StateDict()` returns copied tensors, not live references to the model's
parameters. You can remove keys, serialize the dictionary, or load it into a
compatible model without accidentally mutating the source model.

**Format:** JSON with format marker `"nivara-ss-v2"` / `"nivara-ckpt-v2"`, version field, type name, and parameter entries. Each parameter entry stores:
- `Shape` — `int[]` dimension sizes
- `Values` — base64-encoded binary (via `MemoryMarshal.AsBytes`), length-validated on load

The AutoDiff domain is non-nullable (ADR-001), so no null mask is persisted.
Files written with the v1 format (which stored `HasNulls` / `NullMask`) are
rejected loudly on load with an "unsupported format" error.

**Validation on load:** shape rank, exact shape, element count, and parameter
name matching with descriptive error messages. `LoadStateDict(..., strict:
true)` additionally requires every model parameter to be present.

### Checkpoint\<T\>

```csharp
public sealed class Checkpoint<T> where T : struct, IFloatingPointIeee754<T>
{
    public int Epoch { get; init; }
    public double Loss { get; init; }
    public IReadOnlyDictionary<string, ParameterData<T>> Parameters { get; init; }
}
```

### Example

```csharp
// Train
var loop = new TrainingLoop<float>(model, loader, lossFn, optimizer, epochs: 10);
var result = loop.Run();
result.PrintSummary();

// Save model
ModelSerializer.Save(model, "trained_model.json");

// Load into fresh model for inference
var loaded = new MLP(784, 256, 10);
ModelSerializer.Load(loaded, "trained_model.json");
loaded.Eval();
var prediction = loaded.Forward(testInput);

// Or keep it in memory for transfer learning / partial loading
var state = model.StateDict();
state.Remove("Module_2.Weight");
state.Remove("Module_2.Bias");
loaded.LoadStateDict(state);
```

---

## Utility Functions (GradientUtils)

### Gradient Management

| Method | Description |
|--------|-------------|
| `ZeroGrad(tensor)` | Clears gradients recursively via `ComputationGraph.ZeroGrad` |
| `ZeroGrad(tensors)` | Batch zero-grad |
| `Detach(tensor)` | Removes from computation graph |
| `Detach(tensors)` | Batch detach |

### Gradient Clipping

| Method | Description |
|--------|-------------|
| `ClipGradValue(tensor, maxValue)` | Clips each gradient element to `[-maxValue, maxValue]` (uses `TensorPrimitives.Clamp`) |
| `ClipGradNorm(tensor, maxNorm)` | Scales gradient if L2 norm exceeds `maxNorm` (uses `TensorPrimitives.SumOfSquares`) |
| `ClipGradNorm(tensors, maxNorm)` | Global norm clipping across multiple tensors |

All clipping preserves null positions (nulls are skipped).

### Constant Tensor Creators

| Method | Description |
|--------|-------------|
| `Constant(data)` | Creates non-gradient tensor from array or column |
| `Zeros(length)` | Filled with `T.Zero` |
| `Ones(length)` | Filled with `T.One` |
| `Full(length, value)` | Filled with specific value |

### Diagnostics

AutoDiff hot paths participate in the shared `DiagnosticsTracker` when
diagnostics are enabled. Recorded operation names include
`AutoDiffBackward`, `AutoDiffMatMul`, `AutoDiffTranspose`, `AutoDiffRelu`,
`AutoDiffSigmoid`, `AutoDiffTanh`, and `AutoDiffSgdUpdate`; each record
captures elapsed time, managed allocation deltas, element type, input length,
null participation, and operation-specific notes such as shape metadata.

| Method | Description |
|--------|-------------|
| `GetGraphInfo(tensor)` | Returns a typed `GraphInfo` record (`TotalNodes`, `IsLeaf`, `RequiresGrad`, `OperationCounts`) |
| `PrintGraphSummary(tensor)` | Human-readable graph summary string |
| `DescribeTensor(tensor)` | Detailed tensor debug info (length, grad norm, operation, etc.) |
| `HasGradient(tensor)` | Whether `Grad != null` |
| `GetGradientNorm(tensor)` | L2 norm of gradient (uses `TensorPrimitives.SumOfSquares`) |
| `GetGlobalGradientNorm(tensors)` | Combined L2 norm across tensors |
| `CanBackward(tensor)` | Whether `Backward()` can be called with no seed (scalar + requiresGrad) |
| `CanBackward(tensor, gradient)` | Whether `Backward(gradient)` is valid (requiresGrad, matching length and shape) |

---

## Type System

### Supported Types

**float**, **double**, **Half**, and **BFloat16** are supported for autograd. Enforcement happens at two levels:

1. **Generic constraint**: `where T : struct, IFloatingPointIeee754<T>` — a precise bound matching the supported numeric types exactly
2. **Runtime check**: `TypeValidator.IsSupportedType(typeof(T))` returns true for `float`, `double`, and `Half`; the legacy `ValidateNumericType<T>()` gatekeeper was removed since the constraint now enforces the boundary at compile time

### Type Conversion (TypeConverter)

| Method | Description |
|--------|-------------|
| `Convert<TSource, TTarget>(source, requiresGrad?)` | Converts between supported types |
| `ToFloat(source, requiresGrad?)` | Converts to float |
| `ToDouble(source, requiresGrad?)` | Converts to double |
| `ToHalf(source, requiresGrad?)` | Converts to Half |
| `TryConvert<TSource, TTarget>(...)` | Returns null on failure |
| `CanConvert<TSource, TTarget>()` | Checks if both types are supported |

Conversion preserves `requiresGrad` (unless overridden) and shape metadata.

---

## Nivara Frame Integration

The `NivaraAutoGradExtensions` class (in `Nivara.AutoDiff.Extensions`) provides conversion between Nivara types and autograd tensors:

### Column/Series → Tensor

```csharp
column.ToReverseGradTensor(requiresGrad: false)    // NivaraColumn<T> → ReverseGradTensor<T>
series.ToReverseGradTensor(requiresGrad: false)    // NivaraSeries<T> → ReverseGradTensor<T>
```

### Frame → Tensor batch

```csharp
// Specific columns by name
var tensors = frame.ToReverseGradTensors<float>(
    new[] { "Age", "Income" }, requiresGrad: true);

// Auto-detect numeric columns
var tensors = frame.ToReverseGradTensorsAuto(requiresGrad: false);
// Returns Dictionary<string, object> — float, double, Half, and BFloat16 columns are converted
```

### Tensor batch → Frame

```csharp
var dataFrame = tensors.ToFrame();           // Values as NivaraFrame
var gradFrame = tensors.ToGradientFrame();   // Gradients as NivaraFrame (null if no grads)
```

### Batch Operations

```csharp
tensors.BatchBackward(loss);    // Runs loss.Backward(), then verifies every listed
                                // requires-grad tensor received a gradient
tensors.BatchZeroGrad();        // Calls ZeroGrad() on all tensors
```

### Type Checking

```csharp
NivaraAutoGradExtensions.IsAutoGradSupported<T>();
NivaraAutoGradExtensions.GetSupportedAutoGradTypes();  // [typeof(float), typeof(double), typeof(Half), typeof(BFloat16)]
```

---

## Exception Types

| Exception | Context | Key Properties |
|-----------|---------|----------------|
| `AutoGradException` | Base class for all autograd errors | `OperationContext`, `InvolvedShapes`, `GetDetailedContext()` |
| `ShapeIncompatibilityException` | Shape mismatch in operations | `ExpectedShape`, `ActualShape` |

---

## Examples

Manual examples that call `Backward()` run inside `GradientUtils.Grad()`.
Plain forward calls outside this scope are inference and do not build a
computation graph.

### 1. Basic scalar gradient

```csharp
using (GradientUtils.Grad())
{
    var a = new ReverseGradTensor<float>(
        NivaraColumn<float>.Create(new float[] { 3.0f }), requiresGrad: true);
    var b = new ReverseGradTensor<float>(
        NivaraColumn<float>.Create(new float[] { 4.0f }), requiresGrad: true);

    var result = ReverseGradOperations.Add(a, b);  // 7.0
    result.Backward();

    Console.WriteLine(a.Grad[0]);  // 1.0  (∂result/∂a)
    Console.WriteLine(b.Grad[0]);  // 1.0  (∂result/∂b)
}
```

### 2. Non-scalar backward with explicit gradient

```csharp
var x = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f }), requiresGrad: true);
using (GradientUtils.Grad())
{
    var relu = ReverseGradOperations.Relu(ReverseGradOperations.Negate(x));
    // relu = max(-x, 0) = [0, 0, 0]

    var gradInput = new ReverseGradTensor<float>(
        NivaraColumn<float>.Create(new float[] { 1.0f, 1.0f, 1.0f }), requiresGrad: false);
    relu.Backward(gradInput);
}

Console.WriteLine(x.Grad[0]);  // 0.0  (∂relu/∂x at index 0: -1 < 0 → 0)
Console.WriteLine(x.Grad[1]);  // 0.0
Console.WriteLine(x.Grad[2]);  // 0.0
```

### 3. Small neural network

```csharp
// y = mean(relu(x * w + b))
var x = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f }), requiresGrad: false);
var w = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new float[] { 0.5f, 0.5f, 0.5f }), requiresGrad: true);
var b = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new float[] { -1.0f, 0.0f, 1.0f }), requiresGrad: true);

using (GradientUtils.Grad())
{
    var mul = ReverseGradOperations.Multiply(x, w);    // [0.5, 1.0, 1.5]
    var add = ReverseGradOperations.Add(mul, b);       // [-0.5, 1.0, 2.5]
    var relu = ReverseGradOperations.Relu(add);        // [0.0, 1.0, 2.5]
    var mean = ReverseGradOperations.Mean(relu);       // 1.1667

    mean.Backward();
}

Console.WriteLine(w.Grad[0]);  // 0.0 (relu blocked gradient at index 0: -0.5 < 0)
Console.WriteLine(w.Grad[1]);  // >0 (∂mean/∂w at index 1 = x[1]/3 = 2/3 ≈ 0.333)
Console.WriteLine(w.Grad[2]);  // >0 (∂mean/∂w at index 2 = x[2]/3 = 3/3 = 1.0)
```

### 4. Matrix multiplication

```csharp
var a = ReverseGradTensor<float>.FromMatrix(
    new float[] { 1, 2, 3, 4 }, rows: 2, cols: 2, requiresGrad: true);
var b = ReverseGradTensor<float>.FromMatrix(
    new float[] { 5, 6, 7, 8 }, rows: 2, cols: 2, requiresGrad: true);

using (GradientUtils.Grad())
{
    var c = ReverseGradOperations.MatMul(a, b);  // 2x2 matrix product
    var sum = ReverseGradOperations.Sum(c);      // scalar sum
    sum.Backward();
}

// a.Grad = grad @ bᵀ  (grad = [1], so a.Grad = bᵀ)
Console.WriteLine(a.Grad[0]);  // 5
Console.WriteLine(a.Grad[1]);  // 7
Console.WriteLine(a.Grad[2]);  // 6
Console.WriteLine(a.Grad[3]);  // 8
```

### 5. SGD optimizer update

```csharp
var param = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f }), requiresGrad: true);

using (GradientUtils.Grad())
{
    var loss = ReverseGradOperations.Sum(param);  // loss = 6
    loss.Backward();                       // grad = [1, 1, 1]
}

var updated = SGD<float>.SgdUpdate(param, 0.1f);
// updated = param - 0.1 * grad = [0.9, 1.9, 2.9]
// updated.RequiresGrad == false
```

### 6. Nivara Frame integration

```csharp
using Nivara.AutoDiff.Extensions;

var frame = NivaraFrame.Create(
    ("Age", NivaraColumn<float>.Create(new float[] { 25, 30, 35 })),
    ("Income", NivaraColumn<float>.Create(new float[] { 50000, 70000, 90000 }))
);

// Convert to tensors
var tensors = frame.ToReverseGradTensors<float>(
    new[] { "Age", "Income" }, requiresGrad: true);

// Forward: compute a loss
var income = tensors["Income"];
var loss = ReverseGradOperations.Sum(income);

// Backward
tensors.BatchBackward(loss);

// Extract gradients back to a frame
var gradFrame = tensors.ToGradientFrame();
// Columns: Age (null if no grad), Income (grad = [1, 1, 1])

// Extract updated parameters
var updatedFrame = tensors.ToFrame();

// Zero gradients for next iteration
tensors.BatchZeroGrad();
```

### 7. Module-based training with TrainingLoop

```csharp
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Training;

// Define model
class LinearModel : Module<float>
{
    Linear<float> L1;
    public LinearModel()
    {
        L1 = new Linear<float>(3, 1);
        RegisterModules(L1);
    }
    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> x)
        => L1.Forward(x);
}

// Data
var frame = NivaraFrame.Create(
    ("x0", NivaraColumn<float>.Create([1.0f, 2.0f, 3.0f, 4.0f])),
    ("x1", NivaraColumn<float>.Create([2.0f, 3.0f, 4.0f, 5.0f])),
    ("x2", NivaraColumn<float>.Create([3.0f, 4.0f, 5.0f, 6.0f])),
    ("y", NivaraColumn<float>.Create([6.0f, 9.0f, 12.0f, 15.0f]))
));

var loader = new DataLoader<float>(
    new TensorDataset<float>(frame, ["x0", "x1", "x2"], "y"),
    batchSize: 2, shuffle: false);

// Train
var model = new LinearModel();
var optimizer = new SGD<float>(learningRate: 0.01f);
optimizer.AddParameterGroup(model.GetParameters().Values);

var loop = new TrainingLoop<float>(
    model, loader,
    (pred, target) => new MSELoss<float>().Forward(pred, target),
    optimizer,
    epochs: 5);

var result = loop.Run();       // TrainingResult<float>
result.PrintSummary();
// Epoch   1 | Loss:  ... | Batches:  2 | Time: 0.02s
// Epoch   2 | Loss:  ... | Batches:  2 | Time: 0.02s
```

### 8. Graph diagnostics

```csharp
var loss = ...; // from a computation graph
var info = GradientUtils.GetGraphInfo(loss);
// GraphInfo { TotalNodes: 4, IsLeaf: false, RequiresGrad: true,
//   OperationCounts: { Multiply: 1, Add: 1, Relu: 1, Mean: 1 } }
// typed access: info.TotalNodes, info.IsLeaf, info.RequiresGrad, info.OperationCounts

var summary = GradientUtils.PrintGraphSummary(loss);
// Computation Graph Summary:
//   Total Nodes: 4
//   Is Leaf: False
//   Requires Grad: True
//   Operation Counts:
//     Multiply: 1
//     Add: 1
//     Relu: 1
//     Mean: 1

var description = GradientUtils.DescribeTensor(loss);
// ReverseGradTensor<Single>:
//   Length: 1
//   Requires Grad: True
//   Has Gradient: True
//   Is Leaf: False
//   Has Nulls: False
//   Gradient Norm: 1.000000
//   Operation: Mean
```

### 9. Gradient clipping

```csharp
// Per-value clipping
GradientUtils.ClipGradValue(tensor, maxValue: 1.0f);
// Each gradient element clamped to [-1.0, 1.0]

// Norm clipping
GradientUtils.ClipGradNorm(tensor, maxNorm: 5.0);
// Scales gradient if L2 norm > 5.0

// Global norm clipping across all parameters
GradientUtils.ClipGradNorm(new[] { w, b }, maxNorm: 5.0);
// Combines all gradients into one global norm, scales proportionally
```

### 10. Type conversion

```csharp
var floatTensor = new ReverseGradTensor<float>(
    NivaraColumn<float>.Create(new float[] { 1.5f, 2.5f }), requiresGrad: true);

var doubleTensor = floatTensor.ToDouble();
// ReverseGradTensor<double> with values [1.5, 2.5], requiresGrad: true

var backToFloat = doubleTensor.ToFloat(requiresGrad: false);
// ReverseGradTensor<float> with values [1.5, 2.5], requiresGrad: false
```

### 11. Non-nullable boundary (ADR-001)

AutoDiff is a non-nullable domain. Resolve nulls before crossing the
`NivaraColumn<T>` → `ReverseGradTensor<T>` boundary (constructors assert on
nulls via `Debug.Assert`):

```csharp
var raw = NivaraColumn.CreateFromNullable(new float?[] { 1.0f, null, 3.0f });

// Option 1: fill nulls with a sentinel value
var filled = raw.FillNull(0.0f);
var a = filled.ToReverseGradTensor(requiresGrad: true);  // [1.0, 0.0, 3.0]

// Option 2: drop null positions entirely
var dropped = raw.DropNulls();
var b = dropped.ToReverseGradTensor(requiresGrad: true);  // [1.0, 3.0]

// Option 3: resolve nulls during DataFrame → tensor batch conversion
var frame = NivaraFrame.Create(("x", raw));
var tensors = frame.ToReverseGradTensors<float>(new[] { "x" }, requiresGrad: true);
```

### 12. Model serialization

```csharp
// Train
var loop = new TrainingLoop<float>(model, loader, lossFn, optimizer, epochs: 10);
var result = loop.Run();

// Save model
ModelSerializer.Save(model, "model.json");

// Load and run inference
var loaded = new LinearModel();
ModelSerializer.Load(loaded, "model.json");
loaded.Eval();
var prediction = loaded.Forward(testInput);
```

---

## Implementation Map

| Component | File |
|-----------|------|
| `GradTensor<T>` base class | `src/Nivara/AutoDiff/GradTensor.cs` |
| `ReverseGradTensor<T>` | `src/Nivara/AutoDiff/ReverseGradTensor.cs` |
| `ForwardGradTensor<T>` | `src/Nivara/AutoDiff/ForwardGradTensor.cs` |
| `OpNode<T>` | `src/Nivara/AutoDiff/OpNode.cs` |
| `ComputationGraph` | `src/Nivara/AutoDiff/ComputationGraph.cs` (internal) |
| `ReverseGradOperations` (all ops) | `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` |
| `ForwardGradOperations` (JVP ops) | `src/Nivara/AutoDiff/Operations/ForwardGradOperations.cs` |
| `AutoDiffDiagnostics` | `src/Nivara/AutoDiff/AutoDiffDiagnostics.cs` |
| `SGD<T>.SgdUpdate` (static) | `src/Nivara/AutoDiff/Optimizer/SGD.cs` |
| `GradientUtils` | `src/Nivara/AutoDiff/Utilities/GradientUtils.cs` |
| `TypeValidator` | `src/Nivara/AutoDiff/Utilities/TypeValidator.cs` |
| `TypeConverter` | `src/Nivara/AutoDiff/Utilities/TypeConverter.cs` |
| `NivaraAutoGradExtensions` | `src/Nivara/AutoDiff/Extensions/NivaraAutoGradExtensions.cs` |
| Exception types | `src/Nivara/AutoDiff/Exceptions/AutoGradExceptions.cs` |
| `Parameter<T>` | `src/Nivara/AutoDiff/Nn/Parameter.cs` |
| `Module<T>` | `src/Nivara/AutoDiff/Nn/Module.cs` |
| `Linear<T>` | `src/Nivara/AutoDiff/Nn/Linear.cs` |
| `Sequential<T>` | `src/Nivara/AutoDiff/Nn/Sequential.cs` |
| `Activation<T>` / `Dropout<T>` | `src/Nivara/AutoDiff/Nn/Activation.cs` / `Dropout.cs` |
| `Embedding<T>` | `src/Nivara/AutoDiff/Nn/Embedding.cs` |
| `SparseEmbedding<T>` | `src/Nivara/AutoDiff/Nn/SparseEmbedding.cs` |
| `TextClassifierModel<T>` | `samples/Nivara.Samples/TextClassifierModel.cs` *(moved from core in 1.2.0)* |
| `TokenClassifierModel<T>` | `samples/Nivara.Samples/TokenClassifierModel.cs` *(moved from core in 1.2.0)* |
| `VAE<T>` | `src/Nivara/AutoDiff/Nn/VAE.cs` |
| `ConvVAE<T>` | `src/Nivara/AutoDiff/Nn/ConvVAE.cs` |
| `Conv1d<T>` | `src/Nivara/AutoDiff/Nn/Conv1d.cs` |
| `Conv2d<T>` / `ConvTranspose2d<T>` | `src/Nivara/AutoDiff/Nn/Conv2d.cs` |
| `BatchNorm1d<T>` / `BatchNorm2d<T>` | `src/Nivara/AutoDiff/Nn/BatchNorm.cs` |
| `BatchNormKernel<T>` | `src/Nivara/AutoDiff/Nn/BatchNormKernel.cs` |
| `LayerNorm<T>` | `src/Nivara/AutoDiff/Nn/LayerNorm.cs` |
| `LayerNormKernel<T>` | `src/Nivara/AutoDiff/Nn/LayerNormKernel.cs` |
| `RMSNormKernel<T>` | `src/Nivara/AutoDiff/Nn/RMSNormKernel.cs` |
| `Gelu<T>` (tanh approximation; `ReverseGradOperations.Gelu` + `Activation.Gelu` wrapper) | `src/Nivara/AutoDiff/Nn/Activation.cs`, `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` |
| `MaxPool2d<T>` | `src/Nivara/AutoDiff/Nn/MaxPool2d.cs` |
| `AdaptiveAvgPool2d<T>` | `src/Nivara/AutoDiff/Nn/AdaptiveAvgPool2d.cs` |
| `DepthwiseSeparableConv2d<T>` | `src/Nivara/AutoDiff/Nn/DepthwiseSeparableConv2d.cs` |
| `TransformerBlock<T>` | `src/Nivara/AutoDiff/Nn/TransformerBlock.cs` |
| `MultiheadAttention<T>` | `src/Nivara/AutoDiff/Nn/MultiheadAttention.cs` |
| `Sampler<T>` | `src/Nivara/AutoDiff/Nn/Sampler.cs` |
| `TextTokenizer` | `src/Nivara/AutoDiff/Nn/TextTokenizer.cs` |
| Initializers (8) | `src/Nivara/AutoDiff/Nn/Initializers/*.cs` |
| Loss functions (7) | `src/Nivara/AutoDiff/Nn/Functional/*.cs` |
| `Optimizer<T>` base | `src/Nivara/AutoDiff/Optimizer/Optimizer.cs` |
| `SGD<T>` | `src/Nivara/AutoDiff/Optimizer/SGD.cs` |
| `Adam<T>` | `src/Nivara/AutoDiff/Optimizer/Adam.cs` |
| `AdamW<T>` | `src/Nivara/AutoDiff/Optimizer/AdamW.cs` |
| `TensorDataset<T>` | `src/Nivara/AutoDiff/Training/TensorDataset.cs` |
| `DataLoader<T>` | `src/Nivara/AutoDiff/Training/DataLoader.cs` |
| `Batch<T>` | `src/Nivara/AutoDiff/Training/Batch.cs` |
| `TrainingLoop<T>` | `src/Nivara/AutoDiff/Training/TrainingLoop.cs` |
| `DataParallelTrainer<T>` | `src/Nivara/AutoDiff/Training/DataParallelTrainer.cs` |
| `DataParallelTrainingResult<T>` | `src/Nivara/AutoDiff/Training/DataParallelResult.cs` |
| `ModelSerializer` | `src/Nivara/AutoDiff/Serialization/ModelSerializer.cs` |
| `Checkpoint<T>` | `src/Nivara/AutoDiff/Serialization/Checkpoint.cs` |
| Tests (22 files) | `tests/Nivara.Tests/AutoDiff/*.cs` |
