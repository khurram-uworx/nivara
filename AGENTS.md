# Guidance for AI-assisted coding

**GitHub repo:** `https://github.com/khurram-uworx/Nivara` — use `--repo khurram-uworx/Nivara` for `gh` commands.

## Facts & Research

- Nivara AutoDiff product direction updated: inference is the default/common path; reverse-mode training is opt-in via using (GradientUtils.Grad()). Do not implement NoGrad as the primary API. Built-in training loops should enter Grad() internally, while manual training examples/docs should wrap forward/loss/backward/optimizer code in Grad(). <!-- id=46e6ece9693740e49c42877dcc92abab entity=default type=fact ts=2026-06-14T14:10:17.7172876+00:00 v=1 tags=Nivara,AutoDiff,GradScope,inference-default -->
- Nivara AutoDiff ADR-001 (non-nullable domain) is fully implemented: null-mask infrastructure removed from AutoDiff ops/hot paths; Debug.Assert guards on ReverseGradTensor constructors and ComputationGraph.AddNode enforce the boundary. Type constraint relaxed from INumber<T> to IFloatingPointIeee754<T>, which passes Half/F16 and BFloat16 through runtime validation alongside float/double (on .NET 11 `System.Numerics.BFloat16` implements `IBinaryFloatingPointIeee754<BFloat16>` and is admitted by `TypeValidator`; kernel tests added in issue #137). All AutoDiff ops are span-ified (no NivaraColumn.Data access; Span<T> + TensorPrimitives). <!-- id=3fec03ab399d47b7a6e7450785292834 entity=default type=fact ts=2026-06-14T13:51:53.8257787+00:00 v=3 tags=Nivara,AutoDiff,refactor,planning -->
- Key BCL .NET 10 tensor patterns found via MS Learn: TensorPrimitives now generic (200+ overloads for any INumber/IRootFunctions T), ReadOnlyTensorSpan<T> with TryGetSpan, implicit conversion from T[] to ReadOnlyTensorSpan<T>, Tensor<T> stable in .NET 10. Spans are the currency. <!-- id=8e2b4e3243a847899af3c2d8ce7acceb entity=default type=research ts=2026-06-08T05:58:48.7959337+00:00 v=1 tags=MSLearn,BCL,tensor,patterns -->
- Nivara v1.4.0 shipped: public streaming API (`QueryFrame.AsStream`, `ScanAsQueryFrame` factories), `Over()`/`WindowSpec` window functions, fused expression engine (`FusedExpressionEvaluator` with SIMD backend + flat IR fallback), genuinely async `CollectAsync`, conditional expressions (`?:` in LINQ DSL), new aggregations (Quantile, Median, StdDev, Variance), public QueryPlan/ExecutionEngine/IExecutionStrategy/QueryDiagnostics/ExecutionProgress. <!-- id=v140-shipped entity=default type=fact ts=2026-08-21T00:00:00Z v=1 tags=Nivara,v1.4.0,streaming,windows,expressions,async -->

## Shell environment (Windows with GNU coreutils)

This environment has GNU coreutils at `C:\Program Files\coreutils\bin\` on PATH. Most Linux commands work directly (`grep`, `find`, `touch`, `sort`, `head`, `tail`, `wc`, `cat`, `ls`, `rm`, `mv`, `cp`). PowerShell aliases map `rm`/`mv`/`cp`/`cat`/`ls` to their PowerShell cmdlet equivalents, which behave similarly for basic file operations. Use normal command syntax — avoid verbose PowerShell idioms like `Remove-Item -LiteralPath`.

**GitHub CLI body gotcha:** Always write issue/PR bodies to a temp file and use `--body-file` instead of inline `--body`. PowerShell interprets backslash sequences in double-quoted strings (e.g. `\t` → tab, `\n` → newline), and special characters (backticks, quotes, backslashes) silently break or truncate `gh` inline bodies. This applies to `gh issue create`, `gh issue comment`, `gh pr create`, and `gh pr edit`. Pattern:
```
write the body to a temp file → gh issue comment N --repo ... --body-file /path/to/file.md
```
Do NOT rely on inline `--body "..."` for anything beyond trivial one-liners.

**Solution file:** This repo uses `.slnx` (XML-based solution format), not `.sln`. Build with `dotnet build Nivara.slnx`.

Purpose
- Concise, machine-friendly rules and locations to guide automated code generation and human edits that use System.Numerics.Tensors opportunistically.
- Designed to be consumed by AI assistants when producing or refactoring tensor-aware code.

Principles (high level)
- Use tensor-backed storage for vectorizable, unmanaged types; use memory-backed storage otherwise.
- Prefer zero-copy tensor/span paths when the data contains no nulls and a TensorSpan/AsTensorSpan is available.
- Use System.Numerics.Tensors `Tensor<T>`, `TensorSpan<T>`, and `TensorPrimitives` for float/double kernels; provide safe scalar fallbacks for other types.
- Preserve explicit null semantics: null masks are authoritative and must be propagated (mask OR semantics) in arithmetic and comparison results.
- Minimize allocations on hot paths: avoid repeated FlattenTo allocations, rent large buffers, and cache flattened buffers when safe.

Where to look (implementation map)
- Storage and selection
  - `src/Nivara/Storage/ColumnStorage.cs` — the single unified storage class (sole-owner `T[]` + optional `bool[]` null mask; zero-copy Slice; lazy zero-copy `AsTensor()` for unmanaged types).
  - `src/Nivara/Storage/ColumnStorageFactory.cs` — runtime selection: `IsVectorizable<T>()` and `Create<T>(ReadOnlySpan<T>)`.

- Column kernels and high-level ops
  - `src/Nivara/NivaraColumn.cs` — arithmetic, comparison, mask propagation, and use of `TensorPrimitives` for `float`/`double`.

- Tensor helpers & interop
  - `src/Nivara/Tensors/TensorInteropExtensions.cs` — conversions `Series/Frame <-> Tensor`, `TensorSpan` utilities, reshape/flatten helpers.
  - `src/Nivara/Tensors/TensorsHelper.cs` — consolidated tensor kernel helpers (MatMul, SoftMax, Sigmoid, Tanh, Transpose) with null-aware variants and BCL swap-target annotations.

- Kernel selection & diagnostics
  - `src/Nivara/KernelSelector.cs` — centralized `DetermineKernelType` heuristics (used by `NivaraColumn` and `ColumnDiagnostics`).

- Execution engine
  - `src/Nivara/Execution/ExecutionEngine.cs` — `ExecutionStrategy` enum, `IExecutionStrategy` interface, strategy routing, `LastDiagnostics`.
  - `src/Nivara/Execution/ExecutionStrategyBase.cs` — shared base class eliminating boilerplate across all four strategies.
  - `src/Nivara/Execution/LazyExecutionStrategy.cs` — deferred plan execution with optimization.
  - `src/Nivara/Execution/EagerExecutionStrategy.cs` — immediate execution.
  - `src/Nivara/Execution/StreamingExecutionStrategy.cs` — chunk-based streaming with memory budget.
  - `src/Nivara/Execution/ParallelExecutionStrategy.cs` — multi-threaded dispatch with parallel operation interfaces.
  - `src/Nivara/Execution/ParallelExecutionHelper.cs` — chunking, parallel processing, aggregation utilities.

- Interfaces & query contracts
  - `src/Nivara/Interfaces.cs` — consolidated `IColumn`, `IColumn<T>`, `IColumnStorage<T>`, `IFrame`.
  - `src/Nivara/Query/IQueryInterfaces.cs` — consolidated `IQueryOperation<T>`, `IQueryOperation`, `IQuerySource`.

- Operation type constants
  - `src/Nivara/Query/OperationType.cs` — `OperationType.Filter`, `.Select`, `.Sort`, `.GroupBy`, `.Join`, etc.

- Window functions
  - `src/Nivara/Operations/WindowSpec.cs` — immutable `WindowSpec` with `PartitionBy` / `OrderBy` builders.
  - `src/Nivara/Tensors/PartitionedWindowEngine.cs` — shared partition → sort → compute → scatter engine for all window functions.
  - Window function overloads on `NivaraFrame` (eager) and `QueryFrame` (lazy) accept `WindowSpec`.

- Fused expression engine
  - `src/Nivara/Expressions/FusedExpressionEvaluator.cs` — lowers `ColumnExpression` AST to single-pass kernel (SIMD delegate path + `FusedKernel` fallback).

- Frame-level row-major interop
  - `src/Nivara/NivaraFrame.cs` — `ToTensors`, `TryGetRowMajorSpan`, `CopyToRowMajor`, `AsQueryFrame()` (frame tensor-axis math `Dot`/`CosineSimilarity`/`ColumnNorms`/`RowNorms` was removed in the AutoDiff refactor — use `TensorPrimitives` on column/row spans directly).
  - `src/Nivara/Query/QueryFrame.cs` — `AsStream(chunkSize, ct)` yields per-chunk `NivaraFrame` via `IAsyncEnumerable<T>`; `CollectAsync`/`ToListAsync` are genuinely async (no `Task.Run` hop).
  - Lazy query-frame factories: `Csv.ScanAsQueryFrame`, `Json.ScanAsQueryFrame`, `Parquet.ScanAsQueryFrame`.

- AutoDiff subsystem
  - `docs/AUTODIFF.md` — canonical reference for all AutoDiff operations, modules, optimizers, forward-mode AD, and DataFrame integration
  - `src/Nivara/AutoDiff/` — core reverse-mode autograd engine (ReverseGradTensor, GradNode, IGradOperation)
  - `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` — all differentiable operations (VJP rules), span-ified with `TensorPrimitives`
  - `src/Nivara/AutoDiff/Operations/ForwardGradOperations.cs` — forward-mode JVP operations
  - `src/Nivara/AutoDiff/Optimizer/SGD.cs` — SGD optimizer, SgdUpdate, momentum buffers via `ArrayPool` (single-path `TensorPrimitives`)
  - `src/Nivara/AutoDiff/Optimizer/Adam.cs` — Adam optimizer with bias correction, SIMD `TensorPrimitives` chain, ArrayPool state
  - `src/Nivara/AutoDiff/Optimizer/AdamW.cs` — AdamW optimizer with decoupled weight decay, SIMD `TensorPrimitives` chain
  - `src/Nivara/AutoDiff/Nn/` — module system (Linear, Sequential, Parameter, activations)
  - `src/Nivara/AutoDiff/Nn/Conv1d.cs` — 1D convolution (im2col → TensorPrimitives.Dot kernel, PyTorch-compatible layout)
  - `src/Nivara/AutoDiff/Nn/Conv2d.cs` — 2D convolution (tiled im2col, PatchLocation lookup, grouped conv, 1×1 fast path, InputGrad specializations) and ConvTranspose2d
  - `src/Nivara/AutoDiff/Nn/BatchNorm.cs` — BatchNorm1d (2D `[N,C]` + 3D `[B,C,L]`) and BatchNorm2d (fused span kernel)
  - `src/Nivara/AutoDiff/Nn/BatchNormKernel.cs` — fused BatchNorm kernel implementation
  - `src/Nivara/AutoDiff/Nn/LayerNorm.cs` — LayerNorm module
  - `src/Nivara/AutoDiff/Nn/LayerNormKernel.cs` — LayerNorm kernel (TensorPrimitives.Dot SIMD)
  - `src/Nivara/AutoDiff/Nn/RMSNormKernel.cs` — shared PerRowRMSNorm kernel (TensorPrimitives.Dot backward)
  - `src/Nivara/AutoDiff/Nn/Activation.cs` — activation functional wrappers (ReLU, Sigmoid, Tanh, LeakyReLU, GELU) + GELU ops
  - `src/Nivara/AutoDiff/Nn/MaxPool2d.cs` — 2D max pooling module
  - `src/Nivara/AutoDiff/Nn/AdaptiveAvgPool2d.cs` — adaptive average pooling module
  - `src/Nivara/AutoDiff/Nn/DepthwiseSeparableConv2d.cs` — MobileNet-style depthwise separable conv
  - `src/Nivara/AutoDiff/Nn/ConvVAE.cs` — Fully convolutional VAE
  - `src/Nivara/AutoDiff/Nn/TransformerBlock.cs` — Pre-norm transformer block (NormType enum: RMSNorm/LayerNorm, GELU FFN)
  - `src/Nivara/AutoDiff/Nn/MultiheadAttention.cs` — Standalone self/cross-attention module (padding mask support)
  - `src/Nivara/AutoDiff/Nn/VAE.cs` — Variational autoencoder with optional conditioning
  - `src/Nivara/AutoDiff/Training/` — TrainingLoop, DataParallelTrainer, batch management
  - `src/Nivara/AutoDiff/Serialization/` — ModelSerializer for JSON save/load and state-dict JSON wrappers

- Factory & utilities
  - `src/Nivara/Storage/ColumnStorageFactory.cs` — `Create<T>(ReadOnlySpan<T>)`, `Create<T>(ReadOnlySpan<T>, ReadOnlyMemory<bool>? nullMask)`, `CreateFromOwnedArray<T>`, and `IsVectorizable<T>()`; all produce the single `ColumnStorage<T>`.
  - `src/Nivara/Tensors/NivaraTensorExtensions.cs` — `NivaraColumn<T>` extension methods for element-wise math, gradient helpers.
  - `src/Nivara/Tensors/TensorInteropExtensions.cs` — tensor helpers used across codebase.

Key rules for AI Agents to follow when generating tensor-aware code
1. Storage selection
   - Call `Nivara.Storage.ColumnStorageFactory.IsVectorizable<T>()` when you need to know whether `T` supports vectorized kernels.
   - All columns use the single `ColumnStorage<T>` (sole-owner `T[]` + optional `bool[]` null mask); kernel dispatch is decided by `KernelSelector`, never by the storage class.

2. Zero-copy preference
   - Use `NivaraColumn<T>.TryGetSpan` (returns a zero-copy read-only view when the column has no nulls, `false` otherwise) or `AsTensorView()` (lazy zero-copy `Tensor<T>` for unmanaged types; check `HasNulls` first) to avoid allocations.
   - When creating a `TensorSpan`/`Tensor<T>` from an existing span, ensure there are no nulls; otherwise throw or fallback to copy.

3. Kernel use
   - Use `TensorPrimitives` for `float` and `double` arithmetic/comparisons. Prefer generic `TensorPrimitives` overloads (e.g., `TensorPrimitives.CosineSimilarity<T>`) when available for type flexibility without branching.
   - Provide robust scalar fallbacks using `Span<T>` loops or `INumber<T>` where available.

4. Null-mask semantics
   - Null propagation rule: resultNullMask = leftNullMask OR rightNullMask (or leftNullMask for scalar op)
   - Where result is boolean (comparisons), null positions should be represented in the result null mask and the boolean output at those positions should be false (SQL-like semantics). Always keep the mask.

5. Minimize allocations
   - Avoid calling `Tensor.FlattenTo` repeatedly in hot loops.
   - For temporary buffers larger than 1024 elements, rent arrays from `ArrayPool<T>.Shared` (core) or `BufferPool` (Extensions) and return promptly.
   - `AsTensorView()` caches the lazy `Tensor<T>` view inside `ColumnStorage<T>`; reuse it rather than rebuilding tensors.

6. Kernel selection heuristics
   - Implement or reuse `DetermineKernelType()` that considers:
     - `IsVectorizable` (type level)
     - `Vector.IsHardwareAccelerated`
     - `Length >= vectorSize * 4` (heuristic threshold, configurable)
   - If kernel selection resolves to scalar, avoid preparing tensor copies.

7. Safe type dispatch
   - When converting spans/arrays to typed `Nivara.Storage.ColumnStorage<T>`, convert to arrays first and then use `ColumnStorageFactory.Create<T>(...)` or `CreateFromOwnedArray<T>`. Avoid `MemoryMarshal.Cast` unless `T` is unmanaged.
   - Keep explicit type-switch branches for each supported primitive (int, float, double, long, short, byte, bool, etc.).

8. Consolidating duplicate logic
    - When the same helper logic (e.g., type checks, validation, utility methods) is duplicated across multiple files, promote one authoritative implementation as an extension method on the most natural receiver type, then remove all copies and update call sites. Keep thin wrappers only if they serve a distinct API contract (e.g., `protected` visibility for subclass use). Place the extension in the same assembly to avoid cross-project visibility concerns.

9. Testing & diagnostics
    - Add unit tests covering null-mask propagation across arithmetic and comparisons.
    - Validate tensor conversions keep correct shape (`tensor.Lengths` is `nint[]`) and use casts `(int)tensor.Lengths[0]` in tests.
    - Record `Nivara.Diagnostics.OperationDiagnostics` for kernel selection and include in performance tests.
    - Use `ColumnDiagnostics`, `DiagnosticsTracker`, and `QueryDiagnostics` when changing kernel selection, query execution, or optimization behavior.

Suggested small, safe improvements to implement (prioritized)
- ✓ Cache flattened buffer in `Nivara.Storage.TensorStorage` (internal, lazy) — SUPERSEDED by the 1.2.0 storage consolidation: `TensorStorage` was deleted; `ColumnStorage<T>` owns a `T[]` and `Slice` is zero-copy, so no flattened copy cache is needed.
- ✓ Add internal `AsTensorSpanIfNoNulls()` to `Nivara.Storage.TensorStorage` — SUPERSEDED by `ColumnStorage<T>.AsTensor()` / public `NivaraColumn<T>.AsTensorView()`.
- ✓ Add `BufferPool.Rent(int size)` usage in `NivaraColumn` heavy paths — DONE (Phase 0).
- ✓ Implement `DetermineKernelType` central helper — DONE as `KernelSelector.DetermineKernelType()` (Phase 1).
- ✗ `RowNorms`/`ColumnNorms` on `NivaraFrame` — REMOVED in the AutoDiff refactor (Task 10): the frame tensor-axis methods (`Dot`/`CosineSimilarity`/`ColumnNorms`/`RowNorms`) had no production callers and were deleted along with the `TensorsHelper.RowNorms` SIMD kernel. Use `TensorPrimitives` on column/row spans directly.
- ✓ Add `TopKDescending` on `NivaraSeries<T>` — DONE (Phase 3).

Common gotchas (use these as lint-like checks in generated code)
- `ReadOnlyMemory<T>?` has `HasValue == true` for empty memory; always check `.Length > 0` to decide if mask exists.
- Slicing null masks: always check `.Length > 0` before slicing to avoid invalid operation on empty memory.
- `Tensor.Create(..., [length])` in codebase should use `new nint[] { length }` or `new ReadOnlySpan<nint>(new nint[] { length })` for dimensions; ensure creation uses correct API overloads.
- `Tensor.Lengths` is `nint[]`, not `int[]`.
- Avoid reflection/emits that attempt to pass `Span<T>` to `MethodInfo.Invoke` — convert to arrays first in tests and generated helpers.
- Nullable generics & static constraints (CS0080): avoid `where T : struct` on static methods in generic classes. Validate at runtime and throw clear exceptions.
- MemoryMarshal.Cast requires unmanaged constraints; use explicit type switch with `(T)(object)` casting for safe conversion.
- Tensor interop: zero-copy is limited — `NivaraColumn` doesn't expose underlying data as `Span`; interop requires element-by-element copying.
- Series indexer ambiguity: boxed `int` routes to label indexer. Use explicit casts or `GetByLabel()` to disambiguate integer labels vs positions.
- Method overload resolution: disambiguate 1D vs 2D tensor methods with explicit parameters (e.g., `FromTensor<T>(tensor, null)` for 2D).
- Expression Equals/GetHashCode: always override when adding custom equality operators to expression types.
- `Memory<T>` disposal: implement `IDisposable` consistently for frames, columns, and data sources.
- `NivaraColumn.TryGetSpan` returns `ReadOnlySpan<T>` (immutable guarantee), diverging from BCL's `Tensor<T>.TryGetSpan` which returns `Span<T>` (mutable). This is deliberate — Nivara columns are immutable. Use `CopyTo(Span<T>, T)` for the explicit-fill path when nulls are present or mutation is needed.
- `DataFrameOperation` no longer has strategy-switch dispatch or `Strategy` property — it was simplified to a single `Execute()` abstract method. Strategy dispatch is the `ExecutionEngine`'s responsibility via `IExecutionStrategy`.

Example patterns (pseudocode for AI Agent to reuse)

- Zero-copy tensor kernel (safe path)

```csharp
// Precondition: tensorStorage.HasNulls == false
var span = tensorStorage.AsTensorSpan(); // returns TensorSpan<T>
// Call kernel that accepts TensorSpan<T> directly
MyKernels.AddTensorSpan(span, otherSpan, destinationSpan);
```

- Safe tensor creation from nullable values

```csharp
var data = new T[len];
var nullMask = new bool[len];
for (int i=0;i<len;i++) {
  if (values[i].HasValue) data[i] = values[i].Value; else { data[i] = default; nullMask[i] = true; }
}
var tensor = Tensor.Create(data, new nint[] { len });
var nullTensor = hasNulls ? Tensor.Create(nullMask, new nint[] { len }) : null;
```

How AI Assistant should use this file
- Prefer deterministic, explicit code that follows rules above.
- Emit checks and fallbacks rather than optimistic, zero-check assumptions.
- When suggesting performance changes, include a small test that validates correctness (null mask and value equality).

Testing & diagnostics patterns
- Ask before running `dotnet test` or any long-running test/verification command; wait for explicit confirmation before starting it.
- The `tests/` directory holds three projects: `Nivara.Tests` (NUnit 4.x unit tests), `Nivara.PerformanceTests` (standalone stopwatch harness — scenario rows + on-demand modes like `--safetensors-mmap <path>` and `--dataset-test`, no NUnit/BDN, `dotnet run -c Release`), and `Nivara.SimdProbe` (self-contained SIMD probe, `dotnet run -c Release -- correctness|benchmark`). Each non-NUnit project has its own `README.md`.
- Avoid `[TestCase]` with null arrays; use regular `[Test]` with inline arrays.
- For complex anonymous-type arrays, prefer explicit typed tests or separate focused tests per type.
- Reflection cannot pass `Span<T>` via `MethodInfo.Invoke` — convert to array first.
- Test for key phrases in error messages rather than exact message strings.
- Property-like tests: implement with parameterized NUnit test suites rather than full FsCheck.
  - FsCheck has limited visibility in mainstream C#; AI agents struggle to produce correct FsCheck code in C#.
- Native integer types (`nint`): use `nint` for test assertions when comparing tensor dimensions.
- Method overload disambiguation: use explicit parameters to resolve ambiguous generic method calls in tests.
- Property-based test naming: use descriptive names with "Property" prefix and feature categories.
- Resource-management tests that depend on weak-reference cleanup may force multiple GC cycles; avoid GC forcing in normal code paths.

Representative testing pattern for null handling
```csharp
[Test]
public void NullMaskMaintenance_ArithmeticOperations_PreservesNullPositions()
{
    var testCases = new[] { new int?[] { 1, null, 3 } };
    foreach (var values in testCases)
    {
        var column = NivaraColumn<int>.CreateFromNullable(values);
        var result = column.Multiply(5);
        for (int i = 0; i < values.Length; i++)
            Assert.That(result.IsNull(i), Is.EqualTo(values[i] == null));
    }
}
```

Property-based test pattern
```csharp
[Test]
[Category("Feature: nivara-frame, Property 13: Type compatibility validation")]
public void Property_ArithmeticCompatibility_ValidatesCorrectly()
{
    foreach (var (leftType, rightType) in compatiblePairs)
    {
        Assert.DoesNotThrow(() =>
            TypeCompatibilityValidator.ValidateArithmeticCompatibility(leftType, rightType, "test"));
    }
}
```

---

## Code Style

- `.editorconfig` at repo root is authoritative; follow it over any convention below.
- **Private fields:** `camelCase` without `_` prefix (`logger`, not `_logger`).
- **Member ordering:** inner classes → constructors → properties → methods; static before instance; private → protected → internal → public.
- **Primary constructors:** preferred for service/DI classes over classic constructor with field assignment.
- **Sealed by default:** use `sealed class` for non-abstract classes unless inheritance is explicitly designed.
- **Collection expressions:** `[]` for empty/static collections; `new List<T>()` or `new Dictionary<K,V>()` for mutable ones.
- **Nullable reference types:** enabled; do not introduce avoidable warnings.
- **Omit braces** from single-line `if`/`else` bodies when the body fits one line and is on the same line as the condition.
- **No comments** in generated code unless explaining a non-obvious design decision.
- **No Hungarian notation** — no prefixes encoding scope or mutability.

## Testing Conventions

- **Framework:** NUnit 4.x — `[Test]`, `Assert.That(...)`, `Assert.ThrowsAsync`, no `[TestCase]`
- **Naming:** `Method_Scenario_ExpectedBehavior` PascalCase
- **Pattern:** Arrange-Act-Assert (AAA); no explicit comments needed
- **Organization:** one test class per source class, `*Tests.cs` suffix; split by behavior when a class is large
- **Mocking:** prefer real implementations where feasible; use `Substitute.For<T>()` only when external dependencies require it

---

## I/O & Interop Guidance

### General Principles
- Keep third-party dependencies in `Nivara.Extensions`; core stays dependency-free.
- Map CLR ↔ Arrow ↔ Parquet with explicit dictionaries and fallback suggestions.
- Handle nullable value types by extracting underlying types via `Nullable.GetUnderlyingType()`.

### Arrow Interoperability
- Build Arrow arrays using builders and individual `Append`/`AppendNull` calls.
- Convert `DateTime` to UTC (or configured timezone) and use `DateTimeOffset` for Timestamp arrays.
- Handle chunked arrays by iterating `chunkedArray.ArrayCount` and extracting each chunk.
- Create valid empty schemas/record batches for empty tables rather than returning null.

### Parquet Read/Write
- **Reading**: validate schema first, then reconstruct columns — use `CreateFromNullable` for value types, build arrays preserving nulls for reference types.
- **Writing**: `Parquet.Net DataColumn` expects non-nullable arrays matching `DataField<T>` generic type; pass `default(T)` for nulls and set field as nullable. Preserve string nulls as null, not empty string.
- **Empty frames**: if Parquet requires fields and frame is empty, write a dummy "empty" column.

### CSV/JSON Sources
- Lazy sources: `IsLazy = true`, infer schema from samples (e.g., 100 rows).
- Eager sources wrap lazy ones and materialize immediately.
- Conservative type detection: int → double → string; fallback to string in ambiguous cases.
- Lazy sources should validate structure/schema early, collect scan errors while traversing data, and throw during `Collect()` with source and operation context.

### Current Dependency Versions (Extensions only)
- CsvHelper 33.1.0, Apache.Arrow 23.0.0, Parquet.Net 6.0.3, Microsoft.ML 5.0.0, System.Numerics.Tensors 10.0.10
- Treat these versions as a snapshot; check the relevant `.csproj` before API-sensitive work.

---

## Performance & Optimization Thresholds

- **Buffer pooling threshold**: rent arrays >1024 elements from `BufferPool` (in Extensions).
- **Default memory budget for streaming**: 256 MB (configurable).
- **Streaming chunk size**: derived from memory budget when unset (`clamp(budget/10 ÷ 100 bytes/row, 1000, 100000)`); row-group aligned for Parquet. Full contract in `docs/STREAMING.md`.
- **Vectorization overhead threshold**: prefer only when `Length >= vectorSize * 4` (heuristic).
- **Fused expression engine**: primary path compiles `ColumnExpression` AST to cached delegates (SIMD auto-vectorized). `FusedKernel` fallback for non-compilable expressions. No boxed fallback — non-fusible expressions throw.
- **FlattenTo**: cache flattened tensor data if multiple accesses needed; use single `FlattenTo` for one-time access.
- **StreamingBufferManager**: use bounded buffer manager (in Extensions) for large datasets with memory budgets and GC triggers.
- **Vectorization checks**: verify `Vector.IsHardwareAccelerated` and type vectorizability before using SIMD kernels.
- **Unmanaged constraint**: `ColumnStorage<T>.AsTensor()` / `NivaraColumn<T>.AsTensorView()` require unmanaged `T` (`int`, `float`, `double`, `long`, `bool`, etc.).
- **Resource management**: implement object-disposed guards and dispose frames, columns, and data sources consistently.
- **Diagnostics**: preserve diagnostic context when wrapping kernel, query, optimization, and I/O failures.

---

## Known Issues & Follow-ups

- **Parquet round-trip**: nullable value type null preservation may degrade — investigate (high priority).
- **Zero-copy Arrow arrays**: removed from the public API (claims-integrity triage, see CHANGELOG); real zero-copy returns with `ARROW-ROADMAP` Phase D (issue #94).
- ✓ **Interop coverage for extended CLR types (#190)**: Parquet and Arrow round-trip the extended domain (`Half`, `nint`/`nuint`, `char`, `DateOnly`/`TimeOnly`, `Guid`, `DateTimeOffset`, `TimeSpan`) — native Parquet/Arrow field/array arms plus widened types restored via `nivara.clrType.<column>` metadata (Parquet `CustomMetadata`, Arrow schema metadata); `Int128`/`UInt128` throw documented `UnsupportedTypeException`s. Arrow `Half` uses the native `HalfFloatArray` builder (Apache.Arrow 23.0.0) — no manual `ArrayData`. ML.NET `ToNivaraFrame` faithfully reads every primitive DataView type (numeric kinds, bool, string via `ReadOnlyMemory<char>`, DateTime, DateTimeOffset, keys→uint) instead of silently dropping or coercing; `ConvertToFloat` throws for non-numeric types instead of silent `0f` (null still maps to 0f). 
- ✓ **Column creation dynamic dispatch (#158)**: `ColumnFactory` (`src/Nivara/Helpers/ColumnFactory.cs`) centralizes dynamic column creation via cached `MakeGenericMethod` over null-safe kernels, covering the extended CLR domain (`Half`, `nint`/`nuint`, `Int128`/`UInt128`, `sbyte`/`ushort`/`uint`/`char`, `DateOnly`/`TimeOnly`, `DateTimeOffset`, `Guid`, `TimeSpan`). All four dispatch sites (aggregation/group-by result columns, join coalesce/gather, fused constant columns) route through it; window ops accept the full `INumber<T>` domain. Interop layers closed by #190.
- ✓ **Public zero-copy tensor view (#107)**: `NivaraColumn<T>.AsTensorView()` / `NivaraSeries<T>.AsTensorView()` are public — lazy `Tensor<T>` view over the backing array (throws on nulls / reference types, cached via `ColumnStorage<T>.AsTensor()`); callers treat it as read-only.
- **Tensor interop**: investigate more efficient conversion patterns for large datasets (element-wise `Series`/`Frame` ↔ `Tensor<T>` interop still copies); `AsTensorView()` covers the flat, null-free column case.
- **NivaraSeries TopKDescending**: added in Phase 3 on `NivaraSeries<T>` (not `NivaraFrame`), returns labeled results with null-propagating scores; threshold-based optimization not yet implemented.
- ✓ **NivaraFrame RowNorms/ColumnNorms removed**: the frame tensor-axis methods (`Dot`/`CosineSimilarity`/`ColumnNorms`/`RowNorms`) and the `TensorsHelper.RowNorms` kernel were deleted in the AutoDiff refactor (Task 10) — they had no production callers. Use `TensorPrimitives` on column spans (`TryGetSpan`) or row-major spans (`CopyToRowMajor`).
- **Phase D complete**: Execution engine overhauled — Pattern B (`DataFrameOperation` strategy dispatch) eliminated, real parallel and streaming implementations, diagnostics integration across all strategies, `OperationType` constants replacing magic strings, 1948 tests passing.
- ✓ **Group-by Sum/Mean full numeric domain (#169)**: `SumAggregation`/`MeanAggregation` now accept the full 17-type `GetNumericTypes()` domain plus `bool`. `uint`/`ushort`/`sbyte`/`ulong`/`char`/`bool` previously passed validation then threw in `Apply`; `Half`/`nint`/`nuint`/`Int128`/`UInt128` were rejected outright. Sum promotes per NivaraSeries rules (byte/sbyte/short/ushort/int/uint/char/bool → long, ulong → ulong, nint → Int128, nuint → UInt128, Int128 → Int128, UInt128 → UInt128, float/Half → double, decimal → decimal); widening uses typed `TResult.CreateChecked` instead of `Convert.ChangeType` because extended numeric cross-type conversions (for example int to Int128) are not generally supported by `Convert.ChangeType`; Mean converts widened sums via a typed `ToDouble` switch; group-by sums produce typed result columns (`CreateColumnFromValues` gained ulong/Int128/UInt128 arms).
- ✓ **Boxed/dynamic fallbacks purged**: the legacy `ExpressionEvaluator` no longer has a boxed object fallback — unsupported type/operator combos throw a clear `NotSupportedException` ("not supported by the typed evaluator"); bool And/Or runs through a typed `Zip` kernel and Guid/Half comparisons go through the typed comparison dispatch. `BoxedPathEvaluationCount` is a zero-guardrail (always 0). `NivaraColumn` arithmetic and `NivaraSeries` Sum/Average `dynamic` loops were replaced with `NumericTensorKernels<T>` type-switch dispatch (decimal/Half/nint/nuint/Int128/UInt128 added) plus clear throws for the residual; `divideByCount` covers 17 types (all `GetNumericTypes()` minus `bool` — `NivaraSeries.Average` throws only for bool; Half/nint/nuint/Int128/UInt128 added in #172, char added in #168). `TypeCompatibilityValidator.GetNumericTypes()` domain extended with nint/nuint/Int128/UInt128.
- ✓ **AutoDiff P0–P6 complete**: reverse-mode autograd, NN module system, full optimizer family (SGD, Adam, AdamW), training loops, data-parallel training, model serialization — all implemented in core `src/Nivara/AutoDiff/`
- ✓ **BCEWithLogitsLoss fused backward**: replaced multi-op decomposition (Relu + Abs + SoftPlus) with single `OpNode<T>` computing `sigmoid(x) - z` directly. Fixes subgradient error at x=0 where Relu and Abs both return 0. `LeakyRelu` default slope corrected from 0 to 0.01. ADR-001 null cleanup removed ~200 lines of dead null branches from AccumulateGradient, KL/sample ops, and AdamW.
- ✓ **Loss API unified (#180)**: all losses in `Nivara.AutoDiff.Nn.Functional` (`MSELoss`/`L1Loss`/`BCELoss`/`BCEWithLogitsLoss`/`CrossEntropyLoss`) inherit the abstract `Loss<T>` base storing a ctor-defaulted `Reduction` enum (`Sum`/`Mean`/`None`, default `Mean` for PyTorch parity); `Forward(p, t, Reduction)` overrides per call and shared `Loss<T>.Reduce` centralizes reduction (None = elementwise, Sum = `ReverseGradOperations.Sum`, Mean = Sum ÷ divisor). `BCELoss.eps` stays a ctor arg. `CrossEntropyLoss` keeps the batch-size divisor, gains `Reduction.None` per-sample NLL (`[N]`), and keeps the `int[]` label overload. The `reduceToMean` bool overloads are gone — use `Reduction.Sum`/`Reduction.Mean`. The `Softmax<T>`/`LogSoftmax<T>` Functional classes are deleted; use `Activation.Softmax(input, dim)` / `Activation.LogSoftmax(input, dim)`. PyTorch `reduction='none'` fixtures + NivaraTorch tests added.
- ✓ **BatchNorm1d 3D input**: accepts `[B, C, L]` in addition to `[N, C]`; normalizes each L position independently for Conv1d pipelines. Previously rejected 3D input with a rank error.
- ✓ **BatchNormKernel xHat fix**: `xHat` is now always populated regardless of `affine` flag. Previously, `BackwardInput` read uninitialized data when `affine=false`, producing incorrect gradients.
- ✓ **MSELoss reduceToMean**: `Forward(predictions, targets, reduceToMean: true)` divides sum-of-squares by element count, matching PyTorch's default `reduction='mean'`. Superseded by the unified `Loss<T>`/`Reduction` API (#180): the default reduction is now `Mean` for every loss and the bool overload is removed.
- ✓ **Inference-only path**: Slice/Concat/PerRowRMSNorm in `ReverseGradOperations` now use `GradientUtils.ShouldTrackGrad()` (not raw `RequiresGrad`), so graph nodes are never created outside `Grad()` scope. Regression guard added via `Debug.Assert` in `ComputationGraph.AddNode()`. Verification tests in `InferenceGraphTests.cs`. The product direction (inference-default) is now fully enforced in all operations.
- **ConvTranspose2d**: no grouped convolution support; grouped transpose would require new kernel paths.
- **ConvTranspose2d**: direct scatter produces zero-padded interior positions (stride > 1); test verified numerically correct but may look unexpected.
- **BatchNorm2d**: uses generic per-element kernel (not the fused `BatchNormKernel<T>` span path); functionally correct but slightly slower than optimal.
- **PerRowLayerNorm**: delegates to `LayerNormKernel` with per-row slicing instead of a fused multi-row kernel; functionally correct but not optimal for large row counts.
- ✓ **NivaraTorch suite**: 55 PyTorch-validated functional tests (`tests/Nivara.Tests/NivaraTorch/`) comparing forward/backward values against `gen_reference.py` fixtures covering 21+ layer types; regenerated via Python scripts in `samples/NivaraTorch/`.
- ✓ **IFloatingPointIeee754<T>**: AutoDiff type constraint relaxed from `INumber<T>`; `Half`/F16 and `BFloat16` pass runtime validation alongside `float`/`double`. All AutoDiff ops span-ified with `TensorPrimitives`. `System.Numerics.BFloat16` implements `IBinaryFloatingPointIeee754<BFloat16>` on .NET 11 and is admitted in `TypeValidator` (issue #137); its matmul runs via the BCL `TensorPrimitives.Dot` path. Load-time BF16→F32 widening in `SafeTensorsLoader` remains the default for float/double pipelines.
- ✓ **SIMD-accelerated kernels**: Adam, AdamW, PerRowRMSNorm backward, LayerNorm sum-of-squares all use `TensorPrimitives` chains; Adam/AdamW state buffers use `ArrayPool<T>.Shared`. `RMSNormKernel<T>` consolidates duplicated PerRowRMSNorm logic.
- ✓ **SafeTensorsLoader**: generic dtype-aware `Read<T>()` (I32/I64/F16/BF16/F32 → target element type) in `samples/Nivara.Samples`; powers MobileNetV2/ResNet-18 inference and DistilBERT fine-tuning samples.

---

## Quick Reference

- **Vectorizable types (confirmed)**: `int`, `float`, `double`, `long`, `short`, `byte`, `uint`, `ulong`, `ushort`, `sbyte`, `bool` (requires unmanaged constraint)
- **Target framework**: .NET 10.0 with System.Numerics.Tensors 10.0.10
- **Common deps (Extensions only)**: CsvHelper 33.1.0, Apache.Arrow 23.0.0, Parquet.Net 6.0.3, Microsoft.ML 5.0.0, System.Numerics.Tensors 10.0.10
- **Useful helpers**: `ColumnDiagnostics`, `DiagnosticsTracker`, `ColumnStorageFactory.IsVectorizable<T>()`, `NivaraColumn<T>.CreateFromNullable(T?[])`, `Tensor.Create(array)` + `FlattenTo(buffer)`, `KernelSelector.DetermineKernelType()`, `SGD<T>.SgdUpdate()`, `Adam<T>.AdamUpdate()`, `AdamW<T>.AdamWUpdate()` (functional single-tensor updates), `Adam<T>`, `AdamW<T>`, `Linear<T>`, `Sequential<T>`, `Module<T>.StateDict()`, `Module<T>.LoadStateDict()`, `TrainingLoop<T>`, `DataParallelTrainer<T>`, `ModelSerializer`, `Loss<T>`/`Reduction` (loss base + `Sum`/`Mean`/`None`), `Activation.Gelu`, `ReverseGradOperations.Gelu`
- **AutoDiff type constraint**: `IFloatingPointIeee754<T>` (float, double, Half) — ADR-001 non-nullable domain
- **Storage**: single `ColumnStorage<T>` for all types (sole-owner `T[]` + optional `bool[]` null mask; zero-copy Slice; lazy `AsTensorView()`); vectorization decided by `KernelSelector`
- **Null handling**: explicit boolean masks, no NaN-based semantics
- **Query execution**: lazy by default, multiple strategies (eager, streaming, parallel)

References (implementations to inspect)
- `src/Nivara/Storage/ColumnStorage.cs`
- `src/Nivara/Storage/ColumnStorageFactory.cs`
- `src/Nivara/NivaraColumn.cs`
- `src/Nivara/Tensors/TensorsHelper.cs`
- `src/Nivara/Tensors/TensorInteropExtensions.cs`
- `src/Nivara/KernelSelector.cs`
- `src/Nivara/NivaraFrame.cs`
- `src/Nivara/Interfaces.cs`
- `src/Nivara/Query/IQueryInterfaces.cs`
- `src/Nivara/Execution/ExecutionEngine.cs`
- `src/Nivara/Execution/ExecutionStrategyBase.cs`
- `src/Nivara/Execution/ParallelExecutionHelper.cs`
- `src/Nivara/AutoDiff/ReverseGradTensor.cs`
- `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`
- `src/Nivara/AutoDiff/Operations/ForwardGradOperations.cs`
- `src/Nivara/AutoDiff/Optimizer/SGD.cs`
- `src/Nivara/AutoDiff/Optimizer/Adam.cs`
- `src/Nivara/AutoDiff/Optimizer/AdamW.cs`
- `src/Nivara/AutoDiff/Nn/Linear.cs`
- `src/Nivara/AutoDiff/Training/TrainingLoop.cs`
- `src/Nivara/AutoDiff/Serialization/ModelSerializer.cs`

## Active AutoDiff Direction

The AutoDiff refactor (see ADR-001/ADR-002 and the CHANGELOG for the executed
plan) is **complete** — the implementation follows the direction below.

Current product direction:

- Inference is the common path and the default behavior.
- Reverse-mode graph construction is opt-in via `using (GradientUtils.Grad())`.
- `requiresGrad` still marks trainable tensors/parameters, but operation history
  is only recorded while `GradientUtils.IsGradEnabled` is true.
- Built-in training APIs (`TrainingLoop`, `DataParallelTrainer`) should enter
  `GradientUtils.Grad()` internally so high-level training stays simple.
- Manual training examples should wrap forward/loss/backward/optimizer code in
  `using (GradientUtils.Grad())`.
- Do not introduce `NoGrad` as the primary API. The intended user-facing model is
  "predict by default, train explicitly."
- The recent AutoDiff plan items are implemented: inference-default
  `GradientUtils.Grad()` plus `StateDict()` / `LoadStateDict()` and
  serializer helpers.

## Agent Framework Workflow Patterns

See the "Agent Framework integration patterns" section of `samples/NivaraChat/README.md` for Nivara-specific integration notes from the NivaraChat sample.

## Architectural Decisions (ADRs)

See `docs/adr/` for recorded decisions:

- **ADR-001** (`docs/adr/001-autodiff-nonnullable-domain.md`): AutoDiff is a non-nullable domain. Null boundary enforced at domain entry points (`NivaraColumn<T>` → `ReverseGradTensor<T>` conversion). All AutoDiff ops assume non-null data. Storage layer remains nullable.
