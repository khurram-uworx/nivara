# Changelog

All notable changes to Nivara are documented here. Released versions are published to NuGet via the tag-triggered CD workflow (`v*` tags on `main`).

## [Unreleased]

### Added

- **Optional QKV bias on Llama attention for Qwen2-family checkpoints (#384)** —
  `LlamaCausalAttention<T>` and `LlamaDecoderBlock<T>` gained a `bool qkvBias = false`
  constructor option. Qwen2-style models attach a bias to the self-attention
  `q_proj`/`k_proj`/`v_proj` projections; canonical Llama uses bias-free projections and
  stays byte-identical with the default. When enabled, only the Q/K/V projections carry a
  `[1, outFeatures]` bias parameter (`o_proj` and the FFN projections remain bias-free),
  and `LlamaLoader.Load` auto-detects biased Qwen2 checkpoints from safetensors
  (presence of `model.layers.0.self_attn.q_proj.bias`). Backward gradient flow through the
  new bias parameters is Torch-verified (`llama_attn_bias`/`llama_decoder_bias` parity
  fixtures) plus structural AutoDiff coverage.

### Changed

- **Nested (non-slot) cumulative windows fall back to exact boundary materialization (#360)** —
  `StreamingWindowProcessor.isStreamableNode` now only admits a cumulative window
  (`CumulativeSum`/`CumulativeMax`/`CumulativeMin`/`CumulativeProduct`/`CumulativeCount`)
  when it is a top-level carry slot. A cumulative window nested inside a larger expression
  (e.g. `CumulativeProduct(Col("A")) + Col("B")`) previously passed the streamable check
  but was not a carry slot, so per-chunk boundary runs re-scanned the cumulative from the
  run's first value — wrong values for sums/max/min and checked-`long` `OverflowException`
  for int-family products. Such selects now materialize the boundary once, exactly like
  rank/broadcast windows; top-level cumulative windows keep chunked carry-slot streaming.

- **net11 BCL tensor swap targets verified (#136)** — `TensorsHelper` MatMul/Transpose
  annotations now reflect the verified state of `System.Numerics.Tensors`
  11.0.0-preview.7: `Tensor.Transpose<T>` ships as a zero-copy strided view, so the
  cache-tiled kernel remains the physical materializer for contiguous-span consumers
  (`TensorPrimitives.Dot` paths); `Tensor.MatrixMultiply<T>` is unshipped
  (dotnet/runtime#95863, BLAS epic #93286), so handwritten matmul kernels stay with the
  swap-target annotation intact. The dead Tensor-level `Multiply` overloads were removed;
  new parity + performance regression gates in `TensorsHelperTests` fail if the BCL
  view-materialization route ever beats the tiled transpose.

- **BFloat16 joined the AutoDiff supported type set (#137)** — on .NET 11
  `System.Numerics.BFloat16` implements `IBinaryFloatingPointIeee754<BFloat16>`, so it
  natively satisfies the `IFloatingPointIeee754<T>` constraint. `TypeValidator` now admits it
  (and `ToReverseGradTensorsAuto` converts BFloat16 frame columns); new `BFloat16Tests`
  verify forward/backward parity with float references, `Linear` training under SGD/Adam, and the
  inference-default graph guard. The generic `TensorsHelper.MultiplyCoreGeneric` matmul dropped
  its hand-rolled `Vector<T>` SIMD branch (which threw `NotSupportedException` for `BFloat16`)
  in favor of the BCL `TensorPrimitives.Dot` row dot product, so Half matmul now computes too.

- **Lead / negative-shift windows stream via delayed emission (#331)** —
  `StreamingWindowProcessor` now covers lookahead windows: `Lead` expressions,
  negative-period `Shift` window expressions, and standalone `ShiftOperation` with
  `Periods < 0` stream per chunk instead of forcing full boundary materialization. Each
  round executes the boundary over one contiguous run — the last `lookback + lead`
  input rows plus the fresh chunk — and emits only the rows whose entire lookahead is
  satisfied by data seen so far; the held-back tail is re-computed once the next chunk
  arrives and finalized at drain with the operation's natural end-of-data semantics
  (nulls or `fillValue`). Cross-chunk memory stays bounded by
  `max(rolling lookback, lag periods) + max(lead periods)` rather than frame size, and
  mixed selects (lead alongside rolling/lag/cumulative windows) no longer materialize;
  cumulative columns defer computed values through a per-slot FIFO so they stay aligned
  with the delayed emission without replaying rows into the running aggregate.
  Cadence note for `AsStream`: plans containing lookahead windows yield one frame per
  chunk delayed by a single chunk plus a final flush frame carrying exactly the held-back
  rows (empty prefixes are suppressed). `StreamMaterializationCount` reports zero for
  such plans.

## [1.4.0] - 2026-08-21

### Added

- **Public streaming API with explicit non-streamable boundary behavior (#264)** —
  `QueryFrame.AsStream(chunkSize: 10000, ct)` is now public and `NivaraQuery<T>` gains an
  `AsStream` passthrough, so chunked async processing is reachable by external consumers.
  `AsStream` yields one `NivaraFrame` per source chunk for fully-streamable plans
  (Filter/Select/Slice/SelectRows, no window expressions) over chunk-capable sources
  (CSV, JSON, Parquet); any non-streamable boundary operation (Sort/SortByExpression/
  GroupBy/Join/Distinct/Rolling/Cumulative/Shift/Rank) or window expression, or a
  non-chunk-capable source, falls back to a single merged frame with rows identical to
  `CollectAsync()`. `NivaraFrame.AsQueryFrame()` / `NivaraQuery<T>.AsQueryFrame()` are
  public, and public lazy query-frame factories `Csv.ScanAsQueryFrame`,
  `Json.ScanAsQueryFrame`, and `Parquet.ScanAsQueryFrame` open the streaming entry point
  directly from files. `chunkSize` is honored by row-oriented sources and advisory
  (row-group aligned) for Parquet; when unset it is derived from the memory budget
  (`clamp(budget/10 ÷ 100 bytes/row, 1000, 100000)`). Full contract in
  `docs/STREAMING.md`; `QueryFrame.ToQueryPlan()` stays internal (see #275).

- **`Over()` / `WindowSpec` builder for window functions (#162)** — SQL-style partitioned
  windows on both `NivaraFrame` (eager) and `QueryFrame` (lazy). `Over()` returns an immutable
  `WindowSpec` with `PartitionBy(params string[])` and three `OrderBy` overloads
  (`SortKey[]`, `string[]`, or `string + SortDirection + NullOrdering`; ascending/NULLS LAST by
  default). Every window function gains a `WindowSpec` overload: rolling (`Sum`/`Mean`/`Min`/
  `Max`), cumulative (`Sum`/`Max`/`Min`/`Product`/`Count`), `Shift`/`Lead`, and the rank family
  (`RowNumber`/`Rank`/`DenseRank`/`PercentRank`, which keep existing null-order-key semantics).
  Execution runs partition → sort → compute → scatter via the shared
  `PartitionedWindowEngine` (`src/Nivara/Tensors/PartitionedWindowEngine.cs`); partition and
  order keys are validated up front (missing/non-comparable columns throw). A spec with no keys
  (`new WindowSpec()`) matches the plain overloads. See the Window Functions section of
  `docs/LINQ.md`.

- **Fused expression kernel IR + span/chunked execution (#167)** — the fused expression engine now lowers every expression tree to a single post-order `KernelPlan` (`KernelLowerer`/`KernelIR`) and routes it to one of three backends: a flat IR span interpreter in `FusedKernel` for null-bearing uniform numeric plans (per-element `ReadOnlyMemory<T>` leaf access, hoisted literals, inline OR null-mask), a `TensorPrimitives` SIMD backend for null-free single-op plans (Add/Subtract/Multiply/Divide, one BCL SIMD call), and the compiled offset delegate (`start`/`count`/`destStart`) for everything else (null-free chains, bool, heterogeneous plans). Execution is chunk-capable: `FusedExpressionEvaluator.EvaluateChunked(expression, input, chunkSize)` slices the existing contiguous leaf storage zero-copy and writes into one shared output array, bit-identical to whole-column evaluation — the memory-budgeted primitive the Phase 4 async streaming bridge (#171) needs. Chunked/whole bit-identity, null-mask propagation, and backend-routing guardrails are pinned by unit tests (`EvaluateChunked_*`, `Evaluate_NullFreeSingleOp_*`); `NodeTreePathEvaluationCount` was renamed `SpanKernelPathEvaluationCount` to reflect the IR span fallback. Design rationale recorded in `docs/adr/004-fused-expression-engine-kernel-ir-span-backends.md`.

- **BFloat16 widen-SIMD Phase 3: A/B + benchmark + CLI flag (branch `khurram/smollm-3`)** — the `NivaraInference` sample gains a `--simd-widen` flag, a `smollm benchmark` mode (median-of-3 full 32-token generation timing), and a `smollm ab` mode that runs scalar-vs-widen side-by-side. Measured on SmolLM-135M: the widen path is **~10× faster** than the BF16 scalar fallback (median 22.6 s vs ~225 s for a 32-token generation), and remains roughly 1.5–2× slower than F32 native (333 ms/token; ratio varies with machine load) — the widen path restores usable BF16 performance (halved weight memory) but does not beat native F32 compute. The toggle is a no-op for `float` (identical token streams, F32 A/B control). Correctness unchanged: SmolLM BF16 22/32 generated-token argmax + 0.937 final-logits cosine vs PyTorch; `distilbert_sst --precision bf16` argmax holds 8/8 with and without `--simd-widen`. Additive, switch-gated, no change to `src/Nivara` kernels.

### Changed

- **`CollectAsync`/`ToListAsync` now run genuinely asynchronously (#266)** — the async query
  path no longer hops to the thread pool via `Task.Run`. `LazyExecutionStrategy` gained an
  async-native `ExecuteCoreAsync` that drives sources through `IQuerySource.ExecuteAsync` and
  operations through `IQueryOperation.ExecuteAsync` (the default `ExecuteAsync` no longer
  wraps sync `Execute` in `Task.Run`), and `CsvLazySource`/`JsonLazySource` read chunks with
  real async I/O (`CsvReader.ReadAsync`, async buffer refill in `JsonRecordStreamReader`).
  Awaits continue on the caller's context and cancellation propagates without an extra thread.

- **Strongly-typed ML.NET training metrics (#233)** — `ModelIntegration.TrainAndEvaluate`
  now returns `(ITransformer Model, ModelEvaluationResult Metrics)` instead of
  `(ITransformer Model, object Metrics)`. `ModelEvaluationResult` is a sealed record with
  a `ModelTaskKind Kind` plus exactly one non-null typed metrics property (`Binary`,
  `Multiclass`, or `Regression`), populated by the pipeline-kind heuristic. Breaking: the
  previous untyped `object` return could not be consumed without reflection, so no
  supported callers are affected.

- **`TrainingResult.PrintSummary()` / `DataParallelTrainingResult.PrintSummary()` accept
  an optional `TextWriter` (#235)** — both now write to `writer ?? Console.Out`, so
  summaries can be routed to logs or captured in tests. Existing no-argument callers are
  unchanged.

- **`CsvOptions.SchemaInferenceRows` renamed to `SchemaInferenceRecords` (#235)** — matches
  the existing `JsonOptions.SchemaInferenceRecords` naming. Breaking for any caller that
  referenced the CSV option by its old name (including the `With(schemaInferenceRows: ...)`
  parameter).

- **Internal surface scoping (#235)** — members of already-internal classes that were
  mistakenly declared `public` are now `internal`: `ColumnStorageFactory` (`Create`,
  `IsVectorizable`), `TensorsHelper` (10 tensor kernels), and `RankKernel.Compute`.
  `RankKind` remains public. No behavioral change.

### Fixed

- **Streaming cancellation no longer masks the OCE with `ChannelClosedException` (#280)** —
  consumer-side cancellation of the bounded-channel pipeline (`StreamingExecutionStrategy.
  ExecuteCoreAsync`) surfaced `QueryExecutionException("Async Streaming execution failed:
  The channel has been closed.")` because the consumer catch called `channel.Writer.Complete()`
  on a channel the producer's `finally` had already completed (and `Complete()` throws when
  the channel is closed). The producer now tracks its in-flight chunk frame and disposes it in
  its own `finally`, both sides complete the channel with no-throw `TryComplete()`, the
  consumer catch drains and disposes channel-buffered frames after completing, and the producer
  task is awaited (swallowing its fault) so the consumer's own `OperationCanceledException`
  propagates cleanly and the task is never left unobserved. Phase 4 AC2 now holds: clean OCE
  with no resource leaks.

- **`docs/STREAMING.md` now documents the correct chunk-frame ownership contract (#278)** —
  the doc claimed chunk frames are "disposed by the pipeline after the consumer moves past
  them", but `StreamChunksAsync` yields raw frames and never disposes them (`await foreach`
  only disposes the enumerator), so callers following the doc leaked chunk frames. The doc now
  states the consumer owns each yielded frame (including the single-frame fallback) and shows
  `try/finally chunk.Dispose()` in the samples; `StreamChunksAsync`'s XML doc carries the same
  note.

- **`NivaraFrame.AsQueryFrame()` no longer aliases the source frame's columns (#279)** — the
  in-memory query source shared the source frame's own column instances but behaved as their
  owner: disposing a `QueryFrame` built from `AsQueryFrame()` disposed each column, and a
  collected result (no-op `Collect()`, or `Select` of a bare column reference, whose evaluator
  passes the input instance through) owned those same instances — so disposing either side
  threw `ObjectDisposedException` from the other. `MemoryQuerySource` is now explicitly
  non-owning: `Dispose()` only invalidates the source, and `Execute()`/`ExecuteAsync()` return
  fresh column instances (zero-copy `Slice(0, Length)` over the same backing storage) with
  independent disposal, so the source frame, the query frame, and each collected result are
  safe to dispose in any order and query frames stay reusable after disposing a result.

- **`QueryFrame` disposal now releases the underlying source on both sync and async paths (#268)** —
  `Dispose()` previously only untracked the frame and never disposed the `IQuerySource`, while
  `DisposeAsync()` released it only when the source happened to implement `IAsyncDisposable` (none
  do), so explicit disposal of a lazy CSV/Parquet frame could leak the persistent chunk-reader
  file handle while GC-abandonment cleanup did release it. `Dispose()` now calls `source.Dispose()`
  (swallowing errors, mirroring the abandoned-resource cleanup) and `DisposeAsync()` falls back to
  `source.Dispose()` for non-`IAsyncDisposable` sources. Fluent chains share one source, so
  disposing any node releases it — the same semantics abandoned-resource cleanup already applied.

- **Sync streaming execution now streams the streamable prefix before non-streamable boundary ops (#269)** —
  sync `ExecuteCore` re-checked `isSuitableForStreaming` and fell back entirely to Lazy for plans
  containing Sort/GroupBy/Join/Distinct/etc., while async `ExecuteCoreAsync` streamed the prefix and
  ran boundary ops on the materialized frame (flush-concatenate-resume). Both paths now behave
  identically: only window-expression plans fall back to Lazy; intermediate chunk frames are
  disposed after concatenation.

- **Streaming empty-source fallback no longer re-applies boundary ops (#270)** — when a chunk-capable
  source yields zero chunks, both sync `ExecuteCore` and async `ExecuteCoreAsync` fall back to a
  single full-plan execution; previously the flush-concatenate-resume segment loop then re-applied
  every non-streamable boundary op on the already-processed result (a pre-existing async edge case
  surfaced while aligning the paths in #269). Boundary ops now run exactly once on the empty-source
  path.

- **`QueryFrame.AsStream(chunkSize)` now honors the requested row count (#267)** — the chunk
  size was previously encoded as `MemoryBudget = chunkSize * 100` and then re-derived by the
  streaming strategy, which clamped small values to a 1000-row minimum. `NivaraExecutionContext`
  now carries an explicit `ChunkSize` that `AsStream` sets directly and the streaming strategy
  prefers over its budget-derived default; `StreamingExecutionStrategy.ValidatePlan` rejects
  non-positive values. Honored by row-oriented sources (CSV, JSON); advisory for columnar
  sources aligned to native row-group boundaries (e.g. Parquet).

- **Fused plan signatures now encode the literal runtime type (#246)** —
  `ExpressionTypeInferer.FormatValue` appends `:{value.GetType().FullName}` to each literal
  signature fragment, so two plans that differ only in literal types (e.g. `Column + (int)1`
  vs `Column + (long)1`) no longer collide in the fused-plan cache and return stale results.

- **Compiled fused kernels write masked positions before real values (#247)** — the compiled
  delegate now passes an OR'd `bool[] mask` to `NivaraColumn<T>.CreateFromSpans` and writes
  `Expression.Default(elementType)` at masked positions before the real-value pass. Previously
  the write order could leave a genuine computed value at a position the mask marked null.

- **Window-bearing operations now run whole-column in streaming/parallel execution (#245)** —
  `StreamingExecutionStrategy` and `ParallelExecutionStrategy` inspect operations for window
  expressions (`SelectOperation` columns, `FilterOperation` conditions) via the new
  `WindowExpressionInspector` and fall back to whole-column execution instead of silently
  producing incorrect chunked/parallel results. Streaming no longer reads chunks for such
  plans (`ChunksRead == 0`) and parallel dispatch honors the same gate.

- **Int-family window accumulators no longer wrap silently (#248)** — `RollingSum`/`RollingMean`
  prefix sums and `CumulativeSum`/`CumulativeProduct` accumulate in `long` for the int family
  (`sbyte`/`byte`/`short`/`ushort`/`int`/`uint`/`char`), matching `NivaraSeries` promotion.
  Per-window sums that stay in range are correct even when the running prefix overflows the
  element type; genuine overflow of the result type now throws `OverflowException` instead of
  silently wrapping. `long`/`float`/`double` and the max/min scans are unchanged.

- **Literal-only fused plans now constant-fold instead of throwing (#249)** — expressions
  built entirely from literals (e.g. `Lit(2) * 2`) previously threw `NotSupportedException`
  wrapped in `QueryExecutionException` because `FusedExpressionEvaluator.EvaluateCore` rejected
  any plan with zero leaf columns. Such plans now run through the compiled target at the input
  length (1 when there are no columns), which already supports zero leaf parameters and handles
  numeric/bool/comparison expressions alike. The span and `TensorPrimitives` backends still
  require a column leaf and are unchanged.

- **Fused-expression promotion and coercion extended to the native/wide numeric domain (#250)** —
  `NumericPromoter.GetPromotedType` and `FusedKernel.CoerceLiteral` now resolve mixed-type
  literal operands across `nint`/`nuint`/`Int128`/`UInt128`/`Half` per C# binary-numeric-promotion
  rules (e.g. `nint + uint → long`, `nint + nuint → double`, `Int128 + int → Int128`,
  `Int128 + UInt128 → double`, `decimal + Int128 → double`). The compiled delegate coerces
  operands and results via typed `TResult.CreateChecked` when `Expression.Convert` has no built-in
  conversion (e.g. `nint → double`, `UInt128 → double`), cached per `(source, target)` pair.
  Previously such mixed-type literal plans threw or produced wrong-typed results.

- **`RowNumber` numbers rows with null order keys instead of emitting null (#254)** —
  **behavior change.** `row_number` now numbers every partition row: null-key rows are placed per
  the order keys' `NullOrdering` (default `NULLS LAST`), e.g. ascending order keys
  `[2, null, 1, null]` produce `[2, 3, 1, 4]`. `rank`/`dense_rank`/`percent_rank` keep the
  existing null-order-key semantics (null output, excluded from numbering and the percent_rank
  denominator), which match Polars. The rank-family kernel is now cross-validated against
  committed Polars fixtures (`samples/data/polars-window/manifest.json`, regenerated by
  `samples/NivaraWindow/gen_reference.py`).

- **Synthetic window names no longer collide with user columns (#255)** — when hydrating
  materialized windows, `FusedExpressionEvaluator` now picks the first free `__window_<n>` name
  by scanning the input column names (ordinal, case-insensitive) instead of blindly using the
  next counter value. Previously a user column literally named `__window_0` was silently
  overwritten by the first materialized window.

- **`JsonLazySource` chunk reads are now truly streaming (#265)** — `ReadChunk`/
  `ReadChunkAsync` no longer slice a whole-file `JsonElement[]` produced by
  `File.ReadAllText` + `JsonSerializer.Deserialize`. A new internal `JsonRecordStreamReader`
  (a persistent `Utf8JsonReader` walker that resumes mid-array across `JsonReaderState`
  reconstructions and grows its rented buffer past the 64 KB start) token-walks the file to
  locate each chunk's `[start, end)` byte range, which is then read and parsed on demand;
  schema inference and the `JsonEagerSource` ctor validation also read only the
  `SchemaInferenceRecords` sample. Memory stays bounded to one chunk, the persistent file
  handle is released once streaming reaches EOF, and backward/random chunk access reopens and
  re-walks. Chunk-level locks serialize parallel chunk reads.

### Removed

- **`MLNetInterop.ToNivaraFrame(IDataView, MLContext)` static (#233)** — removed the
  argument-order trap duplicate of the `MLNetExtensions.ToNivaraFrame(this MLContext,
  IDataView)` extension; use the extension form. The underlying
  `MLNetInterop.ConvertFromDataView` is now `internal`.

- **Array-based tensor conversions (#233)** — `TensorConversions.ReshapeToArray` and
  `TensorConversions.FlattenFromTensor(Array)` are removed in favor of the core
  `Nivara.Tensors` typed `ReshapeToTensor` / `FlattenFromTensor(Tensor<T>)` equivalents.

## [1.3.0] - 2026-08-14

### Added

- **Canonical file-source entry points (#232)** — `Json`/`Csv` each expose exactly three
  methods: `ReadFrame(path, options)` (eager `NivaraFrame`), `ScanFrame(path, options)`
  (lazy `QueryFrame`), and `ScanQuery<T>(path, options)` (lazy `NivaraQuery<T>`); the
  duplicated `ReadJson`/`ReadCsv`/`ScanJson`/`ScanCsv`/`ScanJsonAsQuery<T>`/`ScanCsvAsQuery<T>`
  variants are removed.

- **Immutable I/O options with `With()` builders (#232)** — `JsonOptions`,
  `CsvOptions`, and `ParquetWriteOptions` are now immutable with get-only properties;
  `Default` is a frozen instance and customization goes through `With(...)`
  (`JsonOptions.With` clones the `JsonSerializerOptions` to prevent aliasing).
  `CsvOptions.TrimOptions` is the `CsvTrimOptions { None, Trim }` enum and
  `ParquetWriteOptions.Compression` is the `ParquetCompression` enum (None, Snappy,
  Gzip, Lzo, Brotli, LZ4, Zstd, Lz4Raw). `ParquetWriter` now honors all three options —
  `Compression`, `RowGroupSize` (multi-row-group output), and `WriteMetadata` (gates
  `nivara.clrType.*` custom metadata). `ParquetReader` also reads **all** row groups
  (previously silently truncated to the first).

- **Metadata-aware schema equality (#234)** — `ColumnMetadata` gains
  `ClearDefaultValue()`/`ClearDescription()`/`ClearProperties()` (the previous `With()`
  could not clear values) plus `Equals`/`GetHashCode` over IsNullable, DefaultValue,
  Description, and Properties. `Schema` implements `IEquatable<Schema>` with
  `Equals`/`GetHashCode` including per-column metadata, and `IsCompatibleWith` gains an
  optional `requireMetadataMatch = false` parameter (existing name+type matching is
  unchanged by default).

- **`NivaraSeries<T>.Sum()` / `Min()` / `Max()` instance aggregates (#231)** — the series
  level reductions removed in the AutoDiff refactor are restored (issue #231 reverses that
  decision). All three dispatch through the full 17-type numeric domain via
  `NumericKernelDispatcher` (`Min`/`Max` arms added), null-aware and vectorized over
  `TensorPrimitives`, with column-parity semantics: an empty series throws, all-null `Sum`
  returns `default(T)` (zero), and all-null `Min`/`Max` throw. Non-numeric series throw a
  clear `InvalidOperationException`. `Average()` now shares a `getValidValues` helper with
  the new aggregates.

- **Virtual positional default index for `NivaraSeries<T>` (#231)** — series created
  without custom labels no longer allocate a `NivaraColumn<object>` of boxed integers.
  Label lookup (`GetByLabel`, `ContainsLabel`, `TryGetByLabel`, `GetLabel`) computes
  positional labels directly, and the public `Index` property materializes the boxed
  column lazily on first access. `Slice`/`Align`/`AlignBoth`/`Add`/`Multiply` preserve a
  virtual index for default series instead of building `object[]` arrays. Observable
  behavior is unchanged.

- **Parquet extended-domain round-trip (#190)** — `ParquetWriter`/`ParquetReader` now cover the extended CLR domain from #158. Native `DataField<T>` arms for `DateOnly`, `TimeOnly` (TIME nanoseconds), and `Guid`; `Half`, `nint`/`nuint`, `char`, `DateTimeOffset`, and `TimeSpan` widen to base Parquet types (`float`, `long`/`ulong`, `ushort`, `DateTime`, `long` ticks) with the original CLR type persisted in `CustomMetadata` under key `nivara.clrType.<column>`. Readers restore the typed column; foreign files without metadata read back as the widened type. `Int128`/`UInt128` throw a documented `UnsupportedTypeException`. Null masks and values round-trip (verified by `ParquetExtendedDomainRoundTripTests`).

- **Arrow extended-domain round-trip (#190)** — `ArrowInterop` maps the extended domain to native Apache Arrow arrays: `Half` → `HalfFloatType`, `DateOnly` → `Date32Type`, `TimeOnly` → `Time64Type` (nanoseconds), `Guid` → `FixedSizeBinaryType(16)`, `TimeSpan` → `DurationType` (nanoseconds); widened `nint`/`nuint`/`char`/`DateTimeOffset` → `Int64`/`UInt64`/`String`/`Timestamp` (µs, UTC). Original CLR types are persisted as `nivara.clrType.<column>` schema metadata and restored on read; foreign files read back as the base Arrow types. `Int128`/`UInt128` throw a documented `UnsupportedTypeException`. `DateTimeOffset` instants are clamped to the Timestamp range (1677-09-21…2262-04-11) and normalized to UTC. Note: Apache.Arrow 23.0.0 ships a native `HalfFloatArray` builder, so `Half` uses it rather than a manual `ArrayData` path.

- **ML.NET faithful `ToNivaraFrame` (#190)** — `MLNetInterop.ToNivaraFrame` reads every primitive DataView type without coercion: all numeric kinds, `bool`, `string` (via `ReadOnlyMemory<char>` getters), `DateTime`, `DateTimeOffset`, and key columns (as `uint`). Columns that previously failed type extraction were silently dropped; vector columns of non-single precision now throw a clear `NotSupportedException` (single-precision vector columns keep the first-element contract used for Score).

- **Optimizer API consistency (#181)** — `Optimizer<T>.LearningRate` is now settable: assigning forwards to every parameter group created without an explicit learning-rate override (tracked internally), while groups created with an explicit override — or later managed via `SetGroupLearningRate` — keep their own rate. `SetGroupWeightDecay` mirrors `SetGroupLearningRate` for weight decay, and `ParameterGroup.LearningRate`/`WeightDecay` are now public-read/internal-write so consumers observe but cannot corrupt optimizer state. `Adam<T>.AdamUpdate` and `AdamW<T>.AdamWUpdate` gain public static functional single-tensor entries mirroring `SGD<T>.SgdUpdate`: they take caller-owned `expAvg`/`expAvgSq` buffers plus a 1-based `step`, mutate the buffers in place, and return a new `requiresGrad=false` tensor (the shared span kernels were extracted from the instance `Step()` so both paths run identical math). New tests pin the base-vs-group LR equivalence, weight-decay mutation, Adam/AdamW functional parity with the instance path, and step-without-`ZeroGrad` gradient accumulation.

- **Loss API unification: `Loss<T>` base + `Reduction` enum (#180)** — every loss in `Nivara.AutoDiff.Nn.Functional` (`MSELoss`, `L1Loss`, `BCELoss`, `BCEWithLogitsLoss`, `CrossEntropyLoss`) now inherits the abstract `Loss<T>` base, which stores a ctor-defaulted `Reduction` (`Sum`, `Mean`, `None`) and exposes `Forward(predictions, targets)` plus a per-call `Forward(predictions, targets, Reduction)` override. The default is `Reduction.Mean` everywhere (PyTorch parity). Shared `Loss<T>.Reduce` centralizes reduction: `None` returns the elementwise loss, `Sum` uses `ReverseGradOperations.Sum`, `Mean` divides the sum by a divisor (element count, or batch size for CrossEntropyLoss). `CrossEntropyLoss` also gains `Reduction.None` per-sample NLL (`[N]`, class-weighted sum within each row) matching PyTorch `reduction='none'`. `BCELoss.eps` stays a ctor arg. PyTorch-validated `reduction='none'` fixtures and NivaraTorch tests added. The misgrouped `Softmax<T>`/`LogSoftmax<T>` classes moved into `Activation.Softmax(input, dim)` / `Activation.LogSoftmax(input, dim)` wrappers and were deleted.

- **Window-function expressions in the expression DSL (#159)** — rolling / cumulative / shift / lead / rank windows are now first-class `ColumnExpression`s (`WindowExpression` + `ColumnExpressions` factories `RollingSum`/`RollingMean`/`RollingMin`/`RollingMax`, `CumulativeSum`/`Max`/`Min`/`Product`/`Count`, `Shift`/`Lead`, `RowNumber`/`Rank`/`DenseRank`/`PercentRank`). A window expression can be embedded in `Select`/`Filter`/`SortBy` and composed with elementwise math (e.g. `Select(RollingSum(Col("Salary"), 2) * 2)`); the fused evaluator rewrites window nodes bottom-up via the existing kernels and injects synthetic columns, so nested windows compose and a standalone window stays a single materialization. Window ops in the lazy pipeline accept computed sources and keys: `RollingSum(Col("A") * 2, "r", 2)`, `CumulativeSum(expr, ...)`, `Shift(expr, ...)`, `Lead(expr, ...)`, and `Rank(resultColumn, orderBy: [SortExpressionKey(Col("B") * 1)], partitionBy: [Col("Dept")])` — `RollingOperation`/`CumulativeOperation`/`ShiftOperation` gain an optional `SourceExpression` (`Source` is now `string?`) and `RankOperation` an expression-key constructor, with schema/result types (`WindowFunctionHelpers.GetResultType`) shared between the AST and the ops. The plan layer routes `OperationType.Rolling`/`.Cumulative`/`.Shift`/`.Rank` to a new `VisitWindow` hook in both `QueryPlanVisitorBase` and `QueryPlanTransformerBase<T>` (previously "unknown"), and `QueryPlan.GetOperationDetails` describes window/rank operations in `GenerateDiagnosticInfo`. Note: the pre-existing ambiguous `QueryFrame.RowNumber(string)` overload was removed so `RowNumber(expr, resultColumn, orderBy, partitionBy)` is unambiguous.

- **Typed LINQ object model `frame.Query<T>()` (#130)** — an ergonomic typed layer over the expression engine. `NivaraFrame.Query<T>()` (requires `T : class, new()`) returns an immutable `NivaraQuery<T>` supporting `Where`/`Select`/`OrderBy`/`OrderByDescending`/`ThenBy`/`ThenByDescending`/`Skip`/`Take`/`Slice`/`GroupBy`/`Collect`/`ToObjects`/`ToList`/`ToRows`. Predicates and projections are `Expression<Func<...>>` lambdas translated at build time to `ColumnExpression` by `TypedExpressionTranslator` (property access, literals, arithmetic, comparisons, `&&`/`||`/`!`); method calls, closures, nested access, and ternary fail fast with `UnsupportedQueryExpressionException`. `GroupBy` accepts aggregate `Select` (`g.Key`, `g.Average/Sum/Count/Min/Max`) via `Grouping<TKey,T>` or a bare `Collect` of distinct keys. `Collect()`/`ToList()` return a `NivaraFrame`; `ToObjects()`/`ToRows()` materialize `IReadOnlyList<TResult>` through a compiled, cached per-type row factory (`TypedRowFactory`). Also fixes the expression engine's `And`/`Or` evaluation to produce boolean columns with SQL-like null masking instead of object-typed columns. Known limitation: `p.City == null` comparisons fail at `Collect` time (literal coercion only supports non-null constants); null-check semantics (`IsNull`/`IsNotNull`) are a follow-up.
- **Window functions: rolling / cumulative / shift / lead (#135)** — delivered at three layers with a single per-aggregate method shape. Column primitives (`src/Nivara/Tensors/WindowFunctions.cs`) add `RollingSum/Mean/Min/Max`, `CumulativeSum/Max/Min/Product/Count`, `Shift`, and `Lead` extensions on `NivaraColumn<T>` with explicit null-mask semantics: rolling output is null until the window holds `minPeriods` valid values (default full window); cumulative ops skip nulls with carry-forward; `Shift`/`Lead` move in nulls (or `fillValue`) at boundaries; an optional `nullHandler` replaces each null so it participates and every position satisfies the window. Eager `NivaraFrame` extensions (`src/Nivara/WindowFrameExtensions.cs`) and lazy `QueryFrame` members expose the identical shape, appending a result column while preserving all inputs. In the query pipeline the ops run as `OperationType.Rolling` / `.Cumulative` / `.Shift` (`src/Nivara/Operations/WindowOperations.cs`, `WindowNode`), and are marked non-parallelizable and non-streamable.
- **Rank-family window functions: row_number / rank / dense_rank / percent_rank (#156)** — SQL `OVER (PARTITION BY ... ORDER BY ...)` semantics at the same three layers. Column primitives (`src/Nivara/Tensors/RankFunctions.cs`) drive a shared `RankKernel` that partitions via grouping, orders with `SortKey` direction/null ordering, and emits `RowNumber`/`Rank`/`DenseRank` as `long` and `PercentRank` as `double`. Eager `NivaraFrame` extensions (`src/Nivara/WindowFrameExtensions.cs`) and lazy `QueryFrame` members (`RowNumber`/`Rank`/`DenseRank`/`PercentRank`) expose the same shape; `Rank`/`DenseRank`/`PercentRank` require at least one order key while `RowNumber` allows none. A null order key yields null output for that row and it is excluded from numbering and the percent-rank denominator. In the pipeline the ops run as `OperationType.Rank` (`src/Nivara/Operations/RankOperation.cs`) and are non-parallelizable/non-streamable.
- **Row-wise frame scoring (`NivaraFrame.RowDot` / `RowCosineSimilarity`, #138/#141/#142)** — a scoped tensor interop convenience: each row of a frame is scored against a `NivaraSeries<T>` query vector. `TensorsHelper` gains internal row-slice `TensorPrimitives` kernels (`RowDot`, `RowCosineSimilarity`, `RowNorms`, `ValidateRowKernelArgs`, `AnyTrue`) over a row-major buffer + null mask; the public frame methods materialize row-major through a pooled blocked transpose and return a `NivaraSeries<T>`. SQL-like null semantics: a null in a row masks only that row's score, a null in the query masks all scores, and the result always carries a null mask. `Nivara.PerformanceTests` gains four row-scoring scenarios (per-row status quo, frame API, raw kernels) as the regression gate; the frame API runs ~2.5× faster than the per-row status quo on the 10k × 128 benchmark. No public `RowNorms`/`ColumnNorms`/`Dot`/`CosineSimilarity` were re-added — the removed tensor-axis APIs stay removed (see 1.2.0).
- **`NivaraFrameExtensions.Standardize` (z-score alias, #143)** — data-prep promoted from `Nivara.MLNet` into core frame extensions (`src/Nivara/NivaraFrameExtensions.cs`). `Normalize`/`Standardize` now use `TensorPrimitives` (`Average`/`StdDev`/`Subtract`/`Divide`) for SIMD statistics and transform, compute mean/stddev over non-null values only, and preserve the null mask in the result (`CreateFromSpans`). Auto-select (`Normalize()`/`Standardize()` with no arguments) now normalizes all float/double columns instead of returning an unchanged frame (a latent bug in the old `??=` fallback). `IsNumericColumn` narrowed to float/double; explicitly naming an unsupported column throws `NotSupportedException`.
- **`Normalize`/`Standardize` full `INumber<T>` surface (#144)** — supersedes the float/double-only dispatch from #143. Support is now interface-based: a schema type is normalized when it implements `INumber<>` and is not in the explicit blocklist (`char`, `BigInteger`, `Int128`, `UInt128`). `int`, `long`, `short`, `byte`, `uint`, `ushort`, `sbyte`, `nint`, `nuint`, and `decimal` columns are now z-scored too, with the result promoted to `NivaraColumn<double>` (`TensorPrimitives.ConvertChecked<T,double>` → `Average<double>`/`StdDev<double>` → `Subtract`/`Divide`); `float`/`double`/`Half` keep the in-place SIMD path (`TensorsHelper.TryNormalizeInPlace`). `NormalizeColumn` dispatches through a per-type cached compiled delegate (`ConcurrentDictionary<Type, Func<...>>` + `MakeGenericMethod` + `Expression.Lambda`), so the interface predicate runs once per column type instead of per call. Auto-select now normalizes every supported numeric column; explicitly naming an unsupported column still throws `NotSupportedException`. Null-skip statistics and zero-variance-unchanged semantics carry over unchanged.
- **Mixed-type numerics use the typed promoted path in expression evaluation** — the fused evaluator no longer falls back to a boxed `Convert.ToDouble` path for mixed numeric operands (`double + int`, `decimal + int`, `byte + int`, `Col("A") + 1`, `Col("A") > 5`). Operand pairs are widened to the C# binary-numeric-promotion common type (`NumericPromoter.GetPromotedType`) and the operation runs through the compiled typed kernel with null-OR propagation, producing a typed `NivaraColumn<TResult>` result instead of `NivaraColumn<object?>`. C#-rejected pairs (`ulong` + signed, `decimal` + float/double) resolve to `double`, matching the previous boxed behavior; non-numeric and non-promotable pairs (Guid, string, etc.) are rejected with `NotSupportedException` (the legacy boxed fallback was removed). Integer division remains integral for integral results, matching the same-type typed path.
- **`NivaraRow` typed row view (#154)** — a public readonly struct passed to `NivaraFrame.Where(Func<NivaraRow, bool>)` predicates. Allocation-free over the frame's columns: `GetValue<T>` / `TryGetValue<T>` / `IsNull` / indexer / `RowIndex`, with case-insensitive name lookup, `ColumnNotFoundException` / `ColumnTypeMismatchException` on bad access, and a clear `InvalidOperationException` from the `default` state.
- **Modulo (`%`) arithmetic (#152)** — added to the expression DSL (`ColumnExpression` binary + scalar `%` operators, `BinaryOperator.Modulo`) and the typed LINQ translator (`p.Age % 2`). Runs through the fused compiled evaluator (`Expression.Modulo`) and the generic node-tree kernel with C# numeric promotion and null-OR mask propagation; `byte + byte` produces a `NivaraColumn<int>` like the rest of the DSL. `docs/LINQ.md` updated — `%` is now supported, not fail-fast.
- **Dim-aware `Softmax`/`LogSoftmax` (#179)** — the `dim` parameter (default `-1` = true last dim) is now honored across arbitrary axes via strided kernels (`GradKernels.SoftmaxDim`/`LogSoftmaxDim`/`SoftmaxDimGradient`/`LogSoftmaxDimGradient`) dispatched from `ReverseGradOperations.Softmax`/`LogSoftmax` and the `Activation.Softmax`/`Activation.LogSoftmax` wrappers. Negative dims are normalized against the input rank; out-of-range dims throw `ArgumentOutOfRangeException`; layout mismatches throw `ArgumentException`. Rank-2 last-dim behavior is unchanged, so existing callers (CrossEntropyLoss, samples) are unaffected.
- **`ReverseGradTensor<T>.ToHalf()` / `TypeConverter.ToHalf<T>` (#179)** — completes the conversion surface so `Half` is fully served alongside `float`/`double`. `Half` SIMD fast paths added to `RMSNormKernel` (per-row forward/backward), `Adam`, and `AdamW` via `TensorPrimitives` chains over `MemoryMarshal.Cast<T, Half>` views. Stale `INumber<T>` doc comments corrected to `IFloatingPointIeee754<T>`.
- **`GradientUtils.CanBackward(tensor, gradient)` overload (#179)** — checks `RequiresGrad`, matching length, and matching shape for `Backward(gradient)` calls (the seed-less scalar helper is unchanged). `DescribeTensor` now reports `Can Backward (no seed)`.
- **`NivaraColumn<T>` arithmetic generic-math collapse (#157)** — the six `NivaraColumn<T>` arithmetic kernel helpers (scalar `Multiply`/`Divide`, column `Multiply`/`Add`/`Subtract`/`Divide`) now dispatch `decimal`, `Half`, `nint`, `nuint`, `Int128`, and `UInt128` through the `INumber<T>`-constrained `NumericTensorKernels<T>` typed switch, matching `NivaraSeries`. These types previously threw (`InvalidOperationException` for `Half`/`nint`/`nuint`/`Int128`/`UInt128` via `validateTypeSupportsOperation`, `NotSupportedException` for `decimal` at kernel dispatch). On .NET 10 `TensorPrimitives` runs the six types via SIMD (`Half` widening, `nint`/`nuint`) or the operator-based software fallback (`decimal`/`Int128`/`UInt128`). `IsNumericType()` recognizes the five previously-rejected types so validation no longer blocks them; non-numeric types (`string`/`Guid`/`DateTime`) still throw the clear validation error. `KernelSelector` still reports `KernelType.Scalar` for the six, so diagnostics stay accurate.

### Changed

- **`NivaraSeries<T>.TopKDescending` stringifies labels (#231)** — non-string labels were
  silently nulled (`label is string s ? s : null`), so a default positional index returned
  null labels and int/DateTime custom labels were dropped. Labels are now stringified via
  `ToString()`, surfacing positional indices as their integer string form (e.g. `"0"`,
  `"1"`) and preserving every label as a useful string. Return type
  `(string? Label, T Score)[]` is unchanged.

- **ML.NET float conversion is no longer silently lossy (#190)** — `MLNetInterop.ConvertToFloat` (used by `ToDataView`, `ToFeatureVectors`, `CreateFeatureMatrix`) throws `InvalidOperationException` for non-numeric values (string, bool, DateTime, Guid, …) instead of returning `0f`. Extended numeric types (`uint`, `ulong`, `ushort`, `sbyte`, `nint`, `nuint`, `Half`) are now converted. `null` still maps to `0f` per the ML feature-vector contract.

- **Removed `NivaraColumn<T>.CreateFromNullable(Array)` (breaking, #222)** — the generic-class Array overload is deleted; `NivaraColumn.CreateFromNullable<T>(T?[])` is the single entry point for nullable value-type columns (all internal dispatch and every call site now use it). Migration: `NivaraColumn<T>.CreateFromNullable(values)` becomes `NivaraColumn.CreateFromNullable(values)` — the factory resolves `T` by inference; use an explicit type argument for `null` arrays (`NivaraColumn.CreateFromNullable<int>(null!)`). Reference-type arguments are now rejected at compile time by the `where T : struct` constraint instead of a runtime `InvalidOperationException`.

### Fixed

- **Dynamic column creation covers the extended CLR domain (#158)** - the five
  dynamic column-creation dispatch sites used fixed type switches that fell through to a
  `NivaraColumn<object>` for less common types: `AggregationFunction.CreateColumnFromValues`
  and `GroupByOperation.CreateColumnFromValues` missed `Half`, `nint`/`nuint`,
  `Int128`/`UInt128`, `sbyte`/`ushort`/`uint`/`char`, `DateOnly`/`TimeOnly`,
  `DateTimeOffset`, `Guid`, and `TimeSpan`; `JoinOperation` coalesce/gather and
  `FusedExpressionEvaluator.CreateConstantColumn` had the same gap. A new `ColumnFactory`
  (`src/Nivara/Helpers/ColumnFactory.cs`) centralizes dispatch behind a cached
  `MakeGenericMethod` over null-safe kernels (`CreateFromNullable` for value types,
  `CreateForReferenceType` for reference types, `Nullable<T>` unwrapping) and is used by all
  four sites; join coalesce/gather dispatch directly onto the existing generic kernels, and
  the object-column fallbacks were removed. The existing `Cast<T>()`-based creation also
  threw on null values - the new kernel is null-mask safe. Window operations
  (`WindowFrameExtensions` rolling/cumulative/count/shift) now accept the full `INumber<T>`
  numeric domain (`byte`..`Half`, `nint`/`nuint`, `Int128`/`UInt128`, `char`) instead of
  throwing `NotSupportedException`, and `adaptNullHandler`/`convertFillValue` no longer use
  `Convert.ChangeType` (which throws for `Half`/`nint`/`nuint`/`Int128`/`UInt128`); typed
  fill/null values use a direct cast and string values use a cached `TryParse`. Out of scope:
  Parquet/Arrow/ML interop and CSV/JSON value conversion keep their format-specific type
  systems.

- **`NivaraResourceManager` tracking is now opt-in (#174)** — column/frame/QueryFrame
  construction no longer registers a `WeakReference` + boxed `ResourceInfo` in a global
  `ConcurrentDictionary`, and the process-lifetime 30-second cleanup `Timer` is gone from
  default hosts. Tracking is disabled by default (performance-first); hosts that want
  resource diagnostics opt in via `NivaraResourceManager.Enable()` (internal), which lazily
  creates the timer. `TrackResource` / `UntrackResource` / the timer callback are guarded
  no-ops when disabled, and the column ctor only computes `estimateMemoryUsage()` inside the
  enabled branch. Public surface (`MemoryRecommendations`, `ResourceStatistics`,
  `NivaraFrame.GetMemoryRecommendations`) is unchanged. Behavioral note: `QueryFrame`
  abandoned-lazy-source cleanup (its `CleanupAction`) is now opt-in too.

- **Same-type small-integral promotion in `NumericPromoter` (#152)** — `GetPromotedType` returned the operand type for equal operand pairs, so `byte + byte` produced `byte` instead of the C# spec §12.4.7.3 rule 1 result `int`. Same-type `sbyte`/`byte`/`short`/`ushort`/`char` pairs now promote to `int`; other same-type pairs (`decimal`, `uint`, `float`, `double`, `Half`, …) keep their type. This flows through `ExpressionTypeInferer` plan types and the compiled kernel target, fixing schema/result divergence for small-integral expressions.

- **Row-major hot loops no longer re-evaluate `NivaraFrame.RowCount`** (`columns.Values.FirstOrDefault()` LINQ allocates ~40 B per access): `CopyToRowMajor`, `ToNullableTensor`, and the new `materializeRowMajor` cache `RowCount`/`ColumnCount` in locals. On a 10k × 128 frame this was ~51 MB/op of pure garbage; the fix drops `Frame RowDot` allocation to ~452 KB/op (dominated by result-series construction, not the kernel).

- **`Sum`/`Mean` group-by aggregation for the full numeric domain (#169)** — `SumAggregation` and `MeanAggregation` only handled `int`/`byte`/`short`/`long`/`float`/`double`/`decimal`; `uint`/`ushort`/`sbyte`/`ulong`/`char`/`bool` passed validation then threw in `Apply`, and `Half`/`nint`/`nuint`/`Int128`/`UInt128` were rejected at validation. Both now accept the full 17-type `GetNumericTypes()` domain plus `bool`. Sum promotes per `NivaraSeries` rules (small integrals/`char`/`bool` → `long`, `ulong` → `ulong`, `nint` → `Int128`, `nuint` → `UInt128`, `Int128` → `Int128`, `UInt128` → `UInt128`, `float`/`Half` → `double`, `decimal` → `decimal`); widening now uses typed `TResult.CreateChecked` instead of `Convert.ChangeType` (which throws for `Half`, which has no `IConvertible`). Mean converts widened sums to `double` through a typed `ToDouble` switch (boxed `Int128`/`UInt128` are not `IConvertible`). Group-by sums produce typed result columns for `ulong`/`Int128`/`UInt128`.

- **`NivaraSeries<T>.Average()` for the extended numeric domain (#172)** — `divideByCount` only handled 12 of the 17 types the sum dispatch supports, so `NivaraSeries<Half/nint/nuint/Int128/UInt128>.Average()` computed a SIMD sum then threw `NotSupportedException`; the public `Average()` guard also rejected those types via `IsNumericType()` before the kernel path ran. `divideByCount` gains the 5 missing arms (same-type truncating division, matching the existing integral arms) and the guard accepts the full `GetNumericTypes()` domain (bool remains rejected by the sum dispatch).

- **Frame `Take`/`Skip`/`Slice` no longer slice columns via reflection (#173)** — `NivaraFrame.sliceColumn` called `GetMethod("Slice", ...)` + `MethodInfo.Invoke` on every column for every `Take`/`Skip`/`Slice` (an `object[]` boxing allocation plus a dictionary lookup per column per call). It now calls the `IColumn.Slice(int, int)` interface method directly; the unreachable `ColumnFilterHelper.CreateFilteredColumn` fallback was deleted. The query engine's `SliceOperation.SliceColumn` had the identical reflection pattern and was fixed the same way. A `Frame Slice [10k x 128]` scenario was added to `Nivara.PerformanceTests` so the removed allocations are measurable.
- **BatchNorm running-stats NRE fixed (#179)** — `BatchNorm1d<T>`/`BatchNorm2d<T>.RunningMean`/`RunningVar`/`NumBatchesTracked` threw `NullReferenceException` (via `!`) when the module was created with `trackRunningStats: false`. They now throw a clear `InvalidOperationException` explaining the constructor option, and a new `TrackRunningStats` property exposes the flag. `StateDict`/`LoadStateDict` are unaffected.
- **`BatchBackward` now honors its tensor list (#179)** — `NivaraAutoGradExtensions.BatchBackward(tensors, loss)` previously ignored `tensors` and only ran `loss.Backward()`. It now verifies every listed requires-grad tensor received a gradient after backward and throws `InvalidOperationException` listing the offending keys (constants, i.e. `RequiresGrad == false`, are exempt). `ToGradientFrame` xmldoc clarifies its intentional asymmetry vs `ToFrame` (gradient columns are skipped when null).

### Documentation

- **XML doc comments across the entire AutoDiff public API (#197)** - every public type and member under `src/Nivara/AutoDiff/` (NN modules, losses, optimizers, training, serialization, initializers, operations, tensors) now carries XML doc comments (`<summary>` plus `<param>`/`<returns>`/`<exception>`/`<see cref>` where applicable), so IntelliSense tooltips cover the full ML training surface. `docs/REVIEW-2026-08-12.md` finding #1 (High) is now resolved; the build gate `dotnet build src/Nivara/Nivara.csproj -p:GenerateDocumentationFile=true` reports zero CS1591 across the namespace. Pure documentation change - no API shape, behavior, or signature impact.

### Breaking changes

- **`NivaraSeries<T>` object label indexer removed; `this[string]` added (#231)** — the
  `this[object]` indexer collided with positional `this[int]` access for boxed integer
  labels. It is replaced by `this[string]`, so the common `series["a"]` lookup keeps
  working; integer and other non-string labels must use the explicit `GetByLabel(object)`
  API (boxed `(object)` casts no longer resolve). Series created without custom labels use
  a virtual positional index (`Index` lazily materialized); behavior is otherwise
  unchanged.

- **`Module<T>.Forward(input1, input2)` removed; multi-input forward is opt-in via `IMultipleInputModule<T>` (#202)** — the base class no longer advertises a two-input `Forward` that always threw `NotSupportedException` on every subclass except `MultiheadAttention`. Only `MultiheadAttention<T>` and `VAE<T>` genuinely accept a second input, so the capability moved to a new `IMultipleInputModule<T>` interface (`Forward(input1, input2)`) implemented by those two modules. Consumers holding a `Module<T>` reference dispatch with `if (module is IMultipleInputModule<T> multiInput) { multiInput.Forward(a, b); }`. The removed member only ever worked through MHA's concrete type, so the breaking surface is contained to `Module<T>`-typed two-argument calls (which previously threw or required a concrete cast).

- **Legacy `ExpressionEvaluator` removed (#152)** — the per-operator boxed evaluator (`src/Nivara/Helpers/ExpressionEvaluator.cs`) and its tests are deleted. Every production query op (`FilterOperation`, `SelectOperation`, `SortByExpressionOperation`) and `ParallelExecutionStrategy` already routed through the fused evaluator (`FusedExpressionEvaluator` + `FusedKernel` + `ExpressionTypeInferer`), which is now the sole engine; unsupported operand combinations throw `NotSupportedException` instead of silently falling back to boxed evaluation. The performance benchmark's fused-vs-multi-pass comparison scenario was dropped.
- **`MLNetExtensions.Normalize` removed from `Nivara.MLNet`** — moved to core `NivaraFrameExtensions.Normalize`/`Standardize` (same signature, namespace `Nivara`). Update `using` if you relied on `Nivara.MLNet` for this helper.
- **`NivaraFrame.Where(Func<dynamic, bool>)` removed (#154)** — the last public `dynamic` surface in the core library. It built an `ExpandoObject` per row plus a reflection `Item` lookup per element (`CreateDynamicRow`), both deleted. The overload is now `Where(Func<NivaraRow, bool>)`; a predicate that boxed `dynamic` member access (e.g. `row => row.Age > 25`) must switch to typed accessors (`row => row.GetValue<int>("Age") > 25`). Predicate exceptions now propagate unwrapped instead of being rethrown as `InvalidOperationException`.
- **AutoDiff weight access unified on `Parameter<T>?` `Weight`/`Bias` (#177)** — every leaf module that owns learnable weights now exposes `Weight`/`Bias` of type `Parameter<T>?` (null = parameter omitted via `bias: false` / `affine: false`), matching PyTorch's `module.weight`/`module.bias`. `Linear<T>`, `Embedding<T>`, `SparseEmbedding<T>` lose their `WeightParam` accessor and their tensor-typed `Weight` member; `Conv1d`/`Conv2d`/`ConvTranspose2d` lose `WeightParam`/`BiasParam`. Consumers reach the tensor via `Weight!.Tensor` / `Bias!.Tensor`; `GetParameters()` / `StateDict()` keys are unchanged.
- **AutoDiff dead public surface removed (#178)** — `GradKernels` and `ComputationGraph` are now `internal` (public graph introspection stays on `GradientUtils`: `ZeroGrad`/`GetGraphInfo`/`PrintGraphSummary`/`DescribeTensor`); the four never-thrown sealed exception types (`GradientComputationException`, `CircularDependencyException`, `InvalidBackwardCallException`, `TypeValidationException`) were deleted, keeping only `AutoGradException` + `ShapeIncompatibilityException`; and the legacy static initializers (`KaimingNormal.Init<T>(Dictionary<...>)` and siblings) plus `DefaultInitializers` were deleted — the `IInitializer<T>` instance API (`KaimingUniformInitializer<T>`, etc., passed via `weightInitializer:`/`biasInitializer:`) is the only initializer surface. `Backward()` keeps its deliberate `InvalidOperationException`/`ArgumentException` contract.
- **Loss reduction defaults to `Reduction.Mean`; `reduceToMean` bool overloads removed (#180)** — `MSELoss`/`L1Loss`/`BCELoss`/`BCEWithLogitsLoss` previously defaulted to the sum of elementwise losses, and MSE/BCEWithLogits exposed `Forward(..., bool reduceToMean)`. The bool overloads are removed; the default is now `Mean` (PyTorch parity) via the `Loss<T>` base. Callers that relied on sum scaling should construct with `Reduction.Sum`. `CrossEntropyLoss` reduction is unchanged (mean by batch). The `Softmax<T>`/`LogSoftmax<T>` module classes are deleted — use `Activation.Softmax`/`Activation.LogSoftmax`.

## [1.2.0] - 2026-08-05

### Breaking changes

- **`NivaraFrame.Dot<T>` / `CosineSimilarity<T>` / `ColumnNorms<T>` / `RowNorms<T>` removed** (AutoDiff refactor, Task 10): the four deprecated frame tensor-axis methods are deleted rather than relocated — they had no production callers. Use `TensorPrimitives.Dot` / `TensorPrimitives.CosineSimilarity` / `TensorPrimitives.Norm` on column spans (via `TryGetSpan`) or on row-major spans assembled through `CopyToRowMajor`. The `TensorsHelper.RowNorms` kernel (only consumer was `frame.RowNorms`) was removed with them.
- **`NivaraSeries<T>.Sum()` / `Min()` / `Max()` removed** (AutoDiff refactor, Task 9): NivaraSeries is now a labeled-column wrapper and keeps only `Average()`. Use the null-aware column reductions `NivaraColumn<T>` extensions `Sum` / `Min` / `Max` (`Nivara.Tensors`, `INumber<T>`-constrained) via `series.Values`; empty-column `Sum` throws, all-null `Sum` returns `T.Zero`, all-null `Min`/`Max` throw. Non-numeric (string/object) Min/Max/Sum are no longer supported.
- **`NivaraTensorExtensions` stripped to column reductions** (AutoDiff refactor, Task 8): the column-level activations/gradients/MatMul/Transpose/GELU family extension methods were deleted (they now live in `GradKernels` as span kernels) along with the obsolete Series extensions (`AddTensor`, `MultiplyTensor`, `SumTensor`, `DotProduct`, `Norm`, `TransformTensor`) and `MatrixMultiply`. Remaining members: `Sum`, `Mean`, `Min`, `Max`. `NivaraColumn<T>.Subtract(NivaraColumn<T>)` / `Divide(NivaraColumn<T>)` / `Divide(T)` were promoted from extensions to first-class members.
- **`TextClassifierModel<T>` / `TokenClassifierModel<T>` moved out of core** — the two pre-built NLP classification modules now live in `samples/Nivara.Samples` (`TextClassifierModel.cs`, `TokenClassifierModel.cs`). `TextTokenizer` remains in core (`src/Nivara/AutoDiff/Nn/TextTokenizer.cs`). Samples and docs were updated to reference the new home.
- **Model/checkpoint serialization format bumped to `nivara-ss-v2` / `nivara-ckpt-v2`** (AutoDiff, ADR-001): the null-mask persistence (`HasNulls` / `NullMask` on parameter entries, `ParameterData<T>.NullMask`) was removed from the AutoDiff non-nullable domain. Deserialize now uses the zero-copy `CreateFromOwnedArray` path. v1 files are rejected loudly with an "unsupported format" error instead of being silently misread.
- `ArrowConversionOptions.UseZeroCopy` removed — the option defaulted to `true` but every zero-copy interop path was a placeholder that silently copied. Nivara does not advertise unsupported capability; real zero-copy returns with ARROW-ROADMAP Phase D (adding real APIs then).

### Storage Consolidation

- `Nivara.Storage.MemoryStorage<T>` renamed to `Nivara.Storage.ColumnStorage<T>` and moved to sole-owner contiguous `T[]` backing with an optional `bool[]` null mask (`null` mask ⇒ non-nullable column). `Data`/`NullMaskMemory`/`AsSpan()`/`TryGetSpan`/`Slice` keep their zero-copy, shared-buffer semantics.
- New internal lazy `ColumnStorage<T>.AsTensor()` returns a zero-copy `Tensor<T>` view over the storage's backing array (unmanaged `T` only — `Half`/`BFloat16` pass; reference-containing types throw). Slices are supported via `Tensor.Create(array, start, lengths, strides)`.
- `ColumnStorageFactory` now builds `ColumnStorage<T>` directly for every type — vectorizable primitives no longer route to `TensorStorage<T>`. The tensor/memory split helpers (`createTensorStorage`, `CreateTensorStorageForType`, `CreateTensorStorageForOwnedArray`, `CreateTensorStorageForNullableType`) and the duplicate `IsUnmanagedType<T>()` type list were deleted; the runtime unmanaged guard lives on `ColumnStorage<T>.AsTensor()` via `RuntimeHelpers.IsReferenceOrContainsReferences<T>()`. `IsVectorizable<T>()` is retained for `KernelSelector` heuristics.
- `Nivara.Storage.TensorStorage<T>` deleted and `StorageType`/`StorageType`-based dispatch removed from the storage contract (`IColumnStorage<T>`), `ColumnDiagnostics`, and `NivaraColumn`. All storage is the single `ColumnStorage<T>`; span access is always a genuine zero-copy view (`ProvidesZeroCopySpanAccess` dropped), and the `NivaraColumn` vectorized scalar kernels now operate directly on the storage's zero-copy span instead of pooling + copying the tensor-backed buffers. The scalar-comparison dead branches that threw for unsupported combinations were removed along with the tensor path.
- Storage consolidation onto a single `ColumnStorage<T>` is **complete**: `NivaraColumn` dispatch path collapse, AutoDiff boundary hardening (runtime ADR-001 throws), and the benchmark gate all landed. Before/after results (baseline vs post-consolidation) are captured in `tests/Nivara.PerformanceTests/README.md`.
- **AutoDiff boundary (ADR-001) enforced at runtime**: `ReverseGradTensor`/`ForwardGradTensor` constructors now throw `AutoGradException` (message contains "ADR-001") when the input column `HasNulls` (previously only a stripped-in-Release `Debug.Assert`); `ForwardGradTensor` tangent columns are guarded identically.
- **AutoDiff enter path is zero-copy**: `FromColumn`/`FromSeries` wrap the column without copying; `FromArray`/`FromMatrix` now wrap the caller's array via `CreateFromOwnedArray` — **breaking contract change**, callers must not mutate the source array afterward. `GradTensor.AsTensor()` returns a zero-copy `ColumnStorage<T>.AsTensor()` view sharing the backing array instead of a flattened copy; `NivaraColumn.AsTensorView()` backs it. `ModuleHelpers.GetSpan` fallback copy removed (`TryGetSpan` now always succeeds for AutoDiff tensors).
- **AutoDiff initializers and `TensorDataset<T>` enter the graph zero-copy** (Task 11): all 13 initializer implementations wrap freshly allocated weight arrays with `NivaraColumn<T>.CreateFromOwnedArray` instead of copying through `Create`; `TensorDataset<T>.GetBatch` now slices column spans via `TryGetSpan` and throws ADR-001 when a source column contains nulls (previously the null-mask path always threw at the tensor constructor, so behavior is unchanged).

### AutoDiff (GradKernels & inference fast paths)

- **`GradKernels<T>` span-kernel layer** (ADR-002, Tasks 1–6): all `ReverseGradOperations` and `ForwardGradOperations` now delegate to shared `GradKernels<T>` span kernels (`Span<T>`/`ReadOnlySpan<T>` + `TensorPrimitives`), replacing per-op duplicated column math and eliminating `NivaraColumn.Data` access. Results wrap once via `NivaraColumn<T>.CreateFromOwnedArray` (no copy). ADR-002 records the span boundary as the canonical AutoDiff architecture.
- **Inference-only fast paths**: `Gelu`, `GeluExact`, `LayerNorm` run single-path inference kernels that never construct graph nodes outside `GradientUtils.Grad()` (verified by `InferenceGraphTests`/`InferenceFastPathTests`); conv bias tracking is gated on `Grad()` scope so inference builds no graph. AutoDiff diagnostics are gated behind a static toggle for zero-cost inference.
- **Linear inference & transposed-weight cache** (#87): forward inference passes the raw weight to the kernel's `MatMulTransposedB` path (zero transposes); training reuses a version-stamped transposed-weight cache invalidated only on `Parameter<T>.Version` change.
- **New ops**: `AddBias` row-broadcast (Linear bias), `MatMulTransposedB` (transposed-B matmul), `GeluExact` (exact erf GELU for BERT-family activations, SIMD `TensorPrimitives`), `BatchedMultiHeadAttention` — fused `[B, L, D]` batch attention with per-batch additive `[B, qLen, kvLen]` masks, single `OpNode` VJP producing dQ/dK/dV (PyTorch-parity fixtures + perf scenarios).
- **BCL-tuned MatMul kernels**: `MultiplyCore` optimized against `TensorPrimitives.Dot` (BCL swap-target annotations in `TensorsHelper`); rank-2 backward transpose buffers now pooled via `ArrayPool<T>.Shared` instead of per-call allocations.
- **Enter path is zero-copy** (Task 11): all 13 initializers wrap freshly allocated weight arrays with `CreateFromOwnedArray`; `TensorDataset<T>.GetBatch` slices column spans via `TryGetSpan` and throws ADR-001 on null-containing columns.

### Training & Serialization

- **`Optimizer<T>.StateDict()` / `LoadStateDict()`** — optimizers now expose their moment/velocity buffers for incremental-training scenarios (matching the module `StateDict`/`LoadStateDict` contract).
- **Optimizer state persisted in checkpoints**: `Checkpoint<T>.OptimizerState` added and `ModelSerializer.SaveCheckpoint`/`LoadCheckpoint` now round-trip optimizer state alongside model parameters, so a checkpoint is a full training resume point.
- **Epoch-aware `DataLoader<T>.GetBatches(epoch, skipBatches)`** — yields a single epoch's batches with skip support, enabling incremental/online training loops (`NivaraChat --online-learning` uses it).

### Fixed

- **Owned-array contract documented on remaining factory surfaces** (#106): `Parameter(string, T[], bool)` and `GradientUtils.Constant(T[])` wrap caller arrays zero-copy; XML docs now state ownership transfers and that the source array must not be mutated afterward, matching the `FromArray`/`FromMatrix` contract.
- **Storage consolidation doc debt** (#108): 7 planning/review docs reconciled with the single `ColumnStorage<T>` design; public zero-copy claims aligned with the post-consolidation span semantics (Task 7).

### Added

- **Public zero-copy tensor view** (#107): `NivaraColumn<T>.AsTensorView()` and `NivaraSeries<T>.AsTensorView()` are now public (previously internal). They return a lazy `Tensor<T>` view sharing the column's/series' backing array with no copy; null-containing columns and reference element types throw `InvalidOperationException`. Callers must treat the view as read-only.
- **`NivaraEmbeddingGenerator<TInput>`** in `Nivara.Extensions` (AI): wraps any `IEmbeddingGenerator<TInput, Embedding<float>>` as a label column generator for `NivaraFrame.FromRows`; brings `Microsoft.Extensions.AI.Abstractions` into Extensions. Powers the `NivaraChat --embed` and `--rag`/`--rag-agent` modes.

### Query Engine

- `OrderBy`/`OrderByDescending` support computed sort keys (`OrderBy(x => x["A"] + x["B"])`) via a materialized-key `SortByExpressionOperation` — no longer throws `NotSupportedException`; null placement and direction match `Sort` semantics
- `ThenBy`/`ThenByDescending` compose secondary sorts lexicographically with a preceding `OrderBy`/`Sort`: `NivaraFrame` string overloads and LINQ `QueryFrame` lambda overloads, both computed-key capable. Column-reference keys merge into the efficient multi-key `SortOperation`; computed keys merge into a multi-key `SortByExpressionOperation`. Without a preceding sort they act as a primary sort

## [1.1.0] - 2026-07-31

### Automatic Differentiation (inference-default)

- Reverse-mode graph construction is opt-in via `GradientUtils.Grad()`; inference is the default and records no graph nodes
- Type constraint relaxed from `INumber<T>` to `IFloatingPointIeee754<T>` — `float`, `double`, `Half`/F16 and BFloat16 pass runtime validation
- All differentiable operations span-ified over `TensorPrimitives` (no `NivaraColumn.Data` access)
- ADR-001 non-nullable domain cleanup: null-mask infrastructure removed from AutoDiff ops and hot paths; `Debug.Assert` boundary guards in `ReverseGradTensor` and `ComputationGraph.AddNode`

### NN Module System

- `Conv1d<T>` — im2col + `TensorPrimitives.Dot` kernel, PyTorch-compatible weight layout
- `Conv2d<T>` — tiled im2col, PatchLocation lookup, grouped convolution, 1x1 fast path, InputGrad specializations; `ConvTranspose2d<T>`
- `BatchNorm1d<T>` (2D `[N,C]` and 3D `[B,C,L]` inputs) and `BatchNorm2d<T>` — fused span kernels
- `LayerNorm<T>` (SIMD `TensorPrimitives.Dot`), `DepthwiseSeparableConv2d<T>`, `TransformerBlock<T>` (RMSNorm/LayerNorm + GELU), `MultiheadAttention<T>` (self/cross/causal, padding mask)
- `ConvVAE<T>`, `VAE<T>` (optional conditioning), `MaxPool2d<T>`, `AdaptiveAvgPool2d<T>`, `GELU` activation
- `RMSNormKernel<T>` consolidating duplicated per-row RMSNorm logic

### Performance

- SIMD-accelerated kernels via TensorPrimitives chains: Adam, AdamW, PerRowRMSNorm backward, LayerNorm sum-of-squares, GELU forward/backward
- ArrayPool-backed buffer management in hot paths: `AccumulateGradient`, Gather backward, Adam/AdamW state
- `Gather` zero-copy forward path + ArrayPool backward path; `Embedding` lookup via Gather (replaces one-hot + MatMul)

### Training & Serialization

- Optimizers `SGD`, `Adam`, `AdamW` with SIMD kernels; `BCEWithLogitsLoss` fused backward; `MSELoss` `reduceToMean`
- `TrainingLoop<T>`, `DataParallelTrainer<T>`, `TensorDataset<T>`
- `ModelSerializer` JSON/binary save-load; `StateDict()` / `LoadStateDict()` module state

### Samples & Interop

- `samples/NivaraInference` — MobileNetV2/ResNet-18 inference with `SafeTensorsLoader` (I32/I64/F16/BF16/F32 dtype-aware)
- `samples/NivaraFineTuning` — DistilBERT fine-tuning on GLUE SST-2
- `samples/NivaraTimeSeries` — time-series anomaly detection
- `samples/NivaraTorch` — 55 PyTorch-validated functional tests across 21+ layer types (`gen_reference.py` fixtures)
- Generic dtype-aware weight loading for `DistilBertModel`, `MiniLMDistilled`, `SafeTensorsLoader`

### Documentation

- README, GETTING-STARTED, ARCHITECTURE, docs/AUTODIFF updated for the inference-default AutoDiff direction and new modules

## [1.0.0]

- Initial stable release of the columnar DataFrame core: typed immutable columns/frames, LINQ-like query engine with lazy/eager/streaming/parallel strategies, tensor-accelerated kernels, explicit null masks, join/group-by/aggregation, CSV/JSON sources, Parquet/Arrow/ML.NET interop (Extensions), performance diagnostics and buffer pooling
- Reverse-mode AutoDiff (initial), VAE/ConvVAE samples
