# Plan: #392 — memory-map the safetensors file to drop the ~1 GB File.ReadAllBytes byte[] from peak

Branch: `khurram/392` (stacked on `khurram/388`, off `main`)

## Problem

`SafeTensorsLoader.Read<T>` / `Read` (samples-only, `samples/Nivara.Samples/SafeTensorsLoader.cs`)
now fuse the BF16→F32 SIMD widen directly into each tensor's `float[]` (#388), but every
string-path load still does `File.ReadAllBytes(path)` — loading the **entire file into a managed
`byte[]` before parsing**. For the 989 MB Qwen2.5-0.5B checkpoint that means holding ~989 MB
(`byte[]`) + ~1.88 GB (`float[]`) simultaneously at peak ≈ **~2.9 GB**.

safetensors needs random access into the byte payload by per-tensor offset, so the file must be
randomly addressable. `MemoryMappedFile.CreateFromFile` → `CreateViewAccessor` →
`SafeMemoryMappedViewHandle` lets the OS page the file in on demand, eliminating the separate
989 MB managed `byte[]`: `ParseHeader` needs only the small header bytes (KBs) copied out, and
each tensor's bytes are read directly from the mapped view.

Expected saving: ~1 GB of peak *managed* memory on the default Qwen load, same timing (the
widen kernel is unchanged).

## Decisions (confirmed with human)

1. **Memory-map the string-path loads** (`Read(string)`, `Read<T>(string)`); **keep the
   `byte[]` overloads** (`Read(byte[])`, `Read<T>(byte[])`) as-is — fixture tests build
   in-memory buffers and depend on them; they also give the A/B a clean "before" reference
   (`Read(File.ReadAllBytes(path))`).
2. **Unsafe full-span zero-copy** — `AcquirePointer` → `new ReadOnlySpan<byte>(ptr, length)`
   over the mapped view (recommended; matches the README's existing "zero-copy from the
   memory-mapped file buffer" claim). Requires `AllowUnsafeBlocks` in `Nivara.Samples.csproj`
   (core `src/Nivara`/`Extensions` already enable it). The safe alternative (`view.ReadArray`
   into a pooled buffer per tensor) reintroduces a transient copy per tensor — rejected.
3. **No converter retains the source span** (all `Convert*` allocate `new T[]`), so the mapped
   pointer can be released as soon as the read completes — safe within one method frame.
4. **Branch stacked on `khurram/388`** — `main` still carries the two-step loader
   (`ReadUInt16`/`WidenToF32`); #392 must build on the fused loader. Open as a stacked PR
   targeting `khurram/388` (auto-retargets to `main` after #393 merges).
5. **Metric honesty** — `Process.PeakWorkingSet64` is unreliable on this OS (measured 19 MB in
   #388), so the A/B uses managed-heap high-water via `GC.GetTotalMemory` sampling. The ~1 GB
   saving is a *managed*-heap saving; physical working set may show a smaller delta because OS
   page-cache counts the file pages either way.

## Proposed changes

### 1. `samples/Nivara.samples/Nivara.Samples.csproj`
- Add `<AllowUnsafeBlocks>true</AllowUnsafeBlocks>` to the `PropertyGroup`.

### 2. `samples/Nivara.Samples/SafeTensorsLoader.cs`
- Add `using System.IO.MemoryMappedFiles;`.
- Refactor the read core to a span-based form (behavior identical):

```csharp
public static Dictionary<string, (float[] Data, int[] Shape)> Read(string path)
    => Read<float>(path);

public static Dictionary<string, (float[] Data, int[] Shape)> Read(byte[] bytes)
    => Read<float>(bytes.AsSpan());

public static Dictionary<string, (T[] Data, int[] Shape)> Read<T>(string path)
    where T : struct, IFloatingPointIeee754<T>
{
    ArgumentException.ThrowIfNullOrWhiteSpace(path);
    if (!File.Exists(path))
        throw new FileNotFoundException($"SafeTensors file not found: {path}", path);

    long fileLength = new FileInfo(path).Length;
    if (fileLength < 8)
        throw new InvalidDataException("SafeTensors file is too small to contain a header.");
    if (fileLength > int.MaxValue)
        throw new NotSupportedException(
            "SafeTensors files larger than 2 GB are not supported by the memory-mapped loader.");

    using var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
    using var view = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
    unsafe
    {
        byte* ptr = null;
        view.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
        try
        {
            return Read<T>(new ReadOnlySpan<byte>(ptr, (int)fileLength));
        }
        finally
        {
            view.SafeMemoryMappedViewHandle.ReleasePointer();
        }
    }
}

public static Dictionary<string, (T[] Data, int[] Shape)> Read<T>(byte[] bytes)
    where T : struct, IFloatingPointIeee754<T>
    => Read<T>(bytes.AsSpan());

static Dictionary<string, (T[] Data, int[] Shape)> Read<T>(ReadOnlySpan<byte> buffer)
    where T : struct, IFloatingPointIeee754<T>
{
    var entries = ParseHeader(buffer, out int dataOffset);
    var dataBuffer = buffer.Slice(dataOffset);
    var result = new Dictionary<string, (T[] Data, int[] Shape)>(StringComparer.Ordinal);
    foreach (var (name, dtype, shape, begin, end) in entries)
    {
        ReadOnlySpan<byte> tensorBytes = dataBuffer.Slice(begin, end - begin);
        result[name] = (DtypeToArray<T>(tensorBytes, dtype, name), shape);
    }
    return result;
}
```

- `ParseHeader(byte[] bytes, out int dataOffset)` → `ParseHeader(ReadOnlySpan<byte> buffer,
  out int dataOffset)`; internals switch `bytes.Length` → `buffer.Length`,
  `bytes.AsSpan(0, 8)` → `buffer.Slice(0, 8)`, `Encoding.UTF8.GetString(bytes, 8, (int)
  headerSize)` → `Encoding.UTF8.GetString(buffer.Slice(8, (int)headerSize))`.
- Doc comments updated on the changed members (mmap note, 2 GB limit note).

### 3. `tests/Nivara.Tests/AutoDiff/SafeTensorsLoaderBf16Tests.cs`
- **Add** `ReadFloat_OnBf16FixtureFile_MatchesByteArrayPath`: write the existing 3-tensor
  fixture bytes to a temp `.safetensors` file, assert `Read<float>(path)` (mmap) equals
  `Read<float>(bytes)` for keys, shapes, values (bit-exact). Always runs — the cheap
  regression gate for the mmap path.
- **Add** `ReadFloat_OnQwenCheckpoint_MemoryMappedMatchesByteArrayPath`: gated on the Qwen
  checkpoint existing; load once via `Read<float>(path)` (mmap) and once via
  `Read(File.ReadAllBytes(path))`, assert full 290-tensor key/shape/value equality — the
  definitive end-to-end gate.

### 4. A/B benchmark (temp harness, outside repo, not committed)
- `C:\Users\khurram\AppData\Local\Temp\opencode\safetensors_mmap_bench\`:
  `safetensors_mmap_bench.csproj` (refs `Nivara.Samples`) + `Program.cs`.
- Median-of-3 (Release, this machine) on `samples/data/qwen2.5-0.5b-instruct/model.safetensors`:
  - Before: `SafeTensorsLoader.Read(File.ReadAllBytes(path))`
  - After:  `SafeTensorsLoader.Read(path)`
- Instrument with the #388 managed-heap sampler (`GC.GetTotalMemory` on a background thread);
  report median load ms + managed-heap high-water per path.

### 5. Docs
- `docs/QWEN.md` (~line 343): "989 MB `byte[]` + 1.88 GB F32 stays" → memory-mapped framing
  (file pages paged in on demand, no managed `byte[]`).
- `samples/NivaraInference/README.md`: "SafeTensors loader" bullets (665–666) describe the
  mmap as *aspirational* today — make them accurate + add the `#392` reference; update the
  `Load parse` row/note if the A/B numbers move.
- `docs/BFLOAT16.md`: fused-path section — mention the load is memory-mapped (no full-file
  `byte[]`).

## Verification

- Build: `dotnet build Nivara.slnx` (ask before running).
- Tests: `dotnet test --filter SafeTensorsLoaderBf16Tests` (ask before running) — includes the
  new fixture-file parity test; Qwen checkpoint tests gated on file presence. Existing
  real-file tests (`PerfTests`, `QwenInstructParityTests`, `QwenToolsWeatherLoopTests`,
  `DistilBertPrecisionInferenceTests`) exercise the mmap path automatically.
- A/B benchmark (ask before running): fused ReadAllBytes vs fused mmap — timing + managed-heap
  high-water; record honestly.
- Full suite: ask before running.

## Blast radius

- `SafeTensorsLoader.cs` (samples) — string-path internals switch to mmap; `byte[]` overloads
  unchanged; public API surface unchanged (`Read(string)`, `Read(byte[])`, `Read<T>(string)`,
  `Read<T>(byte[])`, `WidenBf16ToF32` keep signatures).
- String-path callers (transparent, no code change; become mmap regression gates):
  `NivaraInference/Program.cs` (148/157/167), `NivaraInference/Qwen.cs` (537),
  `samples/Nivara.Samples/BertModel.cs` (601), `NivaraFineTuning/Program.cs` (151),
  `NivaraChat/Qwen/QwenMode.cs` (69), `NivaraChat/SmolLM/SmollmMode.cs` (142), plus tests
  `DistilBertPrecisionInferenceTests`, `PerfTests`, `QwenInstructParityTests`,
  `QwenToolsWeatherLoopTests`, `SafeTensorsLoaderBf16Tests` (Qwen checkpoint).
- Byte[]-path callers (unchanged): `SafeTensorsLoaderBf16Tests` fixture tests.
- `Nivara.Samples.csproj`: + `AllowUnsafeBlocks`.
- No `src/Nivara` (core) impact.

## Planned commits

1. `docs: plan #392 in TODO.md` — this file.
2. `feat(samples): memory-map the safetensors string-path loads` — csproj +
   `SafeTensorsLoader.cs` (span core + mmap) together to keep the build green.
3. `test: verify memory-mapped load matches the byte[] path` — fixture-file parity + Qwen
   checkpoint 290-tensor value parity.
4. `docs: describe the memory-mapped safetensors load` — QWEN.md, BFLOAT16.md,
   NivaraInference README (record A/B numbers after the benchmark).

## GitHub issues log

- (empty — #392 is the work being done; follow-ups discovered during execution will be logged
  here as they are created)