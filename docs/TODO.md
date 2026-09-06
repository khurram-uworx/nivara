# Plan: #388 — fuse the BF16→F32 widen into the read (drop interim ushort[] from the default qwen path)

Branch: `khurram/388` (off `main`)

## Problem

The `qwen` default load (NivaraInference) is a two-step:
1. `SafeTensorsLoader.ReadUInt16(path)` materializes every tensor as `ushort[]` (raw BF16 bits, ~1 GB across 290 tensors).
2. `SafeTensorsLoader.WidenToF32(raw)` converts those to `float[]` (~2 GB).

Peak transient ≈ 3 GB with **two full passes** over the 989 MB payload. The interim `ushort[]` is pure overhead: the SIMD `WidenBf16ToF32` kernel is already used (fused) inside the generic `SafeTensorsLoader.Read<float>` → `ConvertBF16<float>`, widening directly into the destination `float[]` with no interim `ushort[]`.

The docs' "~2.5× faster / half the RAM" claim for the two-step was an apples-to-oranges comparison (two-step outputs half-size `ushort[]` with no widen; fused outputs full-size `float[]` with widen). Since we end at F32 either way, that metric is meaningless and must be removed/corrected.

## Decisions (confirmed with human)

1. **Merge to one fused path** — reuse the existing `SafeTensorsLoader.Read<float>` (it already runs the SIMD `WidenBf16ToF32` directly into `float[]`, no interim `ushort[]`).
2. **Remove the two-step entirely** — delete `ReadUInt16(string/byte[])` + `WidenToF32` and their tests. **Keep `WidenBf16ToF32`** — it's the shared kernel powering the fused path (called from `ConvertBF16<float>` at line 274).
3. **Drop `--precision ushort`** from the qwen CLI — default (no flag) and `f32` both use the fused `Read<float>`. Remove the `--ushort` alias flag too.
4. **Scope: just the interim-step removal** — memory-map I/O (dropping the 989 MB `byte[]` from `File.ReadAllBytes`) is a **separate follow-up**, out of #388.
5. **Update all four docs** to remove the invalid "~2.5× / half the RAM" framing and describe the single fused path; update the `Load parse` row with the recorded fused number.

## Proposed changes

### 1. `samples/Nivara.Samples/SafeTensorsLoader.cs`
- **Delete** `ReadUInt16(string)`, `ReadUInt16(byte[])`, `WidenToF32(...)`.
- **Keep** `WidenBf16ToF32(ReadOnlySpan<ushort>, Span<float>)` — internal SIMD kernel used by the fused path.
- Update doc comments / XML-crefs that referenced the removed `ReadUInt16`/`WidenToF32` (lines 57, 97, 133).

### 2. `samples/NivaraInference/Program.cs`
- Remove the `useUshort` branch (lines 164–168); the load becomes unconditional `tensors = SafeTensorsLoader.Read(modelPath)`.
- Line 174: `(useUshort && isQwen ? "BF16->F32 widen" : "F32")` → `"F32"`.
- Remove `bool useUshort = precision == "ushort";` (line 148).
- Remove `--precision ushort` from the parse switch (line 48) and the `--ushort` alias block (lines 64–68).
- Remove the ushort defaulting (lines 94–95) and the `ushort` help text (lines 123–124, 152, 157); update the usage string (line 99) and the `--precision f32` help (line 120).

### 3. `samples/NivaraChat/Qwen/QwenMode.cs`
- **No code change** — already uses fused `SafeTensorsLoader.Read<T>`.

### 4. `tests/Nivara.Tests/AutoDiff/SafeTensorsLoaderBf16Tests.cs`
- **Remove**: `ReadUInt16_RawPatternsWidenToSameF32AsReadFloat`, `ReadUInt16_RejectsNonBf16Dtypes`, `ReadUInt16_OnQwenCheckpoint_MatchesReadFloatKeysAndShapes`.
- **Keep**: `WidenBf16ToF32_AllBitPatterns_MatchesScalarReference`, `WidenBf16ToF32_VariousLengths_MatchesScalarReference`, `WidenBf16ToF32_LengthMismatch_Throws` (kernel still used by the fused path).
- **Add**: a fused-path equivalence test — `Read<float>` on the built 3-tensor BF16 fixture equals the scalar BF16→F32 reference (bit-exact); extend the gated Qwen checkpoint test to verify `Read<float>` key/shape parity + full-output equality against the scalar reference for all 290 tensors.
- Update the class doc comment (line 12–13) that references `ReadUInt16`/`WidenToF32`.

### 5. Docs (all four reviewed and updated)
- **`docs/QWEN.md`** — load-path diagram (39–40), core-additions table row (70), BF16 loader benchmark section (331–346).
- **`docs/BFLOAT16.md`** — "Raw BF16 read path (`ReadUInt16`)" section (259–278).
- **`samples/NivaraInference/README.md`** — Precision text (403–408), `--ushort is now the qwen default` comment (400), "Sample-scoped additions" bullet (445–448), "Already generalized" (457), `Load parse` rows/tables (464–488), capability table (842), the f32-opt-in note (486–488).
- **`samples/NivaraChat/README.md`** — "Uses:" line (401) referencing `ReadUInt16` + `WidenBf16ToF32` (code already uses fused `Read<T>`).

### 6. Follow-up (out of #388)
- Memory-map I/O to drop the 989 MB `byte[]` from `File.ReadAllBytes` — create a separate GitHub issue.

## Verification

- Build: `dotnet build Nivara.slnx` (ask before running).
- Tests: `dotnet test` on `SafeTensorsLoaderBf16Tests` + full suite (ask before running).
- The correctness gate is **fused `Read<float>` output == scalar BF16→F32 reference for all 290 Qwen tensors** (the two-step is deleted, so there's nothing to A/B against going forward).
- Record the fused load-parse timing for the README `Load parse` row.

## Blast radius

- `SafeTensorsLoader.cs` (samples): removing `ReadUInt16`/`WidenToF32` public symbols. Callers: `Program.cs` (NivaraInference) and the test file — both updated in this plan. No `src/Nivara` (core) impact.
- `Program.cs` (NivaraInference): qwen load path + CLI option removal.
- Tests: `SafeTensorsLoaderBf16Tests.cs`.
- Docs: four files.
- `WidenBf16ToF32` is **kept** (used by `ConvertBF16<float>`, the fused BF16 path).

## Planned commits

1. `docs: plan #388 in TODO.md` — this file. ✅ (fcd086c)
2. `refactor(samples): remove ReadUInt16/WidenToF32 two-step from SafeTensorsLoader` + `refactor(samples): drop --precision ushort from qwen default load` — ✅ combined into a single commit (a8562e6) because `SafeTensorsLoader.cs` and its caller `Program.cs` must land together to keep the build green (removing the symbols without updating the caller breaks compilation).
3. `test: add fused Read<float> parity vs scalar reference and Qwen checkpoint check` — update `SafeTensorsLoaderBf16Tests.cs`.
4. `docs: remove invalid 2.5x framing, document single fused BF16->F32 load` — update the four docs + README load-parse row.

## GitHub issues log

- [ ] #392 — Memory-map the safetensors file to drop the ~1 GB `File.ReadAllBytes` `byte[]` from peak load memory (created while working on #388; out of #388 scope, separate follow-up).
