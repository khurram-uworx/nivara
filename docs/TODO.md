# TODO — Issue #384: qkvBias docs + backward verification

## Problem

Qwen2-family checkpoints attach a bias to the self-attention `q_proj`/`k_proj`/`v_proj`
linear projections on every block; canonical Llama uses bias-free projections. The additive
core change (a `bool qkvBias = false` option on `LlamaCausalAttention<T>` /
`LlamaDecoderBlock<T>`) shipped on `main` in commit `fd92dc6` (branch `khurram/qwen`), and
`LlamaLoader.Load` auto-detects the bias from safetensors shapes. That implementation is
Torch-checked for **forward/inference** via `QwenInstructParityTests`.

What #384 still tracks (non-blocking backlog):
- No CHANGELOG entry for the new `qkvBias` parameter.
- No public-docs coverage documenting the parameter on the two module constructors.
- The existing NivaraTorch `LlamaCausalAttention`/`LlamaDecoderBlock` parity fixtures are
  `qkvBias=false`; there is no test that the **bias gradient flows correctly through
  `Backward()`** when `qkvBias=true`.

## Proposed changes

1. **CHANGELOG** — add an `[Unreleased] > Added` entry describing the optional `qkvBias`
   parameter on `LlamaCausalAttention<T>` / `LlamaDecoderBlock<T>`, defaulting to `false`
   (canonical Llama byte-identical), auto-detected by `LlamaLoader`. Reference #384 / #382.

2. **Public docs** — document the `qkvBias` parameter. The class-level XML doc already
   describes it; extend `docs/QWEN.md` (the "library map" table already lists the `qkvBias`
   flag) is optional. The primary doc gap is a clear public-facing statement of the option.
   Add a `qkvBias` note to the `LlamaCausalAttention`/`LlamaDecoderBlock` API surface via an
   entry in the AutoDiff docs (`docs/AUTODIFF.md`) if such a modules section exists, plus the
   CHANGELOG.

3. **Backward verification for `qkvBias=true`** — two layers:
   - **Full Torch-backward fixture** (added by human decision): extend
     `samples/NivaraTorch/gen_reference.py` with a `qkvBias=true` Llama attention + decoder
     block case (dedicated RNG appended at the END of the generation stream so all existing
     fixtures stay bit-identical) that saves the Q/K/V bias tensors, the output, and the
     input **and bias gradients** of the sum. Regenerate the torch-comparison tree
     (`python gen_reference.py`) and add matching `LlamaCausalAttention`/`LlamaDecoderBlock`
     parity tests that load the new fixture, set the Q/K/V bias parameters, and assert
     forward output + input grad + bias grad against the PyTorch reference.
   - **Structural NUnit tests** in `tests/Nivara.Tests/AutoDiff/LlamaCausalAttentionTests.cs`
     (independent of Python):
     - With `qkvBias: true`, `QProj.Bias`/`KProj.Bias`/`VProj.Bias` are non-null with shapes
       `[numHeads*headDim]` / `[numKeyValueHeads*headDim]` (×2), and `OProj.Bias` is null.
     - With `qkvBias: false` (default), all four projection biases are null.
     - Inside `Grad()`, run a forward + `Sum(output).Backward()` and assert the `QProj.Bias`,
       `KProj.Bias`, `VProj.Bias` gradients are finite and non-null (bias gradient flows).
     - A `ForwardCached` parity check: for a `qkvBias: true` attention, cached single-token
       forward matches full forward (bias applied consistently in both paths).
     - Mirror the key assertions on `LlamaDecoderBlockTests` (bias presence/absence + grad flow).

## Verification

- `dotnet build Nivara.slnx` (ask human before `dotnet test`).
- Run `python gen_reference.py` (PyTorch fixture regeneration) and commit the
  torch-comparison tree as one unit. Verifies: torch 2.13 available on this machine
  (existing fixtures were generated with it; if the environment lacks torch, the fixture
  regeneration step is blocked and must be surfaced to the human).
- Run the targeted AutoDiff + NivaraTorch test classes. Full suite only after explicit
  confirmation per AGENTS.md.
- Independent reference: the existing `Model_QkvBias_TensorsLoaded` already pins the loader
  path and forward behaviour; the new tests pin the constructor contract + backward bias flow.

## Blast radius

- **New tests only**: `tests/Nivara.Tests/AutoDiff/LlamaCausalAttentionTests.cs`,
  `tests/Nivara.Tests/AutoDiff/LlamaDecoderBlockTests.cs`,
  `tests/Nivara.Tests/NivaraTorch/LlamaCausalAttentionTests.cs` (additive methods, no
  changes to existing tests).
- **Fixture generation**: `samples/NivaraTorch/gen_reference.py` (new additive cases at end
  of stream) + `samples/data/torch-comparison/` (new `.bin` files + manifest; existing
  fixtures bit-identical).
- **Docs**: `CHANGELOG.md` (new entry), possibly `docs/AUTODIFF.md` / `docs/QWEN.md`.
- **No changes** to `src/Nivara`, `samples/Nivara.Samples`, or the loader — the core is
  already implemented and merged. Downstream callers of `LlamaCausalAttention` /
  `LlamaDecoderBlock` are unaffected (default param preserves existing call sites).

## Planned commits

1. `docs: add qkvBias CHANGELOG entry for #384`
2. `test(torch): add qkvBias=true Llama attention/decoder backward fixtures (#384)` (Python
   script + regenerated binary fixtures as one unit)
3. `test(autodiff): verify qkvBias backward gradient flow (#384)` (structural NUnit + the
   NivaraTorch parity tests that consume the new fixtures)
4. (conditional) `docs: document qkvBias option for #384` if an AUTODIFF modules section is a
   natural home

## Out of scope (decision — flagged)

- **BF16 loader promotion** (`ReadUInt16` + `WidenBf16ToF32` → `src/Nivara`/Extensions): the
  issue explicitly defers this "if a prod caller wants lossless BF16 load" and there is no
  prod caller yet. Promoting the loader would be assumption-driven scope creep and touch the
  core assembly ABI. Skipped; can be a follow-up issue if a real caller appears. (Ask human.)

## GitHub issues log

- [ ] #384 — tracked qkvBias docs + verification (this plan executes the remaining backlog)
