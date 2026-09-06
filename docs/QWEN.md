# Qwen2.5 Tool Calling on Nivara (#382)

Native function calling with **Qwen2.5-0.5B-Instruct** served as a plain
`Microsoft.Extensions.AI.IChatClient` over Nivara's in-process `LlamaForCausalLM`
engine — no LLM server, no Python runtime. `--qwen tools-weather` runs the real
`<tool_call>` → `GetWeather` → `<tool_response>` → final-answer loop, capped at 3
iterations, and closes with a clean natural-language answer.

This document records the ground-truth findings (the format, the vocab-size
subtlety, the tokenizer divergence, the loader benchmark), what was reusable from
Phase B, what was fixed and why, and the verification evidence. The code lives in
`samples/NivaraChat/Qwen/`; the tests in `tests/Nivara.Tests/Qwen/`.

Related Qwen work, tracked separately: #384 (qkvBias loader gap), #386 (Qwen
distillation), #387 (BF16-in-memory SIMD-dot), #388 (fused BF16→F32 read),
#390 (GGUF backend), #391 (revisit BF16 workarounds when `Vector<BFloat16>` SIMD
lands).

---

## How Qwen works on Nivara — the library map

An engineer wanting to run or extend Qwen should start here. Two sample READMEs
anchor the fuller technical detail and are cross-linked throughout:

- **`samples/NivaraChat/README.md`** — the user-facing `--qwen` demo (function
  calling via `IChatClient`): `--qwen tools-weather|chat|plain`, the
  `FunctionInvokingChatClient` tool loop, and the code layout under
  `samples/NivaraChat/Qwen/`.
- **`samples/NivaraInference/README.md`** — the low-level library scratchpad:
  `qwen tools` (function calling), `qwen distill` (teacher distillation,
  #386), `qwen benchmark` (KV-cached vs full re-forward), plus the weight
  loading, narrow-precision, and SafeTensors loader sections.

### The load path (checkpoint → tensors)

```
config.json + model.safetensors + vocab.json + merges.txt + tokenizer.json
  → SafeTensorsLoader.Read<T> (BF16-on-disk, fused SIMD WidenBf16ToF32 → F32, #388)
  → LlamaLoader.Load + LlamaConfig.FromJson → LlamaForCausalLM<T>
  → Gpt2BpeTokenizer  (Qwen Split-regex pretokenizer + added tokens)
```

> **Why we widen to F32 at load (and don't run BF16 compute)** — Qwen is
> BF16-on-disk (~1 GB), but Nivara **widens to F32 once at load time** and runs a
> pure-F32 loop. The reason: `Vector<BFloat16>` is **unsupported** on .NET 11
> (`IsSupported == false` at every width), so BF16 compute falls back to a scalar
> BCL path that is radically slower than F32 SIMD, while the widen is **lossless**
> (BF16 *is* the top 16 bits of float32) and costs only a one-time pass. This is
> Nivara's current recommendation; its full rationale, the A/B numbers, and the
> `WidenBf16ToF32` mechanics live in **[`docs/BFLOAT16.md`](BFLOAT16.md)**.
> The widen is fused into the read: `Read<float>` runs the SIMD kernel directly
> into each tensor's `float[]` with **no interim `ushort[]`** (one pass, ~1 GB less
> peak memory). When `Vector<BFloat16>` SIMD support lands in the runtime this
> tradeoff is worth revisiting (#387, #388, **#391**).

Qwen's config is Llama-loader-compatible: `hidden 896`, 24 layers, **GQA 14↔2**
KV heads, SiLU gated FFN, `RMSNorm` (ε=1e-6), **RoPE θ=1_000_000** (10× SmolLM),
`max_position_embeddings 32768`, `vocab 151936`, tied embeddings, and **biased
Q/K/V projections**. Of these only the biased projections were a real core gap —
everything else reused the SmolLM machinery unchanged (`LlamaLoader`,
`LlamaForCausalLM<T>`, GQA `GqaRepeatKV`, `RMSNorm<T>`, SiLU, RoPE
`rotate_half`, tied LM head). See `samples/NivaraInference/README.md` →
"Qwen2.5-0.5B-Instruct" for the reusable-vs-new breakdown.

### What was added where (core vs sample-scoped)

| Piece | Location | Notes |
|---|---|---|
| `qkvBias` flag on `LlamaCausalAttention<T>` / `LlamaDecoderBlock<T>` | **core** `src/Nivara/AutoDiff/Nn/` | the one additive core change (#384); default `false`, SmolLM unchanged |
| `SafeTensorsLoader.Read<float>` fused BF16→F32 (SIMD `WidenBf16ToF32`) | **sample** `samples/Nivara.Samples/` | #388; BF16 widen fused into the read, no interim `ushort[]`, ~1 GB less peak memory |
| `LlamaForCausalLM<T>`, `LlamaConfig`, `LlamaLoader`, `StateDictLoader` | **sample** `samples/Nivara.Samples/` | Qwen loads via the same route SmolLM uses |
| `Gpt2BpeTokenizer` Qwen preamble | **sample** `samples/Nivara.Samples/` | `Split`-regex pretokenize + added-token merge |
| `LlamaKVCache<T>` + `ForwardCached` | **sample** | per-token KV-cached decode |
| `QwenChatClient<T>`, `QwenChatTemplate`, `QwenToolCallParser`, `QwenSampleTools` | **sample** `samples/NivaraChat/Qwen/` | the `IChatClient` + tool loop |

The `QwenChatClient` is a plain `Microsoft.Extensions.AI.IChatClient` over
`LlamaForCausalLM<T>` + `Gpt2BpeTokenizer` + `LlamaKVCache<T>`, wrapped by MEAI
10.9.0's `FunctionInvokingChatClient` for the loop. It runs **inference-default**
(ADR-001/002): `model.Eval()`, never inside `GradientUtils.Grad()`, so no graph
nodes are built (`samples/NivaraInference/README.md` documents the same
inference-default guarantee for its `qwen` modes). The Gpt2BpeTokenizer/loader
gaps, the byte-exact renderer, and the parser are all described in detail below;
the `qwen tools`/`qwen distill` feature surface is in
`samples/NivaraInference/README.md`, and `qwen tools-weather` wiring in
`samples/NivaraChat/README.md` → "Qwen (`--qwen`)".

> **KV-cache & generation pipeline** — render → `Encode` → `SeedCache` (KV prefill
> per prompt token) → per-token `ForwardCached` → decode → parse. Greedy `ArgMax`
> by default; `temperature > 0` adds temperature softmax + optional top-p, from a
> seeded shared RNG. Stops on `QwenIds.StopIds` `[151645, 151643]`. Details in
> *The client* section below.

---

## Why Qwen2.5-0.5B-Instruct

Phase B (`--smollm tools-weather`, branch `khurram/causal-lm-b`) proved the whole
MEAI loop on a 0.15B community SmolLM2-Hermes fine-tune, but that model **never
produced a final answer** — it re-issued tool calls until the `MaximumIterationsPerRequest`
cap and the demo blanked. The pivot to `Qwen/Qwen2.5-0.5B-Instruct` (a
mainstream, documented, **native** function-calling model; ~988 MB BF16 on disk,
~2 GB in F32) keeps the improvement goal: verify what loads through the existing
library, then implement the gaps properly with PyTorch/Torch compatibility checks.

---

## Ground truth from the checkpoint (pre-flight, verified)

All facts below come from the actual downloaded files, not assumptions.

### Tool format is Hermes-style, not Qwen2-era

`tokenizer_config.json`'s `chat_template` shows Qwen2.5 uses
`<tool_call>…</tool_call>` (added-token ids **151657 / 151658**) — *not* the
`<|tool_call_start|>` markers of Qwen2, and *not* the SmolLM2 `HermesToolMessage`
names. Tool results are rendered as a `user` turn wrapped in
`<tool_response>…</tool_response>`:

```
<|im_start|>assistant
<tool_call>
{"name": "getWeather", "arguments": {"city": "Paris"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Partly cloudy, 18°C. Light breeze from the northwest.
</tool_response><|im_end|>
<|im_start|>assistant
```

### Special-token ids

| Token | Id | Role |
|---|---|---|
| `<|endoftext|>` | 151643 | also the bos id; second eos |
| `<|im_start|>` | 151644 | ChatML turn opener |
| `<|im_end|>` | 151645 | primary eos (`eos_token_id[0]`) |
| `<tool_call>` | 151657 | added token, tool-call opener |
| `</tool_call>` | 151658 | added token, tool-call closer |

`generation_config.json` sets `eos_token_id: [151645, 151643]` — generation must
stop on **either**. `bos_token_id` is 151643. `QwenChatClient` hard-codes
`QwenIds.StopIds = [151645, 151643]`.

`<tool_response>` / `</tool_response>` are **NOT** added/special tokens — they
tokenize as ordinary bytes (`[27, 14172, 9655, 29]` for `/API<`-style chunks).
The renderer writes them as plain text; only the five specials above need
added-token handling in the tokenizer.

### Vocab-size subtlety (the audited risk item)

- `config.vocab_size` = **151,936** → the embed/head table is `[151936, 896]`.
- The base BPE vocab (`vocab.json`) holds **151,643** entries; generation/argmax
  must be able to name any of the 151,936 embed rows, so it runs over the
  config vocab.
- `tokenizer.json` adds 22 more → **151,665** known ids total (the "293-row tail"
  is the added-token gap between base vocab and embed table).
- Conclusion, applied: embed/head from config (151,936), argmax over
  `config.VocabSize`, BOS/EOS by id, the added tokens merged into the tokenizer.

### Tokenizer preamble divergence (Split + ByteLevel, not legacy GPT-2)

Qwen's `tokenizer.json` declares `pre_tokenizer: Sequence(Split(Isolated, regex),
ByteLevel(use_regex:false))`. The legacy GPT-2 path applied the byte-level regex
to the byte-mapped text *first*, which is what SmolLM's `Digits +
ByteLevel(use_regex)` pipeline needs. Qwen is different: the **Split regex applies
to the raw normalized text, then each chunk is byte-mapped** (`Isolated`
semantics, no byte-level regex).

Fix: `Gpt2BpeTokenizer` now reads the declared `Split` regex from a
`tokenizer.json` (when present) and routes encoding through
`PretokenizeSplit` → per-chunk byte map. SmolLM's construction (no
`tokenizer.json` / no Split) still takes the legacy GPT-2 path untouched. This
was the fix (`48c789a`) that made Qwen tokenization match the HuggingFace
reference.

---

## What was not reusable from Phase B (and why)

1. **Tool format.** Phase B rendered SmolLM2-Hermes `HermesToolMessage` style and
   used `v3` JSON names; Qwen emits plain `<tool_call>` JSON. The parser and
   renderer are Qwen-specific.
2. **`FunctionInvokingChatClient` binding.** In MEAI 10.9.0 the tool binder
   consumes `FunctionCallContent.Arguments` as an `IDictionary<string, object?>`.
   Phase B's `BuildToolCallContents` JSON path was fragile and could produce a
   mangled dict (the silent-failure root cause). The Qwen client **builds the dict from
   the parsed JSON** (`QwenToolCallParser`), so the binder always sees a correct,
   serializable argument map.
3. **Assistant-turn contents.** `ChatMessage.Text` is *read-only* (derived from
   `Contents`). A tool-call assistant turn carries **only**
   `FunctionCallContent` — adding `TextContent` alongside double-renders the turn
   in the next request. Phase B had this wired for its own format; the Qwen client
   preserves it.
4. **Tool JSON schema source.** `AIFunction.JsonSchema` is unusable byte-for-byte:
   it emits `description` before `type` and escapes `'` as `\u0027`. The renderer
   builds the schema **manually** from `AIFunction.UnderlyingMethod` +
   `[Description]` attributes with type-first property order and literal
   characters (`QwenChatTemplate.ToolJson`), matching Jinja `tojson`.

---

## The renderer: `QwenChatTemplate` (byte-exact vs Torch)

`Render(IEnumerable<ChatMessage>, bool addGenerationPrompt)` mirrors
HuggingFace's `apply_chat_template` for this checkpoint exactly:

- A leading `system` message is used verbatim; otherwise the default
  `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."` turn is
  emitted (the template's `else` branch).
- `user` / `system` turns: `<|im_start|>role\n{text}<|im_end|>\n`.
- `assistant` with tool calls: `<|im_start|>assistant` + per-call
  `\n<tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>` + `<|im_end|>\n`.
- `role == tool` messages become a `user` turn:
  `<|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>\n`
  (no newline before `<|im_end|>` — the fixtures pin this).
- `addGenerationPrompt` appends `<|im_start|>assistant\n`.

`BuildToolsSystemMessage(tools)` bakes the tool-mode system turn
(`# Tools` instructions + `<tools>` schemas + the `<tool_call>` exemplar),
byte-identical to the checkpoint template's `{%- if tools %}` branch.

JSON layout is handwritten to reproduce Jinja `tojson`: spaces after `:`/`,`, and
`UnsafeRelaxedJsonEscaping` (literal `'`, `°`, no HTML/Unicode escaping) —
`QwenJson.ToSpaced`. `JsonObject`/`JsonArray` preserve insertion order, so the
schema matches the reference property order.

The byte-exact pins are the ground-truth fixtures `qwen_tool_prompt.txt` /
`qwen_tool_final_prompt.txt` in the gitignored model dir, compared byte-for-byte
by `QwenChatTemplateTests`. The round-trip test parses the fixture's tool-call
turn and re-renders it — the exact parse→render cycle the live loop performs —
and asserts the result is the fixture.

---

## The parser: `QwenToolCallParser`

`Parse(text, knownToolNames)`:

1. Strict `JsonDocument` parse of each `<tool_call>(.*?)</tool_call>` block
   (spacing-agnostic). Requires `name` (string) + `arguments` (object);
   `arguments` properties become the `FunctionCallContent.Arguments` dict
   (values are cloned `JsonElement`s — the "correct dict" fix).
2. On `JsonException` (or wrong shape), a tolerant regex fallback extracts
   `name` and stores the unparseable remainder under `__raw` so the failure is
   observable, never silently dropped.
3. `knownToolNames` canonicalizes the emitted name by case-insensitive match
   (so `GetWeather` still resolves to the registered `getWeather` AIFunction);
   unknown names pass through as emitted.

---

## The client: `QwenChatClient<T>`

`IChatClient` over `LlamaForCausalLM<T>` + `Gpt2BpeTokenizer` +
`LlamaKVCache<T>`:

- **Inference-default (ADR-001/002):** `model.Eval()`; the client never enters a
  `GradientUtils.Grad()` scope, so every operation short-circuits to the non-grad
  span path and no graph nodes are ever built (the ADR-002 inference-default guard).
- **Generation:** render → `tokenizer.Encode` → KV-cached prefill
  (`SeedCache` forward per prompt token) → per-token `ForwardCached` → decode →
  parse. Greedy (`ArgMax`) is the default; `temperature > 0` switches to a
  temperature softmax with optional top-p nucleus filtering, drawn from a seeded
  shared RNG. Stops on `QwenIds.StopIds` (151645 or 151643).
- **`GetResponseAsync`:** returns exactly one assistant message — `FunctionCallContent`s
  when `<tool_call>` blocks parsed, else a single `TextContent` — and invokes the
  optional `turnCallback` with the raw decoded text for live printing.
- **`GetStreamingResponseAsync`:** yields one `ChatResponseUpdate` per generated
  id, decoded per-token.
- No KV-cache path available (`useKvCache: false`): full `model.Forward(ids)` per
  step (numeric-identical, slower).

---

## The loop: `--qwen tools-weather`

`QwenMode` wires the client through MEAI 10.9.0's `FunctionInvokingChatClient`
as `new FunctionInvokingChatClient(inner) { MaximumIterationsPerRequest = 3 }`,
passing `ChatOptions.Tools = [weather]` per request. `QwenSampleTools.GetWeather`
is a deterministic `AIFunction` (`getWeather`, description
`"Gets the current weather for a city. Returns a short description like 'Sunny, 22°C'."`,
param `city`). The system turn is the baked `BuildToolsSystemMessage([weather])`
so the first render is byte-identical to the Torch tools prompt.

Observed transcript (real run, F32, greedy):

```
You: What's the weather in Paris?
[assistant → getWeather(city: Paris)]
[tool] Partly cloudy, 18°C. Light breeze from the northwest.
Qwen: The weather in Paris is partly cloudy with a temperature of 18°C. The light breeze from the northwest is expected.
[3 turn(s) in 342567 ms]

```

Two model generations (tool-call turn + final turn), loop closes within the cap
(the 343 s figure includes the ~1-min in-process F32 load of the 988 MB BF16
checkpoint). `--smollm chat|plain` is untouched; `--qwen chat|plain` are plain
streaming modes with the same generation core.

---

## Verification evidence

### Torch parity (`QwenInstructParityTests`, 13 tests)

- The model/tokenizer load through `LlamaLoader` / `Gpt2BpeTokenizer` with zero
  *caller* changes (10 tensor names, `tie_word_embeddings`, q/k/v bias variant).
  The q/k/v-bias load surfaced an additive library gap — `LlamaCausalAttention` /
  `LlamaDecoderBlock` gained an optional `qkvBias` parameter (default `false`,
  canonical Llama unchanged; **tracked separately as #384**).
- **Tool turn: 19/19 ids byte-exact** vs the Torch greedy reference — the
  structural function-call contract (`<tool_call>` … `</tool_call>`, id-exact).
- **Final turn: semantic parity, with the tie-flip provably a near-tie.** At
  generated index 9 the F32 model assigned **0.0234 logits** separating
  `' high'` (19.38) vs `' temperature'` (19.40); PyTorch (computing in BF16,
  `torch_dtype="auto"`) resolves that hairline differently, so exact token ids
  diverge. The oracle is decode-then-compare: the final answer is asserted for
  the weather conclusion (`partly cloudy` / `northwest`), not exact ids.
- Last-row logits over the fixed Torch trajectory: worst observed absolute diff
  **0.399** on a low-probability tail entry (~2.3% of max-logit magnitude),
  argmax = 151645 (`<|im_end|>`) intact. Honest tolerance envelope: 3% relative
  + 0.5 absolute floor. Gross numeric errors (rope/transpose/attention) land far
  outside it.
- Qwen pretokenization parity: the Split-regex finding from above is pinned by
  tokenizer tests (ids equal the HF `AutoTokenizer` reference).

### BF16 loader gap + benchmark (`SafeTensorsLoaderBf16Tests`, #388)

- **Single fused load**: `SafeTensorsLoader.Read<float>(path)` reads a BF16
  checkpoint and widens directly into each tensor's `float[]` via
  `WidenBf16ToF32(ReadOnlySpan<ushort>, Span<float>)` — a `Vector<ushort>` SIMD chain
  (`Vector.Widen` → `<<16` → reinterpret) with a scalar tail for partial vectors,
  run **during the read** with **no interim `ushort[]`** (one pass). All 65,536 BF16
  patterns property-match the scalar reference
  (`WidenBf16ToF32_AllBitPatterns_MatchesScalarReference`).
- **Qwen checkpoint load** (988 MB BF16, on this machine, Release): the fused
  `Read<float>` finishes in roughly **1.1–1.4 s** (median ~1.3 s warm [#392]) — it
  does **not regress** the earlier two-step path (2.07–2.16 s [#388]) and its peak
  **managed** memory drops ~1 GB in two steps: no interim `ushort[]` (#388) and no
  full-file `byte[]` — the string-path read memory-maps the file and touches each
  tensor's pages on demand (#392). Peak managed heap: **2.83 GB → 1.88 GB**
  (measured A/B). The mmap read is ~1.6× slower than a warm `ReadAllBytes` copy
  (per-page soft-fault overhead for random per-tensor access), so the managed-heap
  saving trades ~0.5 s on the one-time model load; physical working set is similar
  either way (the OS page cache holds the file either way). Earlier docs claimed the
  two-step was "~2.5× faster / half the RAM", but that compared a half-size
  `ushort[]` output (no widen) against a full-size `float[]` output (with widen) —
  a meaningless apples-to-oranges metric, since both paths end at F32. With the
  widen fused in at equal F32 output, the two-step offered no timing or memory win
  and was removed (#388).

### Qwen tool-calling (this work, 12 new tests)

- `QwenChatTemplateTests` (4): the tool prompt and the assistant-tool-call +
  tool-response final prompt render **byte-identical to the Torch fixtures**,
  including the parse→render round trip; plain-chat renders the default system
  turn; the tools system message contains the instructions + schema.
- `QwenToolCallParserTests` (7): canonical, compact, multi-call, case-insensitive
  name canonicalization, unknown-name passthrough, no-tool → empty, malformed →
  tolerant `__raw` fallback.
- `QwenToolsWeatherLoopTests` (1, model-gated, ~5m35s): the real
  `FunctionInvokingChatClient` loop emits `<tool_call>`, executes `GetWeather`,
  feeds back the result, and closes with a non-blank final answer containing
  `partly cloudy` — within the cap.
- CLI acceptance: `--qwen tools-weather --text "What's the weather in Paris?"`
  → transcript above, clean answer, loop terminates. `--smollm chat|plain`
  untouched and still working.
- Regression: 17 earlier parity/loader tests re-run green.

---

## GGUF backend (#390)

A future `--qwen gguf` sub-mode would load the Qwen checkpoint as GGUF through a
.NET GGUF library (candidates: LlamaSharp, TensorSharp — see the issue for the
evaluation plan) and reuse the *same* `QwenChatTemplate` renderer,
`QwenToolCallParser`, and tool-loop wiring — the format contract is
model-loading-agnostic. Tracked in #390.