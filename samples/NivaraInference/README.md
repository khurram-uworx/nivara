# Nivara HuggingFace Inference Sample

Load pre-trained HuggingFace models (MobileNetV2, ResNet-18, MiniLM, DistilBERT, DistilBERT SST-2, SmolLM-135M-Instruct) into Nivara's zero-dependency tensor engine and run forward inference in pure managed .NET — no Python runtime, no CUDA, no third-party ML framework.

The same architecture is also implemented in PyTorch (`samples/NivaraInference/Python/`) for direct CPU performance comparison.

## Quick start

```bash
# Download model weights via HuggingFace CLI
hf download google/mobilenet_v2_1.0_224 --local-dir samples/data/mobilenet_v2
hf download microsoft/resnet-18 model.safetensors config.json --local-dir samples/data/resnet18
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir samples/data/minilm
# (distilbert-base-uncased already present under samples/data/distilbert)
hf download distilbert/distilbert-base-uncased-finetuned-sst-2-english config.json model.safetensors vocab.txt tokenizer_config.json --local-dir samples/data/distilbert_sst
# SmolLM-135M-Instruct — BF16-native Llama-family causal LM (GQA, SiLU, RoPE; see "SmolLM" below)
hf download HuggingFaceTB/SmolLM-135M-Instruct config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json special_tokens_map.json --local-dir samples/data/smollm-135m
# Qwen2.5-0.5B-Instruct — function calling + teacher distillation (BF16 on disk; see "Qwen2.5-0.5B-Instruct" below)
hf download Qwen/Qwen2.5-0.5B-Instruct config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json special_tokens_map.json --local-dir samples/data/qwen2.5-0.5b-instruct

# Run inference
dotnet run --project samples/NivaraInference -c Release -- mobilenet_v2
dotnet run --project samples/NivaraInference -c Release -- resnet18
dotnet run --project samples/NivaraInference -c Release -- minilm
dotnet run --project samples/NivaraInference -c Release -- distilbert
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst
dotnet run --project samples/NivaraInference -c Release -- smollm                 # greedy causal-LM generation (F32)
dotnet run --project samples/NivaraInference -c Release -- smollm --precision bf16  # native BF16 (256.6 MB)
dotnet run --project samples/NivaraInference -c Release -- qwen tools            # function calling (KV cache on)
dotnet run --project samples/NivaraInference -c Release -- qwen distill --teacher-examples 12  # teacher distillation
dotnet run --project samples/NivaraInference -c Release -- qwen benchmark        # KV-cached vs full re-forward

# Benchmark (10 passes each)
dotnet run --project samples/NivaraInference -c Release -- mobilenet_v2 benchmark
dotnet run --project samples/NivaraInference -c Release -- resnet18 benchmark
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst benchmark

# Narrow-precision inference (half weight memory; see "Narrow-precision inference" below)
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst bf16
dotnet run --project samples/NivaraInference -c Release -- distilbert bf16
dotnet run --project samples/NivaraInference -c Release -- minilm bf16
# fp16 / Half variant (--precision fp16, or a bare fp16/half positional)
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst --precision fp16
dotnet run --project samples/NivaraInference -c Release -- distilbert --precision fp16
dotnet run --project samples/NivaraInference -c Release -- minilm --precision fp16
# Benchmark also honors --precision (times F32 / fp16 / bf16; see "Speed" below)
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark --precision fp16
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark --precision bf16
```

## Supported models

| Model | Type | Weight size | Tensors | Parameters | Output |
|-------|------|-------------|---------|------------|--------|
| MobileNetV2 | Vision (classification) | 13.5 MB | 262 | 3.4M | 1001 classes |
| ResNet-18 | Vision (classification) | 44.6 MB | 102 | 11.7M | 1000 classes |
| MiniLM (L6-v2) | Text (embedding) | 91 MB | 104 | 22.7M | 384-dim embedding |
| DistilBERT (base-uncased) | Text (encoder) | 255.5 MB | 105 | 67.0M | `[seqLen, 768]` hidden states |
| DistilBERT SST-2 (fine-tuned) | Text (classification) | 255.4 MB | 104 | 66.9M | 2-class sentiment (`NEGATIVE`/`POSITIVE`) |
| SmolLM-135M-Instruct | Text (causal LM) | 269 MB | 272 | 134.5M | token ids (generation) |
| Qwen2.5-0.5B-Instruct | Text (causal LM + function calling) | 989 MB | 290 | ~494M | token ids (tool-assisted generation) |

## Usage

### C# (Nivara)

**Vision models:**
```bash
# Random-data inference
dotnet run --project samples/NivaraInference -- mobilenet_v2
dotnet run --project samples/NivaraInference -- resnet18

# Benchmark (10 synthetic + real-image passes)
dotnet run --project samples/NivaraInference -- mobilenet_v2 benchmark
dotnet run --project samples/NivaraInference -- resnet18 benchmark

# Compare output with PyTorch reference
dotnet run --project samples/NivaraInference -- mobilenet_v2 compare
dotnet run --project samples/NivaraInference -- resnet18 compare

# Step-by-step layer diagnostics
dotnet run --project samples/NivaraInference -- mobilenet_v2 compare_diag
dotnet run --project samples/NivaraInference -- resnet18 compare_diag

# Single image inference
dotnet run --project samples/NivaraInference -- mobilenet_v2 path/to/image.jpg
dotnet run --project samples/NivaraInference -- resnet18 path/to/image.jpg
```

**MiniLM:**
```bash
# Tokenize and embed a sentence
dotnet run --project samples/NivaraInference -- minilm

# Benchmark (10 passes)
dotnet run --project samples/NivaraInference -- minilm benchmark

# Pairwise cosine similarity demo
dotnet run --project samples/NivaraInference -- minilm similarity
```

**DistilBERT:**
```bash
# Forward a sentence through the base encoder (output: [128, 768] hidden states)
dotnet run --project samples/NivaraInference -- distilbert

# Benchmark (3 warmup + 10 timed passes)
dotnet run --project samples/NivaraInference -- distilbert benchmark

# Compare hidden states with a PyTorch reference (run the Python script first)
python samples/NivaraInference/Python/distilbert_compare.py
dotnet run --project samples/NivaraInference -- distilbert compare
```

**DistilBERT SST-2 (sequence classification):**
```bash
# Interactive sentiment REPL over the fine-tuned SST-2 classifier
dotnet run --project samples/NivaraInference -- distilbert_sst

# Benchmark (3 warmup + 10 timed passes)
dotnet run --project samples/NivaraInference -- distilbert_sst benchmark

# Compare logits + softmax probs with a PyTorch reference (run the Python script first)
python samples/NivaraInference/Python/distilbert_sst_compare.py
dotnet run --project samples/NivaraInference -- distilbert_sst compare
```

**SmolLM-135M-Instruct (causal LM / generation):**
```bash
# Greedy generation from the fixed prompt (F32; add --precision bf16 for the native BF16 path,
# or fp16 for Half). Run the Python reference generator first to enable the PyTorch diff.
python samples/NivaraInference/Python/smollm_generate_reference.py
dotnet run --project samples/NivaraInference -- smollm
dotnet run --project samples/NivaraInference -- smollm --precision bf16
```

### Python (PyTorch reference)

```bash
cd samples/NivaraInference/Python
pip install -r requirements.txt

python mobilenet.py           # Basic inference
python resnet18.py

python mobilenet_compare.py   # Forward pass on shared input for C# comparison
python resnet18_compare.py

python mobilenet_diag.py      # Step-by-step layer diagnostics
python resnet18_diag.py

python minilm_benchmark.py     # MiniLM CPU timing (same methodology as C#)
python distilbert_benchmark.py # DistilBERT CPU timing (same methodology as C#)
python minilm_compare.py       # MiniLM reference embeddings for C# comparison
python distilbert_compare.py   # DistilBERT reference hidden states for C# comparison
python distilbert_sst_compare.py # DistilBERT SST-2 reference logits for C# comparison

python generate_input.py      # Regenerate shared comparison fixture
```

## Model architectures

### MobileNetV2

A lightweight classification network built from inverted residual blocks:

- **Stem**: 3×3 conv → BatchNorm → ReLU6
- **16 inverted residual blocks** with expansion/depthwise/project phases
- **Depthwise separable convolutions** (groups = input channels) for 3×3 layers
- **ReLU6** activation via `Clip(Relu(x), 0, 6)`
- **Residual shortcuts** only when `stride == 1 && inChannels == outChannels`
- **Head**: 1×1 conv → global avg pool → 1001-class linear classifier

Nivara modules used: `Conv2d<T>`, `BatchNorm2d<T>`, `Linear<T>`, `ReLU6` via `Clip` + `Relu`, depthwise grouped convolutions.

### ResNet-18

A standard 18-layer residual network:

- **Stem**: 7×7 conv → BatchNorm → ReLU → 3×3 MaxPool
- **4 stages** with channel progression: 64 → 128 → 256 → 512
- **BasicBlock**: two 3×3 convs with BatchNorm + ReLU, identity shortcut (or 1×1 conv when dimensions change)
- **Head**: global average pooling → 1000-class linear classifier
- **Downsampling** at stage boundaries via strided convolution in the shortcut path

Nivara modules used: `Conv2d<T>`, `BatchNorm2d<T>`, `Linear<T>`, `MaxPool2d<T>`, `AdaptiveAvgPool2d<T>`, residual addition via `ReverseGradOperations.Add`.

### MiniLM (sentence-transformers/all-MiniLM-L6-v2)

A 6-layer Post-LN BERT encoder producing 384-dimensional sentence embeddings:

- **Embedding stack**: token + position + segment embeddings summed, then LayerNorm
- **6× Post-LN BERT layers**: LayerNorm → Self-Attention → residual → LayerNorm → FFN → residual
- **GELU activation** in the FFN intermediate (exact erf)
- **Bidirectional self-attention** with optional padding mask (via `MultiheadAttention<T>`)
- **[CLS] token pooling** — extracts the first token's embedding from the output sequence
- **L2 normalization** — output embedding normalized to unit length for cosine similarity
- **Tokenization** via `Microsoft.ML.Tokenizers.BertTokenizer` (sample-only dependency)

Nivara modules used: `Embedding<T>` (Gather path), `LayerNorm<T>`, `Linear<T>`, `MultiheadAttention<T>`, `ReverseGradOperations.GeluExact`, `ReverseGradOperations.Add`.

### DistilBERT (distilbert-base-uncased)

The 6-layer, 768-dim pre-trained encoder (the baby-step before the fine-tuned SST-2 showcase):

- **Embedding stack**: word + position embeddings (no token-type embeddings) summed, then LayerNorm
- **6× Post-LN DistilBERT layers**: self-attention → residual → `sa_layer_norm` → FFN (`lin1` → GELU → `lin2`) → residual → `output_layer_norm`
- **GELU activation** in the FFN intermediate (exact erf)
- **Weight mapping** from `distilbert.*` SafeTensors keys via `DistilBertLoader.LoadEncoderWeights`
- **Verification**: `last_hidden_state` matches HuggingFace to `max abs diff 5e-6` (cosine 0.99999988)

Nivara modules used: `Embedding<T>`, `LayerNorm<T>`, `Linear<T>`, `BertSelfAttention<T>` (fused `ReverseGradOperations.MultiHeadAttention`), `ReverseGradOperations.GeluExact`, `ReverseGradOperations.Add`.

> **GELU note:** BERT-family models (MiniLM, DistilBERT) use the exact erf GELU (`GeluExact`). The tanh approximation (`ReverseGradOperations.Gelu`) matches HF `gelu_new`/GPT-2 and is retained for GPT-style `TransformerBlock`.

### DistilBERT SST-2 (distilbert-base-uncased-finetuned-sst-2-english)

The fine-tuned sequence-classification showcase: the base encoder plus a classification head that outputs binary sentiment logits.

- **Encoder**: identical to the base `distilbert` mode (word + position embeddings, 6 Post-LN layers, exact erf GELU)
- **No token-type embeddings** (`includeTokenTypeEmbedding: false`) — DistilBERT never feeds segment ids
- **Head**: `pre_classifier` (768→768) → **ReLU** → `classifier` (768→2). The HF architecture applies `nn.ReLU()` after `pre_classifier`; a naive port using `GeluExact` on the head produced logits off by ~0.05, so the head uses `ReverseGradOperations.Relu`
- **Softmax + argmax** for the sentiment label and confidence
- **Inference-default path**: `PredictLogits` runs outside any `Grad()` scope, producing leaf logits with no computation-graph overhead
- **Padded-input contract**: `BertEncoder.ForwardBatched` requires attention-mask tensors of length `batchSize * seqLen`; token IDs are passed as exact `int[]` (see the BFloat16 note) so they survive narrow-precision dtypes, and `PredictLogits` passes the padded `[maxLen]` token ids
- **Verification**: `compare` matches HuggingFace to `max abs logit diff 9.5e-7`, `argmax agreement 8/8`; the `bf16` mode matches the same reference at `8/8` argmax with a `max abs logit diff ~0.33` (genuine BFloat16 precision)

Nivara modules used: `DistilBertForSequenceClassification<T>` (shared from `Nivara.Samples`), `Embedding<T>`, `LayerNorm<T>`, `Linear<T>`, `BertSelfAttention<T>`, `ReverseGradOperations.GeluExact` (encoder FFN), `ReverseGradOperations.Relu` (head), `ReverseGradOperations.Softmax`, `ReverseGradOperations.MatMul`.

### SmolLM-135M-Instruct (HuggingFaceTB/SmolLM-135M-Instruct)

The **First causal LM / generative model** in the sample and the primary driver for the BF16 widening work
(`docs/BFLOAT16.md`). It is a **BF16-native** Llama-family causal LM —
all 272 on-disk tensors are `BF16` (269 MB), exercising the native
`SafeTensorsLoader.Read<BFloat16>` zero-hop path (unlike the other 4 models, which
are F32 on disk). The Nivara side runs the full stack in Nivara's AutoDiff engine
over the model ops below and greedily decodes a response.

- **Config**: `hidden_size=576`, `intermediate_size=1536`, 30 layers,
  `num_attention_heads=9`, **`num_key_value_heads=3` (GQA)**, `hidden_act=silu`
  (gated FFN), RMSNorm (`eps=1e-5`), RoPE (`theta=10000`),
  `max_position_embeddings=2048`, `vocab_size=49152`, `tie_word_embeddings=true`
- **Tokenizer**: **GPT-2 byte-level BPE** (not SentencePiece — see the note below),
  chat variant (`<|im_start|>`/`<|im_end|>` template; bos `<|im_start|>`,
  eos/pad `<|im_end|>`), 49152-token vocab built from `vocab.json` + `merges.txt`
- **Tied LM head**: the input embedding weight is reused as the output projection —
  the checkpoint has no separate LM-head tensors
- **Reference fixture**: `Python/smollm_generate_reference.py` saves the token-id
  stream and final-position logits for diffing:

  ```bash
  python samples/NivaraInference/Python/smollm_generate_reference.py
  # -> samples/data/compare_smollm_py.bin, samples/data/compare_smollm_logits_py.bin
  ```

  The C# `smollm generate` mode diffs against these when present (see "Causal-LM
  generation" below).

Nivara modules used: `RMSNorm<T>` (affine gamma, reuses the existing `RMSNormKernel<T>`),
`Activation.Silu` / `ReverseGradOperations.Silu` (forward + VJP) /
`ForwardGradOperations.Silu` (JVP) / `GradKernels.Silu{,Gradient}` (SIMD `TensorPrimitives`),
`RotaryEmbedding<T>` (RoPE, Llama `rotate_half` half-split layout) +
`GradKernels.RotaryForward/Backward`, `LlamaCausalAttention<T>` (GQA 9↔3 KV heads via
`ReverseGradOperations.GqaRepeatKV` + `GradKernels.HeadRepeat{,Backward}`), `LlamaDecoderBlock<T>`
(pre-norm attention + residual + pre-norm gated SiLU FFN + residual),
`ReverseGradOperations.MatMulTransposedB` (tied-embedding LM head). Sample-scoped
`samples/Nivara.Samples` counterparts (`LlamaForCausalLM<T>`, `LlamaConfig`, `LlamaLoader`,
`StateDictLoader`, `Gpt2BpeTokenizer`) are listed in "Sample-scoped additions" below.

**Usage:**

```bash
# Greedy causal-LM generation (F32, BF16 native on disk, or fp16)
dotnet run --project samples/NivaraInference -c Release -- smollm
dotnet run --project samples/NivaraInference -c Release -- smollm --precision bf16
dotnet run --project samples/NivaraInference -c Release -- smollm --precision fp16
```

Each run loads SmolLM-135M, tokenizes the fixed prompt *"The capital of France is"*,
greedily decodes up to 32 new tokens (inference-only: no `GradientUtils.Grad()` scope,
so no graph nodes are built), prints the token ids + decoded text, and — when the
PyTorch reference fixtures exist — diffs the token-id stream and final-position
logits.

**BF16 SIMD widening**: with scalar BFloat16 math, a 32-token generation is
impractical (~100× slower). The `smollm` mode therefore enables
`NivaraPrimitives.UseWidenSimd` for the narrow (BFloat16/Half) runs so the Phase-1
widen-compute-narrow SIMD kernels drive the matmuls (and restores the prior global
value afterwards, so other model modes are unaffected). A `--simd-widen` flag opts any
model into the widen path explicitly, `smollm benchmark` reports median-of-3
full-generation timing, and `smollm ab` runs the scalar-vs-widen side-by-side
comparison (see the A/B table under Performance benchmarks).

**Numerical caveats** (documented tolerance, not bit-exact): greedy argmax agreement
with the PyTorch reference is high but not perfect — F32 matches ~25/32 generated
tokens (decoded text is byte-identical through the first ~25; the tail diverges because
a small numeric difference at a near-tie flips argmax and the error compounds), and BF16
matches ~22/32 with a final-position-logits cosine similarity of ~0.94 vs the
reference (see the narrow-precision Results table below). This is the expected
"numeric precision diff" behavior for a single forward step, not a structural mismatch.

> **Tokenizer correction (historical)**: this README previously listed SmolLM's
> tokenizer as SentencePiece. It is actually a **GPT-2 byte-level BPE** tokenizer
> (`tokenizer_class: GPT2Tokenizer`, `add_prefix_space: false`). The
> `Microsoft.ML.Tokenizers` BPE path cannot reproduce SmolLM's byte-level token IDs
> (every pre-tokenizer variant diverges at space-prefixed tokens), so a sample-local
> `Gpt2BpeTokenizer` (HF `bytes_to_unicode` map + GPT-2 regex + ranked greedy merges)
> implements the reader.

#### Core library improvements (gaps found & filled by the 5th model)

Adding a causal-LM path surfaced capabilities the Nivara core library did not yet
have. SmolLM was the driver, but each gap was filled as a **reusable, unit-tested
addition to `src/Nivara`** (same forward + VJP + JVP + SIMD kernel shape as the ops the
first four models exercise), then verified end-to-end against the PyTorch reference:

- **New core module — `RMSNorm<T>`** (`src/Nivara/AutoDiff/Nn/RMSNorm.cs`): Llama RMS
  normalization with a per-channel affine `gamma` (Llama normalizes by row root-mean-square,
  unlike the existing mean/var LayerNorm). Reuses the existing `RMSNormKernel<T>` and the
  `RMSNormKernel.PerRowRMSNormForward/Backward` span kernels; forward + input-grad + gamma-grad
  all wired.
- **New core activation — SiLU (Swish, `x·sigmoid(x)`)**: `Activation.Silu` +
  `ReverseGradOperations.Silu` (forward + VJP), `ForwardGradOperations.Silu` (JVP,
  `t_out = silu'(a)·t_a`), and `GradKernels.Silu/SiluGradient` (SIMD `TensorPrimitives`
  chain). Llama's gated FFN gates on SiLU rather than GELU.
- **New core op — RoPE (`RotaryEmbedding<T>`)** (`src/Nivara/AutoDiff/Nn/RotaryEmbedding.cs`):
  precomputed cos/sin from `inv_freq = theta^{-2i/dim}`, Q/K rotary position embedding, with
  `GradKernels.RotaryForward/RotaryBackward` and the module's own graph op. **Layout bug found
  & fixed during verification**: the first implementation used the GPT-NeoX **interleaved-
  pairwise** rotation, but the Llama family uses HF **`rotate_half` (half-split)** — the wrong
  layout rotated Q/K so the logits were near-anti-correlated (cosine −0.92) and corrected to
  +0.24; the end-to-end F32 greedy match went 4/32 → 25/32 with byte-identical text through the
  matched prefix (the current F32 counts are in the Results tables below).
- **New core module — `LlamaCausalAttention<T>`** (`src/Nivara/AutoDiff/Nn/LlamaCausalAttention.cs`):
  **GQA** (9 Q / 3 KV heads) via KV-repeat — `ReverseGradOperations.GqaRepeatKV` (VJP with
  `GradKernels.HeadRepeat`/`HeadRepeatBackward`) and `ForwardGradOperations.GqaRepeatKV` (JVP) —
  feeding a fused, causal-masked per-head attention loop.
- **New core module — `LlamaDecoderBlock<T>`** (`src/Nivara/AutoDiff/Nn/LlamaDecoderBlock.cs`):
  pre-norm self-attention + residual, then pre-norm **gated SiLU FFN**
  (`down(silu(gate)⊙up)`) + residual.

**Sample-scoped additions** (not core library — `samples/Nivara.Samples` / `Program.cs`):
`LlamaForCausalLM<T>` (embed → 30 blocks → final RMSNorm → tied-embedding LM head),
`LlamaConfig : LLamaConfigLike`, `LlamaLoader.Load<TModel,TWeight>`, the
`StateDictLoader.LoadRMSNorm/LoadLinear` binding helpers, the `Gpt2BpeTokenizer` byte-level
BPE reader (see the tokenizer-correction note above), and enabling `UseWidenSimd` for the
BF16/Half narrow runs so generation is practical (see the BF16 section above).

### Qwen2.5-0.5B-Instruct (Qwen/Qwen2.5-0.5B-Instruct)

The **second causal LM / generative model** and this branch's headline showcase:
**native function calling** (`qwen tools` — the model emits and consumes a
`<tool_call>` mid-conversation, exactly like a hosted assistant) and
**teacher distillation** (`qwen distill` — an LLM-as-teacher labeling run that
trains a tiny sentiment MLP to match the teacher's tool-call classifications).
It is also the first model with a **real KV cache** (SmolLM decodes cache-free).

- **Config**: `hidden_size=896`, `intermediate_size=4864`, 24 layers,
  `num_attention_heads=14`, **`num_key_value_heads=2` (GQA 14↔2)**, SiLU gated
  FFN, RMSNorm (`eps=1e-6`), RoPE with **`theta=1_000_000`** (10× SmolLM's),
  `max_position_embeddings=32768`, `vocab_size=151936`, tied embeddings,
  **Q/K/V projections with bias** (the one additive `src/Nivara` gap — see below).
- **Tokenizer**: GPT-2 byte-level BPE with an added **`Split` regex
  pretokenizer** (`tokenizer.json` `pre_tokenizer` — the HF `Split` +
  `ByteLevel use_regex:false` composition, applied to the *raw* text until the
  byte map) and **added tokens** `<|im_start|>`/`<|im_end|>`/`<|tool_call|>`/
  `<|tool_response|>`/`<|tool_call_end|>`. See "Sample-scoped additions".
- **KV cache**: `LlamaKVCache<T>` + `ForwardCached` — each generated token runs
  only its own position through the 24 layers instead of re-feeding the whole
  growing sequence (SmolLM's cache-free loop). See the benchmark section.
- **Function calling**: headed-`<tool_call>` JSON with `getWeather`, parsed by
  `QwenToolParser` and fed back as `<tool_response>`; both rendered prompts are
  **byte-verified against the HF `apply_chat_template` fixture** (206 + 258 ids,
  MATCH) and the generated tool-call turn matches PyTorch **19/19 IDs**.
- **Distillation**: `distill` loads the teacher Qwen, classifies the 10 train
  sentences (FNV-1a word+bigram bag-of-words → 4096-dim features), caches the
  labels to `samples/data/qwen2.5-0.5b-instruct/qwen_distill_labels.json`
  (resumable; delete it / `--force` to re-annotate), then trains
  `SentimentMLP` (`Linear(4096→64)` + ReLU + `Linear(64→2)`) with
  `Adam<float>(1e-3)` + `CrossEntropyLoss` (~200 full-batch epochs inside
  `GradientUtils.Grad()`), and prints an eval table against a **linear-only
  baseline** and the real **DistilBERT SST-2** classifier.

**Usage:**

```bash
# Native function calling — the model issues a <tool_call>, the tool result is
# fed back, and the model answers from the observation (KV cache on by default)
dotnet run --project samples/NivaraInference -c Release -- qwen tools
dotnet run --project samples/NivaraInference -c Release -- qwen tools --no-kv-cache   # full re-forward each token

# Teacher distillation into the tiny sentiment classifier (labels cached/resumable)
dotnet run --project samples/NivaraInference -c Release -- qwen distill --teacher-examples 12
dotnet run --project samples/NivaraInference -c Release -- qwen distill --force

# Decode benchmark: median-of-3 tool-call turns, KV-cached vs full re-forward
dotnet run --project samples/NivaraInference -c Release -- qwen benchmark
# (qwen default load is the fused BF16->F32 read; --precision f32 is equivalent)
```

**Precision**: `f32` is the **default** for `qwen` — the checkpoint is
BF16-on-disk and the load **fuses** the SIMD `WidenBf16ToF32` widen directly into
each tensor's `float[]` (numerically identical weights; no interim `ushort[]`).
The 989 MB checkpoint loads in ~0.7–2.2 s on this machine in Release (OS file
cache drives the spread; median ~0.8 s warm). `bf16`/`fp16` are rejected for
`qwen` with a clear error (the generation loop is F32).

**Tools fixture diff** (run automatically when the checkpoint dir has the
reference files; `qwen_tool_*.txt`/`.bin` generated once by
`Python/qwen_tool_reference.py`):

| Check | Result |
|---|---|
| Rendered tool prompt ids (vs `qwen_tool_prompt_ids.bin`, 206) | MATCH |
| Rendered final prompt ids (vs `qwen_tool_final_prompt_ids.bin`, 258) | MATCH |
| Generated tool-call turn ids (vs Py, 19) | 19/19 |
| Final-position logits | maxAbs 0.399, cosine 0.999771, envelope YES |

The tool-call turn's greedy decode is **byte-exact** against PyTorch; the final
answer turn stays semantically correct ("partly cloudy", 18°C, northwest —
verified against the tool observation) though its greedy path can diverge at a
near-tie (25 vs Py's 23 tokens; documented tolerance, same tie-flip mechanism
as SmolLM).

#### Core library improvements (gaps found & filled by the 6th model)

- **Additive `src/Nivara` change — biased Q/K/V projections (#384)**: Qwen's
  attention projects **have bias** (`q_proj.bias` etc.), unlike SmolLM's.
  `LlamaCausalAttention<T>` gained a `qkvBias` ctor flag (default `false` —
  SmolLM semantics unchanged); `QProj/KProj/VProj` are built
  `bias: qkvBias`, `OProj` stays unbiased. Issue #384 tracks the gap.

**Sample-scoped additions** (`samples/Nivara.Samples` / `Program.cs`):
- **Qwen-style `Split`-regex pretokenizer + added-token merge in
  `Gpt2BpeTokenizer`**: when `tokenizer.json` declares a `Split` pretokenizer,
  its regex is applied to the RAW normalized text first and each chunk is
  byte-mapped after (matches HF `Split` + `ByteLevel(use_regex:false)`);
  tokenizer.json `added_tokens` (151643/151644/151645/151657/151658) are
  merged into the vocab and id-to-token map. (This same path fixes a
  renderer/byte-parity bug found during acceptance: the hand-rolled tool JSON
  opened 4 objects but closed only 3, so the prompt tokenized 205 vs the
  fixture's 206 — fixed to `}}}`.)
- **`SafeTensorsLoader.Read<float>` fused BF16→F32 (`WidenBf16ToF32`)**: the default
  load reads the BF16-on-disk weights and SIMD-widens them straight into each
  tensor's `float[]` via the vectorized `TensorPrimitives` kernel — no interim
  `ushort[]`, and the string-path load memory-maps the file so no full-file `byte[]`
  is materialized either (a one-pass read; ~1.88 GB peak managed heap for Qwen,
  ~0.94 GB below the old copy-into-`byte[]` load; ~1.1–1.4 s for the 989 MB file on
  this machine in Release [median ~1.3 s warm] — the mmap read trades ~0.5 s of
  warm-load time vs a raw `ReadAllBytes` copy for the ~1 GB managed-heap saving).
- **`LlamaLoader` qkvBias auto-detect**: presence of
  `model.layers.0.self_attn.q_proj.bias` flips the attention to biased
  projections (no manual flag).

**"Already generalized"** (reused from the SmolLM work, zero new core code):
`RotaryEmbedding<T>` `rotate_half` RoPE (only `theta` differs: 1e6), GQA
14↔2 KV-repeat (`GqaRepeatKV`), `RMSNorm<T>`, SiLU gated FFN, tied LM head,
`LlamaForCausalLM<T>`/`LlamaLoader`. **Out of scope**: fp16/bf16 compute for
qwen (rejected with a clear error), promoting the fused load / the Split-regex
tokenizer into `src/Nivara`, GGUF loading (Phase 5).

#### Results / benchmarks

| Run | Date | Result |
|---|---|---|
| `qwen tools` (default fused load) | 2026-09-06 | tool-call turn: 19 tok, 173,900 ms (~9,153 ms/tok cached); final turn: 25 tok, 227,800 ms; fixture ids MATCH (206/258), 19/19 generated ids (Release) |
| `qwen benchmark` | 2026-09-06 | KV cache median 189,150 ms vs full re-forward 204,385 ms (19 tok) → 1.1× (Debug build; Release run failed mid-decode — #386 will provide fresh numbers) |
| Load parse (fused, default) | 2026-09-06 | ~1.3 s median warm (1.1–1.4 s; Release; 290 tensors, 989 MB BF16-on-disk; memory-mapped, peak managed heap ~1.88 GB) — no interim `ushort[]` (+#388) and no full-file `byte[]` (+#392); the mmap read is ~0.5 s slower than a warm `ReadAllBytes` copy in exchange for the ~1 GB managed-heap saving |

**Why the KV speedup is small here**: the tool prompt is 206 tokens and the
generated turn just 19, so the cache-free path re-feeds only a slightly
longer sequence each step — the O(L²) growth hasn't compounded. The real KV
win appears at long contexts (hundreds of generated tokens); the per-token
cached decode and the same-turn full re-forward are quoted in the table below.

**Decode throughput** — measured on this machine (Intel Core Ultra 7 255H, 16
logical processors, .NET 11.0.0, Release build):

| Load precision | Load parse | KV-cached per-token | Full re-forward per-token | Speedup |
|---|---|---|---|---|
| f32 default (fused BF16→F32 SIMD widen) | ~1.3 s median warm (1.1–1.4 s) | 9,153 ms/tok (Release) | 10,757 ms/tok (Debug) | — (see note) |

<sup>Load parse is the fused memory-mapped `Read<float>` (Release build, 2026-09-06,
`qwen tools`): the 989 MB BF16-on-disk file memory-maps and SIMD-widens straight
into `float[]` with no interim `ushort[]` and no full-file `byte[]` (one pass,
~1.88 GB peak managed heap vs ~2.83 GB for a copy-into-`byte[]` load; the mmap
read trades ~0.5 s of warm-load time for that ~1 GB managed-heap saving —
physical working set is similar either way because the OS page cache holds the
file either way). The KV-cached per-token figure is Release; the
full re-forward per-token and the 1.1× KV speedup are Debug-era (benchmark chain).
The Release benchmark run failed mid-decode and will be refreshed via the
dedicated-machine cycle (issue #386), when both decode paths are re-measured in
the same Release build for a clean speedup ratio.</sup>

**Distill eval** (`qwen distill --teacher-examples 3`, 2026-09-06, Release
build; accuracy over the 8 shared SST-2 eval sentences, teacher labels via
the cached `qwen_distill_labels.json`):

| Model | Accuracy |
|---|---|
| Teacher (Qwen2.5-0.5B-Instruct, `classify_sentiment` tool) | 4/8 (50%) |
| Student (`SentimentMLP`, 3 teacher-labeled train rows) | 4/8 (50%) |
| Linear baseline (BOW 4096→2) | 4/8 (50%) |
| DistilBERT SST-2 (dedicated fine-tuned classifier) | 8/8 (100%) |

<sup>`--teacher-examples 12` labels the full 10-row train set; this README used 3
per a timing decision (each teacher classification ≈ 178–206 s on this machine,
the first ~25 min once, then cached/resumable). The honest read: the 0.5B
teacher's tool call defaults to "positive" on SST-2's subtle negatives (4/8 —
the same as always-positive), so this 3-row student mirrors the teacher's bias
and ties the linear baseline. The point is the *pipeline*: teacher → cache →
student training inside `GradientUtils.Grad()` → eval table. A larger teacher or
a balanced tool prompt would raise the bar; the tiny DistilBERT fine-tuned on
exactly this task remains the accuracy reference (8/8).</sup>

### Weight loading

Each model defines a static `LoadWeights()` factory that maps HuggingFace tensor names to Nivara module parameters. No reflection or generic deserialization — explicit, type-safe loading with full compile-time checking.

- **MobileNetV2**: 262 tensors mapped to 262 module parameters (Conv2d weight/bias, BatchNorm running mean/var/weight/bias, Linear weight/bias)
- **ResNet-18**: 102 tensors mapped to 102 module parameters
- **MiniLM**: 96 tensors mapped from HuggingFace keys like `encoder.layers.N.attention.self.query.weight` to Nivara `Linear<T>` weight/bias fields
- **DistilBERT**: 105 tensors mapped via `DistilBertLoader.LoadEncoderWeights` from `distilbert.embeddings.*` and `distilbert.transformer.layer.{0-5}.*` keys
- **DistilBERT SST-2**: 104 tensors — 102 encoder tensors via `DistilBertLoader.LoadEncoderWeights` + `pre_classifier.{weight,bias}` and `classifier.{weight,bias}` loaded via `DistilBertForSequenceClassification<T>.LoadWeights`
- **SmolLM-135M**: 272 tensors (all BF16 on disk) via `LlamaLoader.Load<TModel,TWeight>` + `LlamaConfig.FromJson` — maps `model.embed_tokens.weight` (reused for the tied LM head), `model.layers.N.*` (input_layernorm, self_attn.{q,k,v,o}_proj, post_attention_layernorm, mlp.{gate,up,down}_proj), and `model.norm.weight`, with RMSNorm/attention/MLP weights bound via `StateDictLoader.LoadRMSNorm`/`LoadLinear`
- **Qwen2.5-0.5B-Instruct**: 290 tensors (BF16 on disk, fused SIMD-widened to F32 at load via `Read<float>`) via the same `LlamaLoader` route, plus the biased attention arms — `model.layers.N.self_attn.{q,k,v}_proj.bias` (auto-detected from `model.layers.0.self_attn.q_proj.bias`), RoPE `theta=1_000_000`, GQA 14↔2

## Narrow-precision inference (BFloat16 / Half)

Every transformer model in this sample is generic over the compute dtype
`T : IFloatingPointIeee754<T>`, so it runs in F32, BFloat16, or Half (fp16).
A `--precision` argument selects the compute dtype (text models only; the
vision samples stay F32):

```bash
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst --precision bf16
dotnet run --project samples/NivaraInference -c Release -- distilbert --precision fp16
dotnet run --project samples/NivaraInference -c Release -- minilm --precision fp16
# SmolLM: native BF16 is the headline narrow mode; Half is unusable (see Results below)
dotnet run --project samples/NivaraInference -c Release -- smollm --precision bf16
```

`--precision` accepts `f32` (default), `bf16`, or `fp16`. A bare
`bf16` / `fp16` / `half` positional is also accepted (`-- distilbert_sst bf16`), so
the pre-#341 `bf16` invocations keep working unchanged.

**What a narrow-precision mode does**
- Loads the on-disk **F32** weights as `BFloat16` (`SafeTensorsLoader.Read<BFloat16>`) or
  `Half` (`SafeTensorsLoader.Read<Half>`) — the loader truncates each `float` to the
  narrow dtype at load time (analogous to PyTorch loading an F32 checkpoint into a
  `torch.bfloat16`/`torch.float16` model; the file on disk stays F32).
- Builds the `<BFloat16>` / `<Half>` model via the generic `LoadWeights<...>` and runs the
  full forward pass in that dtype.
- Diffs the output against the same PyTorch reference fixtures used by `compare` (logits for
  SST-2; normalized embeddings / L2 norms for MiniLM and DistilBERT).

**Token-ID correctness (the subtle bit)** — BFloat16 represents integers *exactly* only up to
256 and Half only up to 2048, but transformer vocabularies reach ~30k. Converting token IDs to a
narrow-precision tensor before the embedding lookup corrupts them (e.g. `30522 → 30512` in BF16),
sending the lookup to the wrong row and producing garbage (we measured a ~7.4 logit diff vs the
F32 reference before the fix). The fix keeps token IDs as **exact `int`**: `Embedding<T>`,
`BertEncoder<T>`, `MiniLMDistilled<T>` and `DistilBertForSequenceClassification<T>` all expose
`Forward(int[] tokenIds, ...)` overloads that look up embeddings by exact integer index,
independent of the compute dtype. Only the attention mask stays a narrow tensor (its `0`/`1`
values round-trip exactly). See `docs/BFLOAT16.md` for the engine-level details.

**Results** (against the F32 HuggingFace reference, CPU):

| Model | Metric | F32 vs Ref | BFloat16 vs Ref | Half (fp16) vs Ref |
|---|---|---|---|---|
| `distilbert_sst` | argmax agreement | 8/8 | **8/8** | **8/8** |
| `distilbert_sst` | max abs logit diff | ~1e-6 | **~0.33** | **~0.22** |

Half uses a 10-bit mantissa (vs BF16's 7), which is why its logits land closer to the F32
reference; both preserve every SST-2 prediction.

For **SmolLM**, the diff is evaluated on the 32-token greedy generation against the **BF16**-native
PyTorch reference (`smollm_generate_reference.py`), so the meaningful metrics are generated-token
argmax agreement and the final-position logits cosine:

| Model | Metric | F32 vs Ref | BFloat16 vs Ref | Half (fp16) vs Ref |
|---|---|---|---|---|
| `smollm` | generated-token argmax match | 25/32 | **22/32** | 0/32 |
| `smollm` | final-position logits cosine | 0.24* | **0.94** | NaN |

BF16 is the strongest numeric match (0.94 cosine, mean abs diff 3.63) and is the natural
native-on-disk choice. F32 comes close (25/32 tokens) but its *final*-position logits are compared
against a divergent suffix once the greedy streams part ways (\*), so the cosine is less meaningful.
**Half is unusable for SmolLM**: a merely 10-bit mantissa cannot hold the accumulating attention
geometry across 30 layers, so the final logits go NaN and the decode collapses to all-pad tokens
(0/32). In short: pick BF16 (native) or F32 — never Half — for SmolLM generation.

**Memory** — narrow precision stores each weight in 2 bytes (FP16/BF16) vs 4 for
F32, so weight memory **exactly halves** (same parameter count, half the bytes):

| Model | F32 weights | FP16 / BF16 weights |
|---|---|---|
| MiniLM | ~91 MB | ~45.5 MB |
| DistilBERT (base) | ~255.5 MB | ~127.8 MB |
| DistilBERT SST-2 | ~255.4 MB | ~127.7 MB |
| SmolLM-135M | ~513 MB (widened) | **~256.6 MB (native on disk)** |

**Speed** — `benchmark` now accepts `--precision` (all three dtypes), so you can time F32,
fp16, and bf16 inference for the same model in one generic code path (3 warmup + 10 timed
passes, avg/min/max ms, with params + weight MB reported):

```bash
# F32, fp16, bf16 MiniLM
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark --precision fp16
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark --precision bf16

# Same for distilbert / distilbert_sst (e.g. --precision fp16)
dotnet run --project samples/NivaraInference -c Release -- distilbert benchmark --precision fp16
```

Measured on CPU, MiniLM (seqLen 128), single thread:

| Precision | Avg ms/pass | Weight MB |
|-----------|-------------|-----------|
| F32       | ~142 | 86.6 |
| Half      | ~3658 | 43.3 |
| BFloat16  | similar to Half (see issue #363) | 43.3 |

The **halved weight memory** is the narrow-precision win; on CPU the narrow matmul runs
through non-SIMD fallbacks and is dramatically *slower* per pass than F32 (fp16 was ~26x
slower in the measurement above, issue [#363](https://github.com/khurram-uworx/Nivara/issues/363)).
Don't read narrow benchmarks as a CPU speed win — treat them as a memory trade that preserves
every prediction.

The base `distilbert` and `minilm` narrow-precision modes run correctly (unit-length
embeddings, sensible cosine similarities — e.g. 0.90 between "I love programming" and "I love
coding"). The column/tensor engine's BFloat16 path is documented in `docs/BFLOAT16.md`.

SmolLM differs from the other models: it is **BF16-native on disk**, so the `smollm
--precision bf16` mode reads the weights directly (no F32→BF16 truncation at load), and the
`smollm` F32 run *widens* those native BF16 weights instead. Its reference fixture (and thus the
generative diff above) is likewise loaded as `torch.bfloat16` on the PyTorch side, so the
apple-to-apple cross is **SmolLM BF16** (0.94 cosine). As with the SmolLM section above, the
BF16/Half runs enable `NivaraPrimitives.UseWidenSimd` (widen-compute-narrow SIMD kernels), yet —
as the Results table shows — Half's 10-bit mantissa still collapses to NaN logits and is not a
usable SmolLM precision.

**Reference fixtures for `compare` / narrow-precision diffs** — the quantitative cosine (or
logit) diff against the HuggingFace reference is shown only when the F32 reference `.bin` files
exist. They are **not checked into the repo** (they live in / beside the gitignored model-weight
directories, which hold the multi-hundred-MB checkpoints), but each has a local Python generator.
Run them once on-demand to enable the diffs:

```bash
# Base DistilBERT hidden states -> samples/data/distilbert/last_hidden_state_py.bin
python samples/NivaraInference/Python/distilbert_compare.py
# MiniLM embeddings -> samples/data/compare_minilm_embeddings_py.bin
python samples/NivaraInference/Python/minilm_compare.py
# DistilBERT SST-2 logits -> samples/data/compare_distilbert_sst_py.bin
python samples/NivaraInference/Python/distilbert_sst_compare.py
# SmolLM greedy token stream + final-position logits -> compare_smollm_py.bin /
# compare_smollm_logits_py.bin (BF16 native; pass --dtype float32 to compare F32 too)
python samples/NivaraInference/Python/smollm_generate_reference.py
```

Without a fixture the relevant mode prints "reference not found; skipping diff" and otherwise
runs normally (e.g. the SST-2 mode still prints each predicted label).

## SafeTensors loader

The sample includes a custom zero-dependency `SafeTensorsLoader` that parses the HuggingFace SafeTensors binary format directly:

- **Memory-mapped file loading** (`MemoryMappedFile` + `CreateViewAccessor` + `AcquirePointer`): the string-path loads memory-map the safetensors file and read each tensor's bytes directly from the mapped view — no full-file `byte[]` is materialized, so peak managed memory is just the widened tensors (~1.88 GB for Qwen, ~0.94 GB below the old copy-into-`byte[]` load). Parses the JSON header from the first 8 bytes + offset table via `System.Text.Json`.
- **Dtype support** — loads **F32** (native `float`), **F16** (`System.Numerics.Half`), and **BF16** (`System.Numerics.BFloat16`) tensors, converting each to the requested result type `T` via `T.CreateChecked`. Narrow on-disk dtypes are widened when `T` is wider (e.g. a BF16 checkpoint read as `float[]` widens losslessly), and a wider on-disk dtype is narrowed when `T` is `BFloat16` (e.g. the `bf16` run mode reads the on-disk F32 weights as `BFloat16`, truncating to genuine 7-bit mantissa). Any other dtype raises `NotSupportedException` with guidance.

## Performance benchmarks

Measured on the same machine (CPU-only, no GPU): Intel Core Ultra 7 255H
(16 logical processors), Nivara in Release mode, PyTorch with MKL-optimized kernels.
Both use batch size 1 with 3-pass warmup + 10 timed passes. Both columns were
recorded in the same session. Numbers vary with machine load — only the same-row
PyTorch-vs-Nivara ratio is meaningful.

| Model | Input | PyTorch (CPU) | Nivara (.NET) | Slowdown |
|-------|-------|---------------|-------------------|----------|
| **MobileNetV2** | 1×3×224×224 | 22 ms | 665 ms | **~30×** |
| **ResNet-18** | 1×3×224×224 | 14 ms | 251 ms | **~18×** |
| **MiniLM-L6** | 128 tokens | 11 ms | 64 ms | **~6×** |
| **DistilBERT** | 128 tokens | 35 ms | 185 ms | **~5×** |
| **DistilBERT SST-2** | 128 tokens | 35 ms | 184 ms | **~5×** |
| **SmolLM-135M** (F32 greedy gen) | 5 prompt + 32 new tokens | 1740 ms | 10976 ms | **~6×** |
| **SmolLM-135M** (BF16 greedy gen) | 5 prompt + 32 new tokens | 1778 ms | 16792 ms | **~9×** |

*Recorded 2026-09-01 — Intel Core Ultra 7 255H, 16 logical processors, Nivara .NET 11.0.0, PyTorch 2.13.0+cpu. Transformer rows: 128-token single forward pass (3 warmup + 10 timed), except SmolLM which is one 32-token greedy generation (median of 3 runs, both sides same-dtype CPU). SmolLM F32 = BF16 checkpoint widened to F32 (513.1 MB); SmolLM BF16 = BF16-native on disk (256.6 MB).*

The SST-2 row reuses the DistilBERT PyTorch timing (same architecture, only the
weights differ; `Python/distilbert_sst_compare.py` is accuracy-only, no timing).
PyTorch vision is multi-threaded MKL; Nivara's conv kernels are single-threaded
naive loops, which widens the vision gap on this low-power 4-core CPU — the
transformer gap (~6×) is the more representative figure on this machine.

The **SmolLM rows** report one full 32-token greedy generation (not a single forward
pass) on both sides on CPU, as a **median of 3 runs** in the same session. PyTorch's
`model.generate` uses a **KV cache** and incremental decoding, while Nivara's `smollm`
greedy loop is a **naive, cache-free decode** that re-feeds the whole growing sequence
through all 30 layers each step (O(L²) in sequence length — the per-token time grows as
the sequence lengthens), so the ratio is best read as "naive no-KV-cache vs KV-cached"
rather than a pure kernel comparison. The underlying per-forward-step transformer gap is
the same ~6× family as the MiniLM/DistilBERT rows.

**Memory vs performance (SmolLM F32 vs BF16, both Nivara, same 32-token generation):**

| Precision | Weights | Nivara | vs F32 |
|---|---|---|---|
| F32 (widened) | 513.1 MB | 10976 ms | — |
| BF16 (native on disk) | 256.6 MB | 16792 ms | **~1.5× slower** |

BF16 halves the weight memory (256.6 MB vs 513.1 MB), but on CPU it is **not** faster —
the F32 path runs fully-optimized native `float` SIMD kernels while BF16 still pays the
widen/widen-back overhead, so F32 is actually ~1.5× *faster* for generation here. The
takeaway: on CPU, use BF16 only when you need the halved memory footprint; if you have the
~513 MB headroom, F32 gives both faster generation and better numerical fidelity. (The
BF16 native load also skips the F32→BF16 truncation the other models' narrow modes do —
on disk SmolLM is already BF16.)

**BF16 scalar-fallback vs widen SIMD A/B (`smollm --precision bf16 ab`, 2026-09-01, same machine):**

| Mode | ms/token | Full gen (32 tokens) | vs |
|---|---|---|---|
| BF16 scalar fallback (`UseWidenSimd = off`) | 7,032 | 225,037 ms | — |
| BF16 widen (`UseWidenSimd = on`, default for narrow) | 705 | 22,591 ms | **~10× faster** |
| F32 native (control) | 333 | 10,660–10,926 ms | widen transparent |

The `--simd-widen` flag toggles the widen path from the CLI; for narrow models it is enabled
by default (without it, BF16 matmul falls back to the ~26–100×-slower scalar dot). The F32
control confirms the toggle is a no-op for `float` (identical token streams, 32/32). Reading
the table together with the memory-vs-performance table above: BF16 **widen** is ~10× faster
than BF16 **scalar** and roughly **1.5–2× slower than F32 native** (the exact F32-vs-BF16
ratio varies with machine load — ~1.5× in the memory-vs-performance table above, captured
at a cooler baseline, ~2× under sustained load) — the widen path restores usable BF16
performance (memory-halving convenience) while remaining slower than native F32 compute.

AutoDiff graph nodes are only created inside `GradientUtils.Grad()` scopes (used by `TrainingLoop` and manual training code). Inference passes outside `Grad()` produce leaf tensors with no computation graph overhead. The AutoDiff refactor closed most of the gap: on the 2026-08-04 machine it cut vision inference ~4× (MobileNetV2 ~2,254 ms → ~563 ms, ResNet-18 ~641 ms → ~263 ms) and transformers ~1.5× (MiniLM ~110 → ~73 ms, DistilBERT ~186 → ~164 ms, SST-2 ~232 → ~187 ms). The vision gap is dominated by convolution kernels (especially depthwise convolutions in MobileNetV2), which use naive nested loops — ResNet-18 benefits from fewer depthwise layers. Transformer inference runs on a transpose-free path: `Linear` passes the raw weight `[out, in]` directly to the kernel's transposed-B matmul (no per-forward weight transpose), bias is applied via a row-broadcast `AddBias` op, op results are wrapped without a copy, and LayerNorm/Gelu/GeluExact skip saved-state allocations when gradients are not tracked. Attention runs through the fused `ReverseGradOperations.MultiHeadAttention` kernel (#86): heads are packed once per forward and QK^T/softmax/PV run as a single per-head pass over `TensorPrimitives` row kernels with no per-head `Slice`/`Transpose` graph nodes, keeping DistilBERT encoder inference at ~508 ms on this laptop.

## Sample data

| File | Purpose |
|------|---------|
| `samples/data/mobilenet_v2/model.safetensors` | MobileNetV2 weights (~13.5 MB) |
| `samples/data/resnet18/model.safetensors` | ResNet-18 weights (~44.6 MB) |
| `samples/data/minilm/model.safetensors` | MiniLM weights (~87 MB) |
| `samples/data/minilm/config.json` | MiniLM BERT config |
| `samples/data/minilm/vocab.txt` | MiniLM wordpiece vocabulary |
| `samples/data/distilbert/model.safetensors` | DistilBERT weights (~255.5 MB, 105 tensors) |
| `samples/data/distilbert/config.json` | DistilBERT config |
| `samples/data/distilbert/vocab.txt` | DistilBERT wordpiece vocabulary |
| `samples/data/distilbert/last_hidden_state_py.bin` | PyTorch reference hidden states (generated by `Python/distilbert_compare.py`) |
| `samples/data/distilbert_sst/model.safetensors` | Fine-tuned DistilBERT SST-2 weights (~255.4 MB, 104 tensors) |
| `samples/data/distilbert_sst/config.json` | DistilBERT SST-2 config (`dim=768`, `n_layers=6`, `n_heads=12`, 2 labels) |
| `samples/data/distilbert_sst/vocab.txt` | DistilBERT wordpiece vocabulary |
| `samples/data/compare_distilbert_sst_py.bin` | PyTorch reference logits + softmax probs (generated by `Python/distilbert_sst_compare.py`) |
| `samples/data/smollm-135m/model.safetensors` | SmolLM-135M-Instruct weights (~269 MB, 272 tensors, all BF16) |
| `samples/data/smollm-135m/config.json` | SmolLM-135M config (`hidden=576`, `n_layers=30`, GQA 9/3, SiLU, RoPE) |
| `samples/data/smollm-135m/tokenizer.json` | SmolLM tokenizer (GPT-2 byte-level BPE; `<|im_start|>`/`<|im_end|>` chat template) |
| `samples/data/compare_smollm_py.bin` | PyTorch reference token-id stream (generated by `Python/smollm_generate_reference.py`) |
| `samples/data/compare_smollm_logits_py.bin` | PyTorch reference final-position logits (generated by `Python/smollm_generate_reference.py`) |
| `samples/data/compare_input.bin` | Shared `[1,3,224,224]` input for compare modes (generated by `Python/generate_input.py`) |
| `samples/data/images/` | Synthetic test images at various resolutions (created by `Python/create_images.py`) |
| `samples/data/qwen2.5-0.5b-instruct/model.safetensors` | Qwen2.5-0.5B-Instruct weights (~989 MB, 290 tensors, BF16 on disk) |
| `samples/data/qwen2.5-0.5b-instruct/config.json` | Qwen2.5 config (`hidden=896`, 24 layers, GQA 14/2, `theta=1e6`, max pos 32768) |
| `samples/data/qwen2.5-0.5b-instruct/tokenizer.json` | Qwen tokenizer (GPT-2 BPE + `Split` regex pretokenizer + added tokens) |
| `samples/data/qwen2.5-0.5b-instruct/vocab.json`, `merges.txt` | Qwen BPE vocabulary / merges (151,936-token vocab) |
| `samples/data/qwen2.5-0.5b-instruct/qwen_tool_prompt.txt`, `qwen_tool_prompt_ids.bin` | Tool-prompt fixture (869 chars → 206 ids; generated by `Python/qwen_tool_reference.py`) |
| `samples/data/qwen2.5-0.5b-instruct/qwen_tool_final_prompt.txt`, `qwen_tool_final_prompt_ids.bin` | Final-prompt fixture (258 ids) |
| `samples/data/qwen2.5-0.5b-instruct/qwen_tool_ids_py.bin` | PyTorch generated tool-call turn ids (42 = 19 tool + 23 final) |
| `samples/data/qwen2.5-0.5b-instruct/qwen_tool_logits_py.bin` | PyTorch final-position logits reference |
| `samples/data/qwen2.5-0.5b-instruct/qwen_distill_labels.json` | Resumable teacher-label cache (runtime-generated; gitignored model dir) |
| `samples/data/qwen-distill/*.bin` | Torch-parity fixtures for the student MLP (committed, 9 files, ~2.16 MB; generated by `Python/qwen_distill_reference.py`) |

## Nivara capabilities exercised

### Vision models

| Capability | Where exercised |
|---|---|
| `Conv2d<T>` with asymmetric padding, grouped convs, 1×1 fast path | All conv layers in both models |
| `BatchNorm2d<T>` with running statistics | Every conv → BN block |
| `MaxPool2d<T>` with argmax | ResNet-18 stem |
| `AdaptiveAvgPool2d<T>` with gradient broadcast | Both model heads |
| `Linear<T>` with MatMul + bias | Classifier heads |
| `Module<T>` tree with `LoadStateDict` | Full model construction |
| Depthwise separable convolutions (groups = channels) | MobileNetV2 3×3 blocks |

### MiniLM (text)

| Capability | Where exercised |
|---|---|
| `Embedding<T>` Gather-based lookup | Token/position/segment embeddings |
| `LayerNorm<T>` with affine parameters | After embedding, after each attention and FFN |
| `MultiheadAttention<T>` bidirectional mode, padding mask | 6 attention layers |
| `ReverseGradOperations.GeluExact` | FFN intermediate activation (exact erf) |
| `ReverseGradOperations.Add` (residual) | Every residual connection |
| `Module<T>.Eval()` | Inference mode (disables dropout) |
| `Microsoft.ML.Tokenizers` integration | BERT WordPiece tokenizer |

### DistilBERT (text)

| Capability | Where exercised |
|---|---|
| `Embedding<T>` without token-type embeddings | `includeTokenTypeEmbedding: false` |
| `BertSelfAttention<T>` padding-mask path | 6 attention layers (768-dim, 12 heads) |
| `ReverseGradOperations.GeluExact` | FFN intermediate activation (exact erf) |
| `DistilBertLoader.LoadEncoderWeights` | `distilbert.*` SafeTensors weight mapping |

### DistilBERT SST-2 (text classification)

| Capability | Where exercised |
|---|---|
| `DistilBertForSequenceClassification<T>` | Shared classifier model (`pre_classifier` → ReLU → `classifier`) |
| `ReverseGradOperations.Relu` | Classification-head activation (matches HF `nn.ReLU`) |
| `GradientUtils.Constant` | Padded token-id / attention-mask input tensors |
| `GradientUtils.Grad()`-free inference | Leaf logits, no computation graph overhead |
| `MiniLMTokenizer.Encode` + `Microsoft.ML.Tokenizers.BertTokenizer` | WordPiece tokenization with `[CLS]`/`[SEP]` |
| Softmax + argmax via tensor span | Sentiment label + confidence |

### SmolLM-135M-Instruct (causal LM / generation)

| Capability | Where exercised |
|---|---|
| `RMSNorm<T>` affine gamma | Pre-norm in every decoder block + final norm |
| `Activation.Silu` (forward/VJP/JVP) | Gated SiLU FFN gate path |
| `RotaryEmbedding<T>` (RoPE, `rotate_half`) | Q/K rotary position embeddings |
| `LlamaCausalAttention<T>` + `GqaRepeatKV` | GQA self-attention (9 Q / 3 KV) |
| `LlamaDecoderBlock<T>` | Pre-norm attention + gated SiLU FFN + residuals |
| `LlamaForCausalLM<T>` + tied LM head | Embed → blocks → final norm → `hidden @ embed^T` |
| `Gpt2BpeTokenizer` | Sample-local GPT-2 byte-level BPE tokenization |
| `NivaraPrimitives.UseWidenSimd` | SIMD widen-compute-narrow BF16 matmul (native path) |
| Greedy generation (inference-default) | 32-token decode, no `GradientUtils.Grad()` scope |
| `smollm ab` A/B + `smollm benchmark` | Scalar-vs-widen comparison; median-of-3 generation timing |

### Qwen2.5-0.5B-Instruct (causal LM / function calling / distillation)

| Capability | Where exercised |
|---|---|
| `LlamaCausalAttention<T>` `qkvBias` (biased Q/K/V) | 24 biased-attention layers (14 Q / 2 KV heads, GQA) |
| `LlamaKVCache<float>` + `ForwardCached` | Per-token cached decode (each new token runs only its position) |
| `LlamaForCausalLM<T>` greedy decode (inference-default) | Tool-call turn (19 tok) + answer turn, no `GradientUtils.Grad()` scope |
| Function-calling loop (`QwenToolParser` + `<tool_call>`/`<tool_response>`) | `getWeather` → tool result fed back → final answer |
| `Gpt2BpeTokenizer` `Split`-regex pretokenizer + added tokens | Qwen tokenizer path (byte-verified against the HF fixture, 206/258 ids) |
| `SafeTensorsLoader.Read<float>` fused BF16→F32 (`WidenBf16ToF32`) | Default load (BF16-on-disk → F32, memory-mapped, no interim `ushort[]`/`byte[]`, ~1.3 s median warm Release, ~1.88 GB peak managed) |
| Teacher distillation inside `GradientUtils.Grad()` | `SentimentMLP` 200-epoch training + linear baseline vs DistilBERT SST-2 eval table |
| FNV-1a word+bigram feature hashing (4096-dim BOW) | Student/linear input features from raw sentences |
| Resumable label cache + `--force` | `qwen_distill_labels.json` merge/recompute |

## Release Benchmark

Run this during release prep (step 5 of `RELEASING.md`). Requires Python, PyTorch,
and HuggingFace model weights (see Quick start for `hf download` commands).

Run both sides in the same session for fair comparison:

```powershell
# Nivara (C#) — one pass per model
dotnet run --project samples/NivaraInference -c Release -- mobilenet_v2 benchmark
dotnet run --project samples/NivaraInference -c Release -- resnet18 benchmark
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst benchmark
# SmolLM: benchmark (median-of-3 full generation timing) or generate (full diff):
dotnet run --project samples/NivaraInference -c Release -- smollm benchmark                     # F32
dotnet run --project samples/NivaraInference -c Release -- smollm --precision bf16 benchmark   # native BF16 (widen)
dotnet run --project samples/NivaraInference -c Release -- smollm --precision bf16 ab          # A/B scalar vs widen
dotnet run --project samples/NivaraInference -c Release -- qwen benchmark                      # KV-cached vs full re-forward (median-of-3)

# PyTorch (Python) — run immediately after on the same machine
cd samples/NivaraInference/Python
python minilm_benchmark.py
python distilbert_benchmark.py
# SmolLM: generate the reference stream AND report PyTorch generation timing.
# Run each dtype at least 3x and take the median for the SmolLM table rows.
python smollm_generate_reference.py --dtype float32   # F32 (widened) row
python smollm_generate_reference.py --dtype bfloat16  # native BF16 row
```

For SmolLM, record the `Generated N tokens in ... ms (.. ms/token)` line from each side
(Nivara prints it from `smollm` / `smollm --precision bf16`; PyTorch prints it from
`smollm_generate_reference.py --dtype <dtype>`). Run each side ~3x and take the **median**
(the last four rows are single-forward-pass timings; the SmolLM rows are a full
32-token generation — see the note under the Performance benchmarks table). Keep both
dtype pairs same-dtype on CPU so the ratio and the F32-vs-BF16 memory/performance tradeoff
are meaningful.

**Update the Performance benchmarks table:**
1. Shift existing timing columns to **Prev (PyTorch)** / **Prev (Nivara)**.
2. Place fresh measurements in **Current (PyTorch)** / **Current (Nivara)**.
3. Add **Prev Slowdown** (old ratio) and **Current Slowdown** (new ratio).
   Alternatively, keep single columns and add a **Δ%** column for Nivara only.
4. Update the machine line, recording date, and prose referencing ratios.
