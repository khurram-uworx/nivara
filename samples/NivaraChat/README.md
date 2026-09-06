# NivaraChat — Hybrid Agent Workflow

A sample project demonstrating Nivara-trained domain-specific models as first-class participants in a `Microsoft.Agents.AI.Workflows` graph, mixed with an Ollama-backed `ChatClientAgent` node. This is the showcase example for Nivara's value proposition: **deterministic, lightweight, fast models working alongside an LLM in a production workflow.**

**Target audience:** .NET developers building AI workflows, integrating ML models into agent pipelines, exploring hybrid deterministic + stochastic architectures.

## What it does

NivaraChat trains four small domain-specific models (sentiment classifier, entity extractor, workflow validator, agents validator) and wires them into a workflow graph. Two execution paths are available:

- **`--workflow`** — classic executor-based graph with fan-out/fan-in topology
- **`--agents` / `--interactive`** — each model wrapped as an `IChatClient` via `NivaraChatClient`, participating as `ChatClientAgent`s through `AsAIAgent()`

With `--ollama`, an Ollama-backed LLM agent is appended after the validator for fluent response generation.

## Quick start

```bash
# Train all four models (overwrites existing)
dotnet run --project samples/NivaraChat -- --train

# Run workflow (Nivara nodes only — no LLM needed)
dotnet run --project samples/NivaraChat -- --workflow

# Single-shot test
dotnet run --project samples/NivaraChat -- --workflow --text "I love this product!"

# Multi-word entity examples
dotnet run --project samples/NivaraChat -- --workflow --text "John Smith from Acme Corp reported great work on January 15"
dotnet run --project samples/NivaraChat -- --workflow --text "Acme Corp in New York announced on March 3"

# Run workflow with Ollama LLM
dotnet run --project samples/NivaraChat -- --workflow --ollama --model llama3.2

# Agents mode (Nivara-only, single-shot)
dotnet run --project samples/NivaraChat -- --agents --text "Jane Doe at TechStart Inc reported issues in San Francisco"

# Agents mode with Ollama LLM
dotnet run --project samples/NivaraChat -- --agents --ollama --text "Acme Corp in New York announced on March 3"

# Interactive agents mode (Nivara-only)
dotnet run --project samples/NivaraChat -- --interactive

# Interactive agents mode with Ollama LLM
dotnet run --project samples/NivaraChat -- --interactive --ollama

# Confidence handoff — Nivara decides if LLM is needed (requires --ollama)
dotnet run --project samples/NivaraChat -- --handoff --ollama --text "I love this product!"
dotnet run --project samples/NivaraChat -- --handoff --ollama --text "This product is interesting but I'm not sure"

# Tool calling — LLM orchestrates Nivara models as AIFunction tools (requires --ollama)
dotnet run --project samples/NivaraChat -- --tools --ollama --text "John Smith from Acme Corp reported great work"

# Writer-critic loop — LLM writes, Nivara scores, retry if poor (requires --ollama)
dotnet run --project samples/NivaraChat -- --critic --ollama --text "Explain quantum computing to a 5-year-old"

# Embedding search (index documents, retrieve context via IEmbeddingGenerator)
dotnet run --project samples/NivaraChat -- --embed

# RAG pipeline: chunk docs, retrieve context, LLM generate answer (requires --ollama)
dotnet run --project samples/NivaraChat -- --rag --ollama --text "How does embedding search work?"

# RAG agent: same with TextSearchProvider auto-context injection (requires --ollama)
dotnet run --project samples/NivaraChat -- --rag-agent --ollama --text "What is NivaraChat?"

# Online learning from LLM feedback (requires --ollama)
dotnet run --project samples/NivaraChat -- --online-learning --ollama

# TinyShakespeare — batched transformer served as an IChatClient (no LLM needed)
dotnet run --project samples/NivaraChat -- --tinyshakespeare

# Smoke run: smaller vocab is ~3x faster (full options: --tinyshakespeare --help)
dotnet run --project samples/NivaraChat -- --tinyshakespeare --vocab-size 1200 --prompt "ROMEO:"

# SmolLM — serve the pretrained SmolLM-135M-Instruct causal LM as an IChatClient (no LLM needed)
# Model files must be present under samples/data/smollm-135m (see "SmolLM" section below)
dotnet run --project samples/NivaraChat -- --smollm plain --text "The capital of France is"

# Interactive multi-turn chat, token-streamed (full options: --smollm --help)
dotnet run --project samples/NivaraChat -- --smollm chat

# Qwen — native function calling with Qwen2.5-0.5B-Instruct:
# Model emits <tool_call>, GetWeather runs in-process, result feeds back, clean final answer
dotnet run --project samples/NivaraChat -- --qwen tools-weather --text "What's the weather in Paris?"

# Plain transcript streaming (full options: --qwen --help)
dotnet run --project samples/NivaraChat -- --qwen plain --text "The capital of France is"
```

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--train` | — | Mode: train all four models (overwrites existing) |
| `--workflow` | — | Mode: executor-based workflow pipeline with fan-out/fan-in |
| `--interactive` | — | Mode: agents pipeline with live interactive input |
| `--agents` | — | Mode: same as `--interactive`, supports `--text` for single-shot |
| `--handoff` | — | Mode: confidence-based handoff — Nivara decides if LLM is needed |
| `--tools` | — | Mode: LLM orchestrator calls Nivara models as AIFunction tools |
| `--critic` | — | Mode: writer-critic loop — LLM writes, Nivara scores, retry if poor |
| `--embed` | — | Mode: embedding search — index documents, retrieve context via `IEmbeddingGenerator` |
| `--rag` | — | Mode: RAG pipeline — chunk markdown docs, retrieve via vector search, LLM generates answer |
| `--rag-agent` | — | Mode: RAG agent — same as `--rag` with `TextSearchProvider` auto-context injection |
| `--online-learning` | — | Mode: online learning from LLM feedback — incremental retrain with validated examples |
| `--intent-train` | — | Mode: train intent classifier (5 classes) |
| `--intent` | — | Mode: intent routing — classify input and route to specialist executor |
| `--tinyshakespeare` | — | Mode: train/serve a batched TinyShakespeare transformer as `IChatClient` (see `--tinyshakespeare --help` for its options) |
| `--smollm` | — | Mode: serve the pretrained SmolLM-135M-Instruct causal LM as `IChatClient` (`chat`/`plain` sub-modes; see `--smollm --help`) |
| `--qwen` | — | Mode: native function calling with Qwen2.5-0.5B-Instruct (`tools-weather`/`chat`/`plain` sub-modes; see `--qwen --help`) |
| `--text <message>` | — | Single-shot: run pipeline on one message and exit |
| `--ollama [url]` | — | Flag: enable Ollama LLM agent (optional URL, default: `http://localhost:11434`) |
| `--model <name>` | `llama3.2` | Ollama model name |
| `--threshold <float>` | `0.8` | Confidence threshold for `--handoff` mode |
| `--docs-dir <path>` | `docs/` | Documents directory for `--rag` and `--rag-agent` modes |
| `--top-k <int>` | `3` | Number of chunks to retrieve for RAG modes |

## Modes of use

### Training (`--train`)
Trains all four models on synthetic data: sentiment classifier, entity extractor, workflow validator, and agents validator. Each model follows the same pattern: generate data → tokenize → build frame → train with `TrainingLoop<T>` → save with `ModelSerializer`. No external datasets required.

### Workflow (`--workflow`)
Classic executor-based pipeline with fan-out/fan-in topology. `TextRouter` broadcasts input to `SentimentExecutor` and `EntityExtractor` in parallel; results merge at `ValidatorExecutor` via barrier. Without `--ollama`, runs Nivara nodes only. With `--ollama`, appends an LLM agent after the validator.

### Agents (`--agents`)
Sequential pipeline where each trained model is wrapped as an `IChatClient` via `NivaraChatClient` and participates as a `ChatClientAgent`. Supports `--text` for single-shot execution. With `--ollama`, an Ollama LLM agent is appended after the validator.

### Interactive (`--interactive`)
Same as `--agents` but with live input. Type `quit` to exit. With `--ollama`, the LLM agent is appended after the validator.

### Confidence handoff (`--handoff`)
Demonstrates the hybrid deterministic/stochastic pattern. Nivara models run first; if both sentiment and entity extraction are confident (>= `--threshold`, default 0.8), the result is returned without calling the LLM. If either is uncertain, the partial Nivara results are forwarded to the LLM for enrichment. Requires `--ollama`.

```
Input text
    │
    v
[TextRouter] --fan-out--> [SentimentExecutor, EntityExtractor]
                               │
                          fan-in barrier
                               │
                               v
                        [ConfidenceRouter]
                         /           \
              confident (>=0.8)    uncertain (<0.8)
                    │                    │
                    v                    v
            Nivara result          [Ollama LLM]
```

Tested examples:

| Input | Threshold | Path taken | Why |
|-------|-----------|------------|-----|
| `"I love this product!"` | 0.8 (default) | LLM | Sentiment 0.58, entity 0.27 — both below threshold |
| `"I love this product!"` | 0.7 | LLM | Entity confidence 0.27 still below 0.7 |
| `"John Smith from Acme Corp reported great work on January 15"` | 0.8 | LLM | Entity confidence 0.798 just below 0.8 |
| `"John Smith from Acme Corp reported great work on January 15"` | 0.7 | Nivara only | Both above 0.7 — no LLM needed |

The entity model's average per-token confidence tends to cap around 0.8 for multi-entity inputs. Use `--threshold 0.7` for a more practical cutoff.

### Tool calling (`--tools`)
Flips the architecture: the LLM *decides* when to call Nivara models. Nivara models are wrapped as `AIFunction` tools via `AIFunctionFactory` with `[Description]` attributes. The LLM receives tool definitions and chooses when to invoke sentiment analysis, entity extraction, or response validation. Requires `--ollama`.

Tested examples:

| Input | Tools called | Notes |
|-------|-------------|-------|
| `"John Smith from Acme Corp reported great work"` | ExtractEntities, AnalyzeSentiment | LLM chose both tools, summarized results |
| `"Acme Corp in New York announced on March 3"` | ExtractEntities, AnalyzeSentiment | Multi-entity extraction works well |

The LLM decides which tools to call based on the `[Description]` attributes. Tool results are fed back automatically by the `ChatClientAgent` framework.

### Writer-critic loop (`--critic`)
The LLM generates a response, a Nivara validator model scores it for quality/consistency, and the LLM re-generates if the score is below threshold. Bounded to 3 iterations with structured feedback. Demonstrates Nivara models evaluating LLM output, not just generating their own. Requires `--ollama`.

Tested examples:

| Input | Result | Score |
|-------|--------|-------|
| `"Explain quantum computing to a 5-year-old"` | PASS on attempt 1 | 0.98 |

The validator model was trained on `"original || response"` format for consistency checking. High scores indicate the response is consistent with the query. Max 3 iterations — if all fail, the last attempt is returned with a notice.

### Embedding search (`--embed`)
Indexes 8 knowledge documents using `IEmbeddingGenerator` backed by a local MiniLM transformer, then runs an interactive REPL. Type a query and the system retrieves the top-4 most relevant documents ranked by cosine similarity. This demonstrates the retrieval step for RAG (Retrieval-Augmented Generation) — in a full pipeline, retrieved context would be injected into the LLM prompt via `TextSearchProvider`. Uses `NivaraEmbeddingGenerator<string>` from `Nivara.Extensions`, the same interface as OpenAI/Ollama embedding providers.

### RAG pipeline (`--rag`)
Full Retrieval-Augmented Generation pipeline. Loads real Nivara documentation (markdown files from `docs/` + `README.md`), chunks them into paragraphs, indexes via `InMemoryVectorStore` with auto-embedding from the local MiniLM model, then runs an interactive REPL. User questions are matched against stored chunks via cosine similarity, top-K chunks are injected into a manually constructed prompt, and the LLM generates a grounded answer. Shows retrieval time and LLM time separately. Requires `--ollama`.

```
Documents (docs/*.md, README.md)
    │
    v
ChunkText (paragraph splitting, ~500 chars)
    │
    v
InMemoryVectorStore + MiniLMEmbeddingGenerator
    │  auto-embeds each chunk via Nivara IEmbeddingGenerator
    v
User query
    │
    v
collection.SearchAsync(query, top: K)  →  ranked chunks
    │
    v
Manual prompt: "Answer based on context:\n{chunks}\n\nQuestion: {query}"
    │
    v
Ollama LLM  →  grounded response
```

Tested examples:

| Input | Top-K | Retrieval | LLM | Answer quality |
|-------|-------|-----------|-----|----------------|
| `"How does embedding search work?"` | 3 | 371ms | 27.5s | Describes MiniLM → InMemoryVectorStore → cosine similarity pipeline |
| `"What is NivaraChat?"` | 3 | 538ms | 17.2s | Correctly identifies RAG pipeline with MiniLM + TextSearchProvider |

Uses: `MiniLMEmbeddingGenerator.Create()`, `InMemoryVectorStore`, `DocumentChunker.ChunkText()`, `collection.SearchAsync()`.

### RAG agent (`--rag-agent`)
Same retrieval pipeline as `--rag`, but uses `TextSearchProvider` from the Agent Framework for automatic context injection instead of manual prompt construction. `TextSearchProvider` intercepts each LLM call, performs a search, and injects the retrieved context before the LLM sees the query. This is the standard ecosystem pattern for RAG and composes with other `AIContextProvider` implementations. Requires `--ollama`.

```
Documents (same as --rag)
    │
    v
InMemoryVectorStore + MiniLMEmbeddingGenerator
    │
    v
TextSearchProvider (SearchTime = BeforeAIInvoke)
    │  auto-searches before every LLM call
    │  injects top-K chunks as additional context
    v
ChatClientAgent + Ollama LLM
    │
    v
Grounded response with source citations
```

Tested examples:

| Input | Answer quality |
|-------|----------------|
| `"How does embedding search work?"` | Describes MiniLM → InMemoryVectorStore → auto-embedding pipeline with code example |
| `"What is NivaraChat?"` | Identifies as Nivara project component for RAG pipeline |

### Intent routing (`--intent`)
5-class intent classifier routes user input to specialist executors using conditional edges. Requires `--ollama` for specialist executors (except escalation). Training produces `models/intent_model.json` and `models/intent_tokenizer.json`.

```
User input
    │
    v
[IntentClassifier]           Nivara TextClassifierModel, 5 classes
    │
    ├── "factual"      ──> [FactualExecutor]       RAG retrieval + LLM generation
    ├── "question"     ──> [QuestionExecutor]      General Q&A via Ollama
    ├── "command"      ──> [CommandExecutor]       LLM with AIFunction tools
    ├── "complaint"    ──> [EscalationExecutor]    Human-in-the-loop (no LLM)
    └── "chitchat"     ──> [ChitchatExecutor]      Casual conversation via Ollama
```

Tested examples:

| Input | Intent | Response quality |
|-------|--------|------------------|
| `"I'm unhappy with the service"` | complaint | Escalation message with timestamp |
| `"What is the capital of France?"` | question | LLM answer: Paris |
| `"Hello!"` | chitchat | Friendly greeting |

Uses: `IntentClassifier`, `FactualExecutor`, `QuestionExecutor`, `CommandExecutor`, `EscalationExecutor`, `ChitchatExecutor`, `AddEdge<string>` conditional routing.

### Online learning (`--online-learning`)
Demonstrates online learning from LLM feedback. The intent classifier runs first; when confidence is below the threshold (default 0.8), the LLM provides a corrected intent which is added to a training buffer. When the buffer reaches 10 examples, the model is incrementally retrained using `IntentTrainer.TrainIncremental()` with a lower learning rate (0.0005). The updated model is saved and continues classifying. Requires `--ollama`.

```
User input
    │
    v
[IntentClassifier]           Nivara TextClassifierModel, 5 classes
    │
    ├── confidence >= 0.8    ──> Return Nivara classification (no LLM needed)
    │
    └── confidence < 0.8     ──> [Ollama LLM]
                                    │
                                    v
                              LLM provides corrected intent
                                    │
                                    v
                              Add (text, intent) to training buffer
                                    │
                                    v
                              Buffer full (10 examples)?
                                    │
                               yes ──> IntentTrainer.TrainIncremental()
                                    │   loads checkpoint (weights + optimizer state)
                                    │   trains 5 epochs at lr=0.0005 via Continue()
                                    │   saves updated model + checkpoint
                                    v
                              Continue with updated model
```

Tested examples:

| Input | Threshold | Result |
|-------|-----------|--------|
| `"hello there"` | 0.8 | chitchat (confidence: 0.934) — no LLM needed |
| `"what is the capital of France"` | 0.8 | LLM-corrected to question (confidence: 0.550) |
| `"turn on the lights"` | 0.8 | LLM-corrected to command (confidence: 0.493) |

Uses: `FeedbackCollector`, `IntentTrainer.TrainIncremental()`, `TrainingLoop.Continue()`, `TrainingLoop.SaveCheckpoint()/LoadCheckpoint()`, `Optimizer.StateDict()/LoadStateDict()`.

### TinyShakespeare (`--tinyshakespeare`)
Trains a **word-level batched causal transformer** on the TinyShakespeare corpus with Nivara's AutoDiff, then serves it through the standard `Microsoft.Extensions.AI.IChatClient` interface and wires it up via DI. No LLM needed — this mode proves Nivara can train a real transformer and serve it in an ecosystem-compatible way. Training runs when no `--load` is given; a saved model skips straight to generation and the DI demo. See the dedicated [TinyShakespeare section](#tinyshakespeare--batched-transformer-ichatclient-mode---tinyshakespeare) below for the full option list, architecture, and the how-it-differs-from-MicroGpt comparison.

```
TinyShakespeare.txt → word-level TextTokenizer → batched causal transformer training
    → ModelSerializer.Save/Load → BatchedChatClient : IChatClient
    → services.AddChatClient(factory) → console/ASP.NET/MAUI
```

Tested examples:

| Command | Model | Result |
|---------|-------|--------|
| `--tinyshakespeare` | 2L × 96D, 4 heads, vocab 8000 (defaults) | Full train on the corpus, 5 generated replies, DI demo reply |
| `--tinyshakespeare --vocab-size 1200 --prompt "ROMEO:"` | 2L × 96D, 4 heads, vocab 1200 | Smoke run, ~3x faster; generates Shakespeare-style continuations for the prompt |
| `--tinyshakespeare --data <small.txt> --vocab-size 800 --n-embd 32 --n-layer 1 --block-size 32 --n-head 2 --epochs 1 --samples 2 --prompt "ROMEO:"` | 1L × 32D, 2 heads, vocab 592 (31584 params) | Trained a 249-line slice in 1.3s (1350 tok/s, loss 6.17) and replied via `IChatClient` |
| `--tinyshakespeare --load models/ts.json --prompt "KING LEAR:" --no-di-demo` | Matches saved model | Skips training, loads weights + tokenizer, generates directly |

Uses: `BatchedTransformer<T>`, `BatchedChatClient` (`IChatClient`), `TextTokenizer`, `ModelSerializer.Save/Load`, `ReverseGradOperations.BatchedMultiHeadAttention`, `Embedding`/`Linear`/`LayerNorm`/`Activation.Gelu`, `CrossEntropyLoss<T>`, `Adam<T>`, `services.AddChatClient()`.

### SmolLM (`--smollm`)
Serves the **pretrained SmolLM-135M-Instruct causal LM** through the standard `Microsoft.Extensions.AI.IChatClient` interface — no training, no LLM server, no Python runtime. The model runs in-process on Nivara's zero-dependency tensor engine: `LlamaForCausalLM<T>` (autoregressive) + the GPT-2 byte-level BPE tokenizer. The conversation is rendered into SmolLM's Hermes/ChatML format (`<|im_start|>role\n...<|im_end|>`) and the reply is **token-streamed** via `GetStreamingResponseAsync`. Two sub-modes:

```
config.json + model.safetensors + vocab.json + merges.txt
    → SafeTensorsLoader.Read → LlamaLoader.Load → LlamaForCausalLM
    → SmolLMChatClient : IChatClient (greedy/sampled, token-streamed, KV-cached)
    → --smollm plain/chat
```

- `--smollm plain --text "..."` — single-shot reply, no REPL.
- `--smollm chat [--text "..."]` — interactive multi-turn REPL (default when no `--text`); a prompt skips straight to one turn. In the interactive menu (`RunInteractive`), generation options are prompted with sensible defaults.

Options after `--smollm`: `chat|plain` sub-mode, `--model-dir <path>` (default `samples/data/smollm-135m`), `--precision f32|bf16` (default `f32`), `--max-new-tokens <n>` (default 64), `--text <string>`, `--temperature <t>` (0 = greedy, >0 = sampling), `--top-p <p>` (nucleus cutoff, 0–1, default 1), `--seed <n>` (RNG seed for reproducible sampling), `--kv-cache` / `--no-kv-cache` (default: cached). Run `--smollm --help` for the full list.

Model files must be present under `samples/data/smollm-135m` (already downloaded in this repo). To re-download:

```
hf download HuggingFaceTB/SmolLM-135M-Instruct config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json special_tokens_map.json --local-dir samples/data/smollm-135m
```

Tested examples:

| Command | Precision | Result |
|---------|-----------|--------|
| `--smollm plain --text "The capital of France is" --max-new-tokens 12` | f32 | Streams "The capital of France is Paris, which is the largest city" |
| `--smollm chat` | f32 | Interactive multi-turn REPL; reply streamed token-by-token (KV-cached by default) |
| `--smollm chat --temperature 0.6` | f32 | Interactive REPL with sampling; varied replies across turns |
| `--smollm plain --text "The capital of France is" --precision bf16` | bf16 | Same reply, ~half the memory footprint |

This is **Stage A** of the SmolLM two-demo plan: plain causal-LM chat only. Tool calling (the SmolLM2 Hermes `<tool_call>`/`<tool_response>` format wiring Nivara's trained models as `AIFunction` tools) is a later stage and is intentionally not included here.

Uses: `LlamaForCausalLM<T>`, `LlamaLoader.Load`, `LlamaKVCache<T>`, `SafeTensorsLoader.Read<T>`, `Gpt2BpeTokenizer` (`Encode`/`Decode`/`TokenId`), `SmolLMChatClient<T>` (`IChatClient`, temperature/top-p sampling, KV-cached generation), `SmollmChatTemplate` (Hermes ChatML rendering).

### Qwen (`--qwen`)

Native **function calling** with the pretrained **Qwen2.5-0.5B-Instruct** checkpoint, still fully in-process on Nivara's zero-dependency tensor engine (no LLM server, no Python). The conversation is rendered **byte-identically** to HuggingFace's `apply_chat_template` for this checkpoint (`QwenChatTemplate`, pinned against Torch ground-truth fixtures), and the reply is decoded autoregressively with a KV cache. Three sub-modes:

```
config.json + model.safetensors + vocab.json + merges.txt + tokenizer.json
    → SafeTensorsLoader.Read → LlamaLoader.Load → LlamaForCausalLM
    → QwenChatClient : IChatClient (greedy/sampled, token-streamed, KV-cached, <tool_call> parsing)
    → FunctionInvokingChatClient (tools-weather loop, cap 3) or --qwen chat/plain
```

- `--qwen tools-weather --text "What's the weather in Paris?"` — the native tool-calling demo: the model emits `<tool_call>\n{"name": "getWeather", "arguments": {"city": "Paris"}}\n</tool_call>`, the `GetWeather` `AIFunction` runs, the result feeds back as `<tool_response>` inside a `user` turn, and the model closes with a clean natural-language answer (loop capped at 3 iterations so a model that never answers still exits cleanly). Interactive REPL when `--text` is omitted.
- `--qwen chat` / `--qwen plain` — interactive multi-turn REPL / single-shot plain-text reply, token-streamed.

Options after `--qwen`: `tools-weather|chat|plain` sub-mode, `--model-dir <path>` (default `samples/data/qwen2.5-0.5b-instruct`), `--precision f32|bf16` (default `f32`; bf16 keeps the BF16 weights native — BF16-on-disk is read as BFloat16 with no widen), `--max-new-tokens <n>` (default 128), `--text <string>`, `--temperature <t>` (0 = greedy, >0 = sampling), `--top-p <p>` (nucleus cutoff, 0–1, default 1), `--seed <n>` (RNG seed), `--kv-cache` / `--no-kv-cache` (default: cached). Run `--qwen --help` for the full list.

Model files must be present under `samples/data/qwen2.5-0.5b-instruct`:

```
hf download Qwen/Qwen2.5-0.5B-Instruct config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json special_tokens_map.json --local-dir samples/data/qwen2.5-0.5b-instruct
```

Tested examples:

| Command | Precision | Result |
|---------|-----------|--------|
| `--qwen tools-weather --text "What's the weather in Paris?"` | f32 | `[assistant → getWeather(city: Paris)]` → `[tool] Partly cloudy, 18°C. …` → "The weather in Paris is partly cloudy with a temperature of 18°C. …" (loop closes inside the cap) |
| `--qwen plain --text "The capital of France is" --max-new-tokens 24` | f32 | Streams "The capital of France is Paris." then stops on `<\|im_end\|>` |

Ground truth, the format findings, the BF16 loader benchmark, and the PyTorch
parity evidence live in `docs/QWEN.md`.

Uses: `LlamaForCausalLM<T>`, `LlamaLoader.Load`, `LlamaKVCache<T>`, `SafeTensorsLoader.Read<T>` (`Read<float>` fused BF16→F32 at load), `Gpt2BpeTokenizer` (Qwen Split-regex pretokenization), `QwenChatClient<T>` (`IChatClient`), `QwenChatTemplate` (byte-exact renderer), `QwenToolCallParser` (`<tool_call>` → `FunctionCallContent`), `FunctionInvokingChatClient` (MEAI tool loop), `AIFunctionFactory` (`GetWeather`).

## Agents pipeline architecture

```
Input text
    │
    v
[NivaraSentiment]          IChatClient → ChatClientAgent
    │   SentimentTextModel wraps TextClassifierModel<float>
    │   Output: "Positive (confidence: 0.92)" or "Unable to determine sentiment (confidence: 0.31)"
    v
[NivaraEntity]             IChatClient → ChatClientAgent
    │   EntityTextModel wraps TokenClassifierModel<float>
    │   Output: {"person":["John"],"org":["Acme Corp"],"date":["January 15"],"location":[]}
    v
[NivaraValidator]          IChatClient → ChatClientAgent
    │   ValidatorTextModel wraps TextClassifierModel<float>
    │   Output: {"validation":"VALID","confidence":0.87}
    v
[OllamaLLM]                (optional) IChatClient → ChatClientAgent
    │   Receives accumulated results, reasons about confidence signals
    v
Final output: structured result with confidence scores
```

Key design decisions:
- **No conditional edges** — low-confidence signals are expressed in the model output text itself (e.g. "Unable to determine sentiment (confidence: 0.31)"), letting downstream agents — including the LLM — reason about uncertainty naturally
- **Stateless models** — each agent extracts the original user message from the conversation history, ignoring prior turns
- **Same `IChatClient` abstraction** — Nivara models and Ollama LLM use the identical `AsAIAgent()` pipeline, no special executor types needed

## Workflow architecture (fan-out/fan-in)

The `--workflow` mode uses a different graph topology with explicit fan-out/fan-in:

```
Input text
    │
    v
[TextRouter]                   Pass-through, fans out to both analyzers
    │
    ├──> [SentimentExecutor]   Nivara-trained model, deterministic, <1ms
    │        returns: "positive" / "negative" / "neutral"
    │
    └──> [EntityExtractor]     Nivara-trained NER model, deterministic, <1ms
             returns: { person, org, date, location }
    │
    v  (fan-in barrier — waits for both)
[ValidatorExecutor]            Rule-based consistency check, deterministic, <1ms
    │
    v
[LLMAgent]                     (optional) ChatClientAgent + Ollama, stochastic
    │
    v
Final output: structured result
```

## Agent Framework integration patterns

Lessons learned from building this sample with Microsoft.Agents.AI.Workflows. Agent Framework is external; this section captures only Nivara-specific integration notes.

- `Executor<TInput, TOutput>` with `public override` — return value auto-sends downstream
- `.WithOutputFrom()` on `WorkflowBuilder` — registers executors as output sources
- Read `run.NewEvents` for `ExecutorCompletedEvent` (executor output) and `AgentResponseEvent` (LLM output)
- `OllamaApiClient` constructor doesn't throw — actual connection happens on `GetResponseAsync`
- **Workflow objects are single-use per run.** Do not reuse a `Workflow` instance across multiple `InProcessExecution.RunAsync` calls. Create a fresh workflow from the builder for each run (use a factory function / lambda). See [State Isolation](https://learn.microsoft.com/agent-framework/workflows/state#state-isolation).
- **Streaming output** arrives as `AgentResponseUpdateEvent` with one token per event. Accumulate per-executor-ID, then flush on `ExecutorCompletedEvent` or after all events to avoid printing each token on its own line.

Further reading:
- [Microsoft Agent Framework docs](https://learn.microsoft.com/agent-framework/workflows/executors)
- API reference and integration patterns: `docs/RESEARCH-AGENT-FRAMEWORK.md`

## Architecture

```
NivaraChat/
├── Program.cs                         # Thin CLI dispatcher: option parsing + mode routing
├── Modes/                             # One static class per CLI mode
│   ├── ModeContext.cs                 # Immutable option bag shared by every mode runner
│   ├── ModeHelpers.cs                 # Shared model loaders + agent run/print helpers
│   ├── TrainingMode.cs                # --train / --intent-train
│   ├── WorkflowMode.cs                # --workflow (fan-out/fan-in pipeline, Ollama optional)
│   ├── AgentsMode.cs                  # --agents / --interactive (agents pipeline)
│   ├── HandoffMode.cs                 # --handoff (confidence-based LLM handoff)
│   ├── ToolsMode.cs                   # --tools (LLM orchestrator + Nivara AIFunction tools)
│   ├── CriticMode.cs                  # --critic (writer-critic loop)
│   ├── IntentMode.cs                  # --intent (5-class routing to specialist executors)
│   ├── OnlineLearningMode.cs          # --online-learning (LLM feedback + retrain)
│   ├── EmbeddingMode.cs               # --embed (embedding search demo)
│   ├── RagMode.cs                     # --rag (retrieval-augmented generation)
│   └── RagAgentMode.cs                # --rag-agent (RAG with TextSearchProvider)
├── Executors/                         # Agent Framework Executor<string, string> subclasses
│   ├── TextRouter.cs                  # Pass-through executor for fan-out routing
│   ├── SentimentExecutor.cs           # Sentiment classification executor (--workflow)
│   ├── EntityExtractor.cs             # NER entity extraction executor (--workflow)
│   ├── ValidatorExecutor.cs           # Rule-based validator executor (--workflow)
│   ├── LlmExecutor.cs                 # Ollama LLM executor (--workflow)
│   ├── ConfidenceRouter.cs            # Confidence-based routing executor (--handoff)
│   ├── CriticExecutor.cs              # Scores LLM response quality (--critic)
│   ├── IntentClassifier.cs            # Intent classification executor (--intent)
│   ├── FactualExecutor.cs             # RAG-based factual executor (--intent)
│   ├── QuestionExecutor.cs            # General Q&A executor (--intent)
│   ├── CommandExecutor.cs             # Tool-calling executor (--intent)
│   ├── EscalationExecutor.cs          # Complaint escalation executor (--intent)
│   └── ChitchatExecutor.cs            # Casual conversation executor (--intent)
├── Models/                            # Text-in/text-out wrappers around ML models
│   ├── ITextModel.cs                  # Text-in/text-out abstraction for ML models
│   ├── SentimentTextModel.cs          # ITextModel wrapping TextClassifierModel<float>
│   ├── EntityTextModel.cs             # ITextModel wrapping TokenClassifierModel<float>
│   ├── ValidatorTextModel.cs          # ITextModel wrapping TextClassifierModel<float>
│   ├── NivaraChatClient.cs            # IChatClient wrapping ITextModel for agent participation
│   └── PassthroughTextModel.cs        # ITextModel wrapping IChatClient (Ollama passthrough)
├── Helpers/
│   ├── ModelInferenceHelper.cs        # Shared inference pipeline (DRY)
│   ├── DocumentChunk.cs               # DocumentChunk + DocumentChunker for RAG indexing
│   ├── FeedbackCollector.cs           # LLM fallback + feedback buffer (--online-learning)
│   └── WriterCriticLoop.cs            # Bounded writer-critic retry loop (--critic)
├── Tools/
│   └── NivaraToolFunctions.cs         # Nivara models as AIFunction tools (--tools)
├── Training/
│   ├── SentimentTrainer.cs            # Train sentiment model
│   ├── EntityTrainer.cs               # Train entity NER model
│   ├── ValidatorTrainer.cs            # Train workflow validator model
│   ├── AgentsValidatorTrainer.cs      # Train agents validator model
│   └── IntentTrainer.cs               # Train intent classifier + incremental retrain
├── Data/
│   ├── SyntheticDataGenerator.cs      # Generate all four datasets
│   └── IntentDataGenerator.cs         # Generate 5-class intent data
├── Transformer/
│   ├── TransformerMode.cs             # --tinyshakespeare CLI mode + interactive entry
│   ├── BatchedTransformer.cs          # BatchedTransformer<T> + BatchedTransformerBlock<T>
│   ├── BatchedChatClient.cs           # IChatClient over a trained BatchedTransformer<float>
│   ├── PositionEncoding.cs            # Fixed sinusoidal position encoding
│   └── TinyShakespeare.cs             # Corpus downloader + line-document loader
├── SmolLM/
│   ├── SmollmMode.cs                  # --smollm CLI mode + interactive entry (chat/plain)
│   ├── SmolLMChatClient.cs            # IChatClient over LlamaForCausalLM<T> (greedy/sampled, KV-cached, token-streamed)
│   └── SmollmChatTemplate.cs          # Hermes ChatML conversation rendering
├── Qwen/
│   ├── QwenMode.cs                    # --qwen CLI mode (tools-weather/chat/plain), tool-loop wiring
│   ├── QwenChatClient.cs              # IChatClient over LlamaForCausalLM<T> (<tool_call> parsing, KV-cached, token-streamed)
│   ├── QwenChatTemplate.cs            # Byte-exact Qwen2.5 ChatML renderer (+ Jinja-tojson JSON writer)
│   ├── QwenToolCallParser.cs          # <tool_call> → FunctionCallContent (strict + tolerant, name canonicalization)
│   └── QwenSampleTools.cs             # GetWeather AIFunction for the tools-weather demo
├── NivaraChat.csproj                  # Core + Agent Framework packages
└── README.md                          # This file
```

### Models

**Sentiment (`TextClassifierModel<float>`):**
```
Embedding(vocab, 32) → MeanPool → Linear(32, 64) → ReLU → Linear(64, 3)
```
3 classes: positive, negative, neutral.

**Entity extraction (`TokenClassifierModel<float>`):**
```
Embedding(vocab, 32) → Linear(32, 64) → ReLU → Linear(64, 5)
```
5 classes per token: O, B-person, B-org, B-date, B-location. No MeanPool — per-token predictions.

**Workflow validator (`TextClassifierModel<float>`):**
```
Embedding(vocab, 32) → MeanPool → Linear(32, 64) → ReLU → Linear(64, 2)
```
2 classes: valid, invalid. Trained on `"original || response"` format.

**Agents validator (`TextClassifierModel<float>`):**
```
Embedding(vocab, 32) → MeanPool → Linear(32, 64) → ReLU → Linear(64, 2)
```
2 classes: valid, invalid. Trained on multi-line accumulated pipeline output format.

**Intent classification (`TextClassifierModel<float>`):**
```
Embedding(vocab, 32) → MeanPool → Linear(32, 64) → ReLU → Linear(64, 5)
```
5 classes: factual, question, command, complaint, chitchat.

## Nivara APIs demonstrated

| API | Where | Purpose |
|-----|-------|---------|
| `TextClassifierModel<T>` | `Nivara.Samples` (shared with NivaraClassifier) | Document-level classification (sentiment, validator) |
| `TokenClassifierModel<T>` | `Nivara.Samples` | Token-level classification (NER) |
| `TextTokenizer` | `Nivara.Samples` (shared with NivaraClassifier) | Word-level tokenization, vocab, encode/decode |
| `Module<T>` | All models | Model base class |
| `Embedding<T>` | All models | Learned word embeddings |
| `Linear<T>` | All models | Fully connected layers |
| `CrossEntropyLoss<T>` | Training | Classification loss |
| `Adam<T>` | Training | Optimizer |
| `TrainingLoop<T>` | Training | Training orchestration |
| `DataLoader<T>` | Training | Batched data loading |
| `TensorDataset<T>` | Training | Frame-backed dataset |
| `ModelSerializer.Save/Load` | Training + inference | JSON model persistence |
| `Executor<TInput, TOutput>` | Executors (`--workflow`) | Workflow node with type-safe routing |
| `WorkflowBuilder` | Program.cs | Workflow graph construction with fan-out/fan-in |
| `AddFanOutEdge` | Program.cs | Broadcast input to multiple executors in parallel |
| `AddFanInBarrierEdge` | Program.cs | Wait for all parallel executors before proceeding |
| `InProcessExecution.RunAsync` | Program.cs | Static workflow execution |
| `AIFunctionFactory.Create` | NivaraToolFunctions.cs | Wrap static methods as LLM-callable tools |
| `IChatClient` | NivaraChatClient.cs | Microsoft.Extensions.AI chat abstraction |
| `AsAIAgent()` | Program.cs | Convert `IChatClient` to `ChatClientAgent` |
| `ChatClientAgent` | Program.cs | Agent Framework participant from `IChatClient` |
| `NivaraEmbeddingGenerator<T>` | Nivara.Extensions | `IEmbeddingGenerator<TInput, Embedding<float>>` implementation for local models |
| `TrainingLoop.Run(startEpoch)` | IntentTrainer.cs | Resume training from checkpoint |
| `Optimizer.StateDict()/LoadStateDict()` | IntentTrainer.cs | Save/restore optimizer state for incremental training |
| `ModelSerializer` (optimizer state) | IntentTrainer.cs | Persist optimizer state in checkpoints |

## TinyShakespeare — batched-transformer `IChatClient` mode (`--tinyshakespeare`)

The former `samples/NivaraChatClient/` companion project now ships as a built-in
mode of this sample: a **word-level batched causal transformer** trained on
TinyShakespeare with Nivara's AutoDiff, then served through the standard
`Microsoft.Extensions.AI.IChatClient` interface and wired up via DI. Where the
rest of this README demonstrates mixing trained ML models with an LLM
(fan-out/fan-in, agents, tools), this mode proves Nivara can *train* a real
transformer and *serve* it in an ecosystem-compatible way:

```
TinyShakespeare.txt → word-level TextTokenizer → batched causal transformer training
    → ModelSerializer.Save/Load → BatchedChatClient : IChatClient
    → services.AddChatClient(factory) → console/ASP.NET/MAUI
```

It is reachable from the interactive menu (option 4, which prompts with
defaults) or directly: `dotnet run --project samples/NivaraChat -- --tinyshakespeare`.

### How it differs from MicroGpt

MicroGpt does a per-position forward with a KV cache; this mode uses a proper
batched causal transformer over `[B, L]` tensors:

| Aspect | MicroGpt | `--tinyshakespeare` |
|---|---|---|
| Forward pass | Per-position, KV cache | Batched sequence `[B, L]` |
| Attention | Per-head dot-product loop | Batched multi-head attention (core `BatchedMultiHeadAttention` op) |
| Embedding | `Forward(int)` single token | `Forward(ReverseGradTensor<T>)` batch `[B, L] → [B, L, D]` |
| Normalization | RMSNorm op | `LayerNorm<T>` module |
| Position encoding | Learned `Embedding` | Sinusoidal (fixed, sample-side) |
| Loss | Hand-rolled NLL | `CrossEntropyLoss<T>` |
| Optimizer | Adam | Adam |
| Training | Manual grad-scope loop | Manual grad-scope loop (same pattern) |
| Serialization | None | `ModelSerializer` full round-trip + tokenizer JSON |
| Inference | Generate via sampling | `IChatClient` standard API |
| Data | Character-level names | Word-level TinyShakespeare |

### Architecture

```
Input tokens: [B, L]
    → Embedding → [B, L, D]
    → + sinusoidal position encoding
    → N × TransformerBlock:
        LayerNorm → Q/K/V projections → BatchedMultiHeadAttention (causal mask) → residual
        LayerNorm → MLP (GELU, expand 4×, compress) → residual
    → LayerNorm → tied LM head (MatMul(x, wteᵀ)) → [B*L, V] logits
```

- `BatchedTransformer<T>` (`Transformer/BatchedTransformer.cs`) composes core modules
  (`Embedding`, `Linear`, `LayerNorm`, `Activation.Gelu`) around the
  `ReverseGradOperations.BatchedMultiHeadAttention` op. The causal `[B, L, L]`
  mask is built once per forward and reused across blocks.
- The tied LM head is `MatMul(x, wteᵀ)`, matching the `[B*L, V]` → `CrossEntropyLoss`
  (shape[0] = batch, shape[1] = classes) convention.
- `BatchedChatClient : IChatClient` (`Transformer/BatchedChatClient.cs`) runs the
  model in eval mode and generates autoregressively with temperature sampling.
  Each call builds its own tensors, so concurrent use is safe; it does **not** own
  the model (disposal belongs to the caller / DI container).

### CLI (options after `--tinyshakespeare`)

Run `--tinyshakespeare --help` for the full list. Options mirror the flags below;
the interactive menu asks for the key ones (model path, vocab size, prompt) with
defaults.

```
--n-embd <int>          Embedding dimension (default: 96)
--n-layer <int>         Transformer layers (default: 2)
--block-size <int>      Context window (default: 64)
--n-head <int>          Attention heads (default: 4)
--dropout <float>       Dropout probability (default: 0.1)
--epochs <int>          Training epochs (default: 20)
--batch-size <int>      Batch size (default: 32)
--lr <float>            Learning rate (default: 3e-3)
--beta1/--beta2 <float> Adam betas (default: 0.9/0.95)
--vocab-size <int>      Max word-vocab size (default: 8000)
--temperature <float>   Sampling temperature (default: 0.8)
--max-new-tokens <int>  Max tokens per reply (default: 96)
--samples <int>         Generated samples (default: 5)
--seed <int>            RNG seed (default: 42)
--data <path>           Corpus path (downloaded to samples/data/tinyshakespeare.txt on first use)
--prompt <text>         Chat with the model using this user prompt
--save <path>           Save trained model to JSON (+ <path>.tokenizer.json)
--load <path>           Load model from JSON (pass the same architecture flags used at save time)
--no-di-demo            Skip the DI + IChatClient demo
--help, -h              Show this help
```

Default behavior: train (when no `--load`), save (when `--save`), print `--samples`
generated replies for `--prompt`, then run a DI demo that resolves `IChatClient`
from a `ServiceCollection` via `AddChatClient(factory)`.

> **CLI delta from the NEXT.md spec:** the spec's `--train`/`--interactive` REPL
> flags were dropped in favor of the NivaraGpt-style default-train + `--load`
> skip-training model; `--seq-len` was renamed `--block-size`. Durable NEXT.md
> content lives here instead.

### Gaps resolved along the way

The NEXT.md gap list is almost entirely obsolete — most items were already fixed
in core before this work (Grounding Audit): `LayerNorm<T>` module, batched
`Embedding<T>.Forward(ReverseGradTensor<T>)` (now Gather-based, no one-hot
MatMul), `ReverseGradOperations.Concat`/`Gather`,
`Embedding<T> : Module<T>`, `CrossEntropyLoss<T>`, `TrainingLoop<T>` +
`DataLoader<T>` + `TensorDataset<T>`, `ModelSerializer`. Two gaps no longer apply:
attention softmax is handled *inside* the MHA op kernel over the last dimension
(`AttentionKernels.SoftmaxRows`), so no public `Softmax(dim)` was needed, and the
"Nivara's MatMul is rank-2 only" workaround (flatten batch×head into one matrix)
is moot because the batched attention op does the batched scores/context math
directly. The remaining gaps were closed by this work:

- **Batched attention op** — `ReverseGradOperations.BatchedMultiHeadAttention<T>`
  (rank-3 `[B, L, D]`, optional `[B, qLen, kvLen]` additive mask, `Parallel.For`
  over the batch past a workload threshold); single-sequence MHA untouched.
  Mirrors PyTorch (see `tests/Nivara.Tests/NivaraTorch/BatchedAttentionTests.cs`).
- **IChatClient thread safety** — eval-mode, re-entrant generation; model
  ownership lives outside the client.

### Not doing (stretch / future)

- Batched KV cache — generation uses a simple autoregressive loop without caching.
- Top-p/top-k sampling, beam search — temperature + random sampling only.
- Fine-tuning/LoRA, quantization — full float32 CPU training only, matching
  Nivara's scope.
- BPE/subword tokenization, multi-modal, ASP.NET hosting — out of scope, but the
  sample is DI-compatible so hosting is a one-liner.

### Notes

- **Performance:** the word vocab dominates cost — core `TensorsHelper.MultiplyCore`
  emits one `TensorPrimitives.Dot` per output element, so the tied LM head over a
  ~8k vocab is ~1M short dot-calls/batch (~730 tok/s at `D=32, B=8`). Smoke runs
  use `--vocab-size 1200`.
- Load requires matching architecture flags; the strict shape validation in
  `LoadStateDict` fails loudly on a mismatch, and the matching
  `<model>.tokenizer.json` is auto-restored.

## Requirements

- .NET 10.0 SDK
- Nivara core library (`src/Nivara/Nivara.csproj`)
- Ollama (optional — only when `--ollama` flag is used; install from [ollama.com](https://ollama.com))

### Packages (example project only — core stays clean)

| Package | Version | Purpose |
|---------|---------|---------|
| `Microsoft.Agents.AI` | 1.16.0 | `ChatClientAgent` for LLM integration |
| `Microsoft.Agents.AI.Workflows` | 1.16.0 | `Executor`, `WorkflowBuilder`, `InProcessExecution` |
| `Microsoft.Agents.AI.Workflows.Generators` | 1.16.0 | Source generator for `[MessageHandler]` |
| `Microsoft.Extensions.AI` | 10.8.3 | `IChatClient` abstraction |
| `Microsoft.Extensions.DependencyInjection` | 10.0.10 | DI wiring for the TinyShakespeare `IChatClient` demo |
| `Microsoft.Extensions.Hosting` | 10.0.10 | Hosting infrastructure for the DI demo |
| `OllamaSharp` | 5.4.30 | `OllamaApiClient` implementing `IChatClient` |

## Library gaps this example resolved

### Library additions driven by this example

| New API | Location | Purpose |
|---------|----------|---------|
| `TextClassifierModel<T>` | `samples/Nivara.Samples/TextClassifierModel.cs` | Embedding → MeanPool → MLP document classifier. |
| `TokenClassifierModel<T>` | `samples/Nivara.Samples/TokenClassifierModel.cs` | Embedding → MLP per-token classifier for NER and sequence labeling. |
| `TextTokenizer` | `samples/Nivara.Samples/TextTokenizer.cs` | Word-level tokenizer with vocab, encode/decode, special tokens, save/load. |
| `MiniLMEmbeddingGenerator` | `samples/Nivara.Samples/BertModel.cs` | Factory wiring MiniLM weights + BertTokenizer into `NivaraEmbeddingGenerator<string>`. |

## Limitations

- **Word-level tokenization** — no subword (BPE) support. Out-of-vocabulary words map to UNK. Sufficient for synthetic data.
- **Synthetic training data** — entity extraction and validation use template-based synthetic data. Real applications would use annotated corpora.
- **No LLM streaming** — the workflow runs non-streaming. The LLM response is collected in full before validation.
- **Sequential agents** — the agents pipeline runs sequentially (Sentiment → Entity → Validator). Fan-out parallelism is only available in `--workflow` mode.
