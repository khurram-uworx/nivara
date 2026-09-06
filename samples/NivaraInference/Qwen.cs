using Microsoft.ML.Tokenizers;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Utilities;
using Nivara.Samples;
using System.Numerics.Tensors;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace NivaraInference;

/// <summary>
/// Special-token ids for Qwen2.5 (verified against the checkpoint's tokenizer.json /
/// generation_config.json, issue #382 Phase 2): eos_token_id is the array [151645, 151643]
/// and generation must stop on either.
/// </summary>
static class QwenIds
{
    public const int EndOfText = 151643;   // <|endoftext|> (also the bos id)
    public const int ImStart = 151644;     // <|im_start|>
    public const int ImEnd = 151645;       // <|im_end|> (primary eos)
    public const int ToolCall = 151657;    // <tool_call>
    public const int ToolCallEnd = 151658; // </tool_call>

    /// <summary>Generation stop ids (from generation_config.json eos_token_id).</summary>
    public static readonly int[] StopIds = [ImEnd, EndOfText];
}

/// <summary>
/// Renders the Qwen2.5 ChatML tool-calling prompt byte-for-byte identical to HuggingFace's
/// <c>apply_chat_template</c> for this checkpoint. This is the MEAI-free replica of the
/// NivaraChat <c>QwenChatTemplate</c> (the sample must not depend on Microsoft.Extensions.AI);
/// the whitespace/JSON layout is pinned against the ground-truth fixtures
/// <c>qwen_tool_prompt.txt</c> / <c>qwen_tool_final_prompt.txt</c>.
/// </summary>
static class QwenChatTemplate
{
    public const string DefaultSystem = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

    static readonly JsonSerializerOptions RelaxedJson = new()
    {
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    static string JsonStr(string s) => JsonSerializer.Serialize(s, RelaxedJson);

    /// <summary>Bakes the tool-mode system turn, identical to the checkpoint's <c>{%- if tools %}</c> branch.</summary>
    public static string BuildToolsSystemMessage(string toolJson)
    {
        return DefaultSystem +
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" +
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" +
            toolJson +
            "\n</tools>\n\nFor each function call, return a json object with function name and " +
            "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n" +
            "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
    }

    /// <summary>Emits a <c>{"type":"function","function":{...}}</c> declaration with Jinja-<c>tojson</c>
    /// spacing (spaces after <c>:</c>/<c>,</c>, literal non-ASCII and <c>'</c>).</summary>
    public static string ToolJson(string name, string description, IReadOnlyList<(string Param, string Type, string? Desc, bool Required)> props)
    {
        var sb = new StringBuilder();
        sb.Append("{\"type\": \"function\", \"function\": {");
        sb.Append("\"name\": ").Append(JsonStr(name)).Append(", ");
        sb.Append("\"description\": ").Append(JsonStr(description)).Append(", ");
        sb.Append("\"parameters\": {\"type\": \"object\", \"properties\": {");
        for (int i = 0; i < props.Count; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(JsonStr(props[i].Param)).Append(": {\"type\": ").Append(JsonStr(props[i].Type));
            if (props[i].Desc is { } desc)
                sb.Append(", \"description\": ").Append(JsonStr(desc));
            sb.Append('}');
        }
        sb.Append('}');
        var required = props.Where(p => p.Required).Select(p => p.Param).ToArray();
        if (required.Length > 0)
        {
            sb.Append(", \"required\": [");
            for (int i = 0; i < required.Length; i++)
            {
                if (i > 0) sb.Append(", ");
                sb.Append(JsonStr(required[i]));
            }
            sb.Append(']');
        }
        sb.Append("}}}");
        return sb.ToString();
    }

    /// <summary>Renders the full prompt: system (tools) + user + assistant tool-call turn + tool
    /// result + <c>&lt;|im_start|&gt;assistant\n</c> generation prompt.</summary>
    public static string RenderToolLoop(
        string systemMessage,
        string userText,
        string? decodedAssistantToolTurn,
        string? toolResult)
    {
        var sb = new StringBuilder();
        sb.Append("<|im_start|>system\n").Append(systemMessage).Append("<|im_end|>\n");
        sb.Append("<|im_start|>user\n").Append(userText).Append("<|im_end|>\n");
        if (decodedAssistantToolTurn != null)
            sb.Append("<|im_start|>assistant\n").Append(decodedAssistantToolTurn).Append("<|im_end|>\n");
        if (toolResult != null)
            sb.Append("<|im_start|>user\n<tool_response>\n").Append(toolResult)
              .Append("\n</tool_response><|im_end|>\n");
        sb.Append("<|im_start|>assistant\n");
        return sb.ToString();
    }

    /// <summary>First-turn prompt (system + user + generation prompt), used for the tool-call turn.</summary>
    public static string RenderFirstTurn(string systemMessage, string userText)
        => RenderToolLoop(systemMessage, userText, null, null);
}

/// <summary>Parses a single <c>&lt;tool_call&gt;{json}&lt;/tool_call&gt;</c> block from generated text.</summary>
static class QwenToolParser
{
    static readonly System.Text.RegularExpressions.Regex Block = new(
        "<tool_call>(.*?)</tool_call>",
        System.Text.RegularExpressions.RegexOptions.Compiled
        | System.Text.RegularExpressions.RegexOptions.Singleline
        | System.Text.RegularExpressions.RegexOptions.CultureInvariant);

    /// <summary>Returns the parsed JSON root of the first tool-call block, or null.</summary>
    public static JsonElement? TryParseToolCall(string text)
    {
        var m = Block.Match(text);
        if (!m.Success) return null;
        try
        {
            using var doc = JsonDocument.Parse(m.Groups[1].Value);
            return doc.RootElement.Clone();
        }
        catch (JsonException)
        {
            return null;
        }
    }

    public static string? JsonString(JsonElement? root, string property)
    {
        if (root is not { } r || r.ValueKind != JsonValueKind.Object) return null;
        return r.TryGetProperty(property, out var el) && el.ValueKind == JsonValueKind.String ? el.GetString() : null;
    }

    /// <summary>Extracts <c>arguments.&lt;name&gt;</c> as a string from a parsed tool-call root.</summary>
    public static string? ArgumentString(JsonElement? root, string argName)
    {
        if (root is not { } r || r.ValueKind != JsonValueKind.Object) return null;
        if (!r.TryGetProperty("arguments", out var args) || args.ValueKind != JsonValueKind.Object) return null;
        return args.TryGetProperty(argName, out var el) && el.ValueKind == JsonValueKind.String ? el.GetString() : null;
    }
}

/// <summary>
/// The <c>qwen</c> mode: native Qwen2.5 function calling (weather tool loop) + teacher
/// distillation into a tiny sentiment classifier. Inference-by-default (ADR-001/002): the
/// teacher runs outside <c>GradientUtils.Grad()</c>; the student trains inside it.
/// </summary>
static class Qwen
{
    const int FeatureDim = 4096;
    const int MaxNewTokens = 160;
    const string WeatherToolName = "getWeather";
    const string WeatherToolDesc = "Gets the current weather for a city. Returns a short description like 'Sunny, 22\u00b0C'.";
    const string CityParamDesc = "The city name, e.g. 'Paris' or 'New York'";
    const string ClassifyToolName = "classify_sentiment";
    const string ClassifyToolDesc = "Classifies the sentiment of a movie review text as positive or negative.";

    static readonly string[] TrainSentences =
    [
        "A heartwarming story that left me smiling long after the credits.",
        "Dull script and wooden acting sink this movie completely.",
        "The director weaves a tense, brilliant thriller from the first scene.",
        "Slow, empty, and painfully boring from start to finish.",
        "Charming performances and beautiful visuals make this a delight.",
        "A messy plot with forgettable characters, hard to recommend.",
        "Clever writing and great chemistry between the leads, loved it.",
        "A disappointing sequel that never justifies its runtime.",
        "Inventive and fun, this film earns every ounce of its praise.",
        "Derivative, loud, and shallow, it wastes a talented cast.",
    ];

    // Gold labels for the 8 eval sentences (CompareSentences): rows 0,2,4,6 positive; 1,3,5,7 negative.
    static readonly int[] EvalGold = [1, 0, 1, 0, 1, 0, 1, 0];

    // ----------------------------------------------------------------------------------
    // Model / tokenizer loading (shared by all sub-modes)
    // ----------------------------------------------------------------------------------

    public static (LlamaForCausalLM<float> Model, LlamaConfig Config, Gpt2BpeTokenizer Tokenizer) LoadModel(
        Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelDir)
    {
        var config = LlamaConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json")));
        var model = LlamaLoader.Load<float, float>(config, tensors);
        var tokenizer = new Gpt2BpeTokenizer(
            Path.Combine(modelDir, "vocab.json"),
            Path.Combine(modelDir, "merges.txt"),
            unkToken: "<|endoftext|>",
            tokenizerJsonPath: Path.Combine(modelDir, "tokenizer.json"));
        return (model, config, tokenizer);
    }

    // ----------------------------------------------------------------------------------
    // Greedy generation (KV-cache prefill-then-decode, stop-before-append parity)
    // ----------------------------------------------------------------------------------

    /// <summary>Greedily decodes after the prompt. With a KV cache it pre-fills the prompt once
    /// then decodes one token at a time; without it each step re-runs the full prefix. Stops on a
    /// Qwen stop id BEFORE appending it — matching the Torch reference <c>_greedy</c>, so the
    /// returned ids exclude the eos token (tool turn = 19 non-eos ids).</summary>
    public static List<int> Generate(
        LlamaForCausalLM<float> model, LlamaConfig config, IReadOnlyList<int> promptIds, int maxNewTokens, bool useKvCache)
    {
        int kvWidth = config.NumKeyValueHeads * (config.HiddenSize / config.NumAttentionHeads);
        using var cache = new LlamaKVCache<float>(config.NumHiddenLayers, kvWidth);

        ReverseGradTensor<float> logits;
        if (useKvCache)
        {
            logits = null!;
            for (int p = 0; p < promptIds.Count; p++)
                logits = model.ForwardCached(promptIds[p], p, cache);
        }
        else
        {
            logits = model.Forward(promptIds.ToArray());
        }

        int position = promptIds.Count;
        var gen = new List<int>();
        for (int t = 0; t < maxNewTokens && gen.Count < config.MaxPositionEmbeddings; t++)
        {
            int next = ArgMaxLastRow(logits, config.VocabSize);
            if (QwenIds.StopIds.Contains(next))
                break;
            gen.Add(next);
            logits = useKvCache
                ? model.ForwardCached(next, position++, cache)
                : model.Forward(BuildSequence(promptIds, gen));
        }
        return gen;
    }

    static int[] BuildSequence(IReadOnlyList<int> promptIds, List<int> gen)
    {
        var seq = new int[promptIds.Count + gen.Count];
        for (int i = 0; i < promptIds.Count; i++) seq[i] = promptIds[i];
        for (int i = 0; i < gen.Count; i++) seq[promptIds.Count + i] = gen[i];
        return seq;
    }

    static int ArgMaxLastRow(ReverseGradTensor<float> logits, int vocab)
    {
        logits.Data.TryGetSpan(out var span);
        int offset = span.Length - vocab;
        int best = 0;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < vocab; i++)
        {
            float v = span[offset + i];
            if (v > bestVal) { bestVal = v; best = i; }
        }
        return best;
    }

    /// <summary>Last-row logits of the model over the full prefix (no cache), for numeric diffing
    /// against the Torch reference on the FIXED trajectory.</summary>
    static float[] ForwardLastLogits(LlamaForCausalLM<float> model, IReadOnlyList<int> prefixIds, int vocab)
    {
        var logits = model.Forward(prefixIds.ToArray());
        logits.Data.TryGetSpan(out var span);
        int offset = span.Length - vocab;
        var result = new float[vocab];
        for (int i = 0; i < vocab; i++) result[i] = span[offset + i];
        return result;
    }

    // ----------------------------------------------------------------------------------
    // RunTools — native weather function calling + fixture diff
    // ----------------------------------------------------------------------------------

    public static int RunTools(
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        string modelDir,
        bool useKvCache,
        string text)
    {
        Console.WriteLine("=== Qwen2.5-0.5B-Instruct: Native Function Calling (getWeather) ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})  Precision: F32 (BF16-upcast)  KV cache: {(useKvCache ? "on" : "off")}");
        Console.WriteLine();

        var (model, config, tokenizer) = LoadModel(tensors, modelDir);
        Console.WriteLine($"Config: hidden={config.HiddenSize}, layers={config.NumHiddenLayers}, " +
                          $"heads={config.NumAttentionHeads}, kvHeads={config.NumKeyValueHeads}, vocab={config.VocabSize}");
        Console.WriteLine();

        string userText = string.IsNullOrWhiteSpace(text) ? "What's the weather in Paris?" : text;
        string toolJson = QwenChatTemplate.ToolJson(
            WeatherToolName, WeatherToolDesc,
            [("city", "string", CityParamDesc, true)]);
        string systemMessage = QwenChatTemplate.BuildToolsSystemMessage(toolJson);

        // ---- Turn 1: assistant issues a <tool_call> ----
        string firstPrompt = QwenChatTemplate.RenderFirstTurn(systemMessage, userText);
        var firstPromptIds = tokenizer.Encode(firstPrompt).ToList();
        Console.WriteLine($"User: {userText}");
        Console.WriteLine($"Rendered prompt tokens: {firstPromptIds.Count}");

        if (File.Exists(Path.Combine(modelDir, "qwen_tool_prompt.txt"))
            && File.Exists(Path.Combine(modelDir, "qwen_tool_prompt_ids.bin"))
            && userText == "What's the weather in Paris?")
            CompareInt32Fixture(modelDir, "qwen_tool_prompt_ids.bin", firstPromptIds, "tool prompt ids");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var toolTurnIds = Generate(model, config, firstPromptIds, MaxNewTokens, useKvCache);
        sw.Stop();
        string toolTurnText = tokenizer.Decode(toolTurnIds);
        Console.WriteLine($"Tool-call turn: {toolTurnIds.Count} tokens in {sw.Elapsed.TotalSeconds:F1}s " +
                          $"({sw.ElapsedMilliseconds / Math.Max(1, toolTurnIds.Count)} ms/token)");
        Console.WriteLine(toolTurnText);
        Console.WriteLine();

        var parsed = QwenToolParser.TryParseToolCall(toolTurnText);
        string toolName = QwenToolParser.JsonString(parsed, "name") ?? WeatherToolName;
        string city = QwenToolParser.ArgumentString(parsed, "city") ?? "Paris";
        Console.WriteLine($"Parsed tool call: name={toolName}, city={city}");
        Console.WriteLine();

        string weatherResult = GoGetWeather(city);
        Console.WriteLine($"Tool result: {weatherResult}");
        Console.WriteLine();

        // ---- Turn 2: feed the tool result back; assistant answers in natural language ----
        string finalPrompt = QwenChatTemplate.RenderToolLoop(systemMessage, userText, toolTurnText, weatherResult);
        var finalPromptIds = tokenizer.Encode(finalPrompt).ToList();

        if (File.Exists(Path.Combine(modelDir, "qwen_tool_final_prompt.txt"))
            && File.Exists(Path.Combine(modelDir, "qwen_tool_final_prompt_ids.bin"))
            && userText == "What's the weather in Paris?")
            CompareInt32Fixture(modelDir, "qwen_tool_final_prompt_ids.bin", finalPromptIds, "final prompt ids");

        sw.Restart();
        var finalTurnIds = Generate(model, config, finalPromptIds, MaxNewTokens, useKvCache);
        sw.Stop();
        string finalAnswer = tokenizer.Decode(finalTurnIds);
        Console.WriteLine($"Final answer turn: {finalTurnIds.Count} tokens in {sw.Elapsed.TotalSeconds:F1}s");
        Console.WriteLine($"Assistant: {finalAnswer}");
        Console.WriteLine();

        // ---- Fixture diffs (silently skipped when the fixtures are absent) ----
        string fixtureDir = modelDir;
        string idsPath = Path.Combine(fixtureDir, "qwen_tool_ids_py.bin");
        if (File.Exists(idsPath))
            CompareGeneratedToolTurn(idsPath, toolTurnIds);

        string logitsPath = Path.Combine(fixtureDir, "qwen_tool_logits_py.bin");
        if (File.Exists(logitsPath) && File.Exists(idsPath) && userText == "What's the weather in Paris?")
        {
            var torchFinalTurn = ReadInt32Fixture(fixtureDir, "qwen_tool_ids_py.bin").Skip(19).ToArray();
            var fullIds = finalPromptIds.Concat(torchFinalTurn).ToArray();
            var csLogits = ForwardLastLogits(model, fullIds, config.VocabSize);
            CompareLogitsFixture(logitsPath, csLogits);
        }
        else if (File.Exists(logitsPath))
        {
            Console.WriteLine("Skipping logits diff: fixtures are pinned to the default Paris prompt.");
        }

        bool ok = finalAnswer.Contains("partly cloudy", StringComparison.OrdinalIgnoreCase);
        Console.WriteLine(ok
            ? "Final answer reflects the tool observation (likely semantic success)."
            : "Note: final answer did not mention the observed weather — inspect above.");
        Console.WriteLine();
        return ok ? 0 : 2;
    }

    static string GoGetWeather(string city)
    {
        return city.Trim().Equals("Paris", StringComparison.OrdinalIgnoreCase)
            ? "Partly cloudy, 18\u00b0C. Light breeze from the northwest."
            : $"Partly cloudy, {12 + Math.Abs(city.Trim().Length) % 10}\u00b0C. Light breeze from the northwest.";
    }

    // ----------------------------------------------------------------------------------
    // RunBenchmark — KV-cache decode throughput
    // ----------------------------------------------------------------------------------

    public static int RunBenchmark(
        Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelDir)
    {
        Console.WriteLine("=== Qwen2.5-0.5B-Instruct: KV-cache decode benchmark ===");
        var (model, config, tokenizer) = LoadModel(tensors, modelDir);

        string toolJson = QwenChatTemplate.ToolJson(
            WeatherToolName, WeatherToolDesc, [("city", "string", CityParamDesc, true)]);
        string systemMessage = QwenChatTemplate.BuildToolsSystemMessage(toolJson);
        var promptIds = tokenizer.Encode(QwenChatTemplate.RenderFirstTurn(systemMessage, "What's the weather in Paris?"));

        Console.WriteLine($"Prompt tokens: {promptIds.Count}. Decoding the tool call turn {3} times each path...");
        Console.WriteLine();

        TimeResult Cached() => TimeGeneration(model, config, promptIds, useKvCache: true);
        TimeResult Full() => TimeGeneration(model, config, promptIds, useKvCache: false);

        // Warmup.
        Cached();

        var cachedRuns = Enumerable.Range(0, 3).Select(_ => Cached()).ToArray();
        var fullRuns = Enumerable.Range(0, 3).Select(_ => Full()).ToArray();

        static (double AvgMsTok, double MedianMs, int Tokens) Summarize(TimeResult[] runs)
        {
            var s = runs.OrderBy(r => r.Ms).ToArray();
            return (runs.Average(r => r.MsPerTok), s[1].Ms, s[1].Tokens);
        }

        var c = Summarize(cachedRuns);
        var f = Summarize(fullRuns);
        Console.WriteLine($"  KV cache:  median {c.MedianMs:F0} ms for {c.Tokens} tokens ({c.AvgMsTok:F1} ms/token, {1000.0 / Math.Max(0.1, c.AvgMsTok):F1} tok/s)");
        Console.WriteLine($"  Full fwd:  median {f.MedianMs:F0} ms for {f.Tokens} tokens ({f.AvgMsTok:F1} ms/token, {1000.0 / Math.Max(0.1, f.AvgMsTok):F1} tok/s)");
        if (f.AvgMsTok > 0)
            Console.WriteLine($"  Speedup:   {f.AvgMsTok / Math.Max(0.01, c.AvgMsTok):F1}x");
        Console.WriteLine();
        return 0;
    }

    struct TimeResult { public double Ms; public double MsPerTok; public int Tokens; }

    static TimeResult TimeGeneration(LlamaForCausalLM<float> model, LlamaConfig config, IReadOnlyList<int> promptIds, bool useKvCache)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var ids = Generate(model, config, promptIds, MaxNewTokens, useKvCache);
        sw.Stop();
        return new TimeResult
        {
            Ms = sw.Elapsed.TotalMilliseconds,
            MsPerTok = sw.Elapsed.TotalMilliseconds / Math.Max(1, ids.Count),
            Tokens = ids.Count,
        };
    }

    // ----------------------------------------------------------------------------------
    // RunDistill — teacher-annotated distillation into a tiny sentiment classifier
    // ----------------------------------------------------------------------------------

    public static int RunDistill(
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        string modelDir,
        int teacherExamples,
        bool force,
        int seed)
    {
        Console.WriteLine("=== Qwen2.5-0.5B-Instruct: Teacher distillation into a tiny sentiment classifier ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})  Precision: F32 (BF16-upcast)");
        Console.WriteLine();

        var (model, config, tokenizer) = LoadModel(tensors, modelDir);
        Console.WriteLine($"Teacher: {config.NumHiddenLayers} layers, hidden={config.HiddenSize}, vocab={config.VocabSize}");
        Console.WriteLine($"Student: hashed word+bigram BOW (FNV-1a, {FeatureDim}-d) -> Linear({FeatureDim}\u219264) -> ReLU -> Linear(64\u21922)");
        Console.WriteLine($"Train sentences: {TrainSentences.Length} (teacher-annotated, resumable cache)");
        Console.WriteLine();

        if (teacherExamples <= 0 || teacherExamples > TrainSentences.Length)
            teacherExamples = TrainSentences.Length;

        // ---- Teacher pass: classify each train sentence (cached, resumable) ----
        string cachePath = Path.Combine(modelDir, "qwen_distill_labels.json");
        var trainLabels = LoadOrRunTeacher(model, config, tokenizer, cachePath,
            TrainSentences, teacherExamples, force);
        PrintTrainLabels(TrainSentences, trainLabels);

        // ---- Build student features (hashed BOW) ----
        var trainFeatures = Array.ConvertAll(TrainSentences, HashSentenceFeatures);
        var featureData = new float[TrainSentences.Length * FeatureDim];
        for (int i = 0; i < TrainSentences.Length; i++)
            for (int d = 0; d < FeatureDim; d++)
                featureData[i * FeatureDim + d] = trainFeatures[i][d];
        var featuresTensor = ReverseGradTensor<float>.FromMatrix(featureData, TrainSentences.Length, FeatureDim, requiresGrad: false);

        var labelInts = Array.ConvertAll(trainLabels, l => l == "positive" ? 1 : 0);

        // ---- Train the student MLP (full-batch, inside Grad scope) ----
        Console.WriteLine("Training student MLP (full-batch, Adam 1e-3, 200 epochs)...");
        using var student = new SentimentMLP();
        using var optimizer = new Adam<float>(learningRate: 1e-3f);
        optimizer.AddParameterGroup(student.GetParameters().Values);
        var lossFn = new CrossEntropyLoss<float>();
        float lastLoss = 0f;
        for (int epoch = 0; epoch < 200; epoch++)
        {
            using var gradScope = GradientUtils.Grad();
            var logits = student.Forward(featuresTensor);
            var loss = lossFn.Forward(logits, labelInts);
            loss.Backward();
            optimizer.Step();
            optimizer.ZeroGrad();
            lastLoss = loss[0];
            if ((epoch + 1) % 50 == 0)
                Console.WriteLine($"   epoch {epoch + 1,3}: loss = {lastLoss:F4}");
        }
        Console.WriteLine($"   final epoch 200: loss = {lastLoss:F4}");
        Console.WriteLine();

        // ---- Linear-only baseline (4096 -> 2) ----
        Console.WriteLine("Training linear-only baseline (Linear 4096->2, Adam 1e-3, 200 epochs)...");
        using var baseline = new LinearBaseline();
        using var baselineOpt = new Adam<float>(learningRate: 1e-3f);
        baselineOpt.AddParameterGroup(baseline.GetParameters().Values);
        for (int epoch = 0; epoch < 200; epoch++)
        {
            using var gradScope = GradientUtils.Grad();
            var logits = baseline.Forward(featuresTensor);
            var loss = lossFn.Forward(logits, labelInts);
            loss.Backward();
            baselineOpt.Step();
            baselineOpt.ZeroGrad();
        }
        Console.WriteLine();

        // ---- Evaluate on the 8 CompareSentences ----
        Console.WriteLine("Evaluating on the 8 shared SST-2 eval sentences...");
        bool haveDistilBert = Directory.Exists(Path.Combine("samples", "data", "distilbert_sst"))
            && File.Exists(Path.Combine("samples", "data", "distilbert_sst", "model.safetensors"))
            && File.Exists(Path.Combine("samples", "data", "distilbert_sst", "config.json"));
        DistilBertForSequenceClassification<float>? sstModel = null;
        BertTokenizer? sstTokenizer = null;
        if (haveDistilBert)
        {
            var sstDir = Path.Combine("samples", "data", "distilbert_sst");
            var sstTensors = SafeTensorsLoader.Read(Path.Combine(sstDir, "model.safetensors"));
            sstModel = DistilBertSst.Load(sstTensors, sstDir);
            sstTokenizer = DistilBertSst.LoadTokenizer(sstDir);
            sstModel.Eval();
        }

        var evalSentences = DistilBertSst.CompareSentences;
        var evalTeacher = LoadOrRunTeacher(model, config, tokenizer, cachePath,
            evalSentences, evalSentences.Length, force);

        var studentRight = 0;
        var baselineRight = 0;
        var teacherRight = 0;
        var sstRight = 0;

        Console.WriteLine();
        Console.WriteLine($"  {"[i]",-4} {"Teacher",-8} {"Student",-8} {"Baseline",-9} {"DistilBERT SST2",-15} {"Gold",-5}");
        for (int i = 0; i < evalSentences.Length; i++)
        {
            var features = ReverseGradTensor<float>.FromMatrix(
                Array.ConvertAll(HashSentenceFeatures(evalSentences[i]), x => (float)x), 1, FeatureDim, requiresGrad: false);

            int teacher = evalTeacher[i] == "positive" ? 1 : 0;
            int studentPred = ArgMax(student.Forward(features));
            int baselinePred = ArgMax(baseline.Forward(features));
            int gold = EvalGold[i];

            string sstLabel = "n/a";
            if (sstModel != null && sstTokenizer != null)
            {
                var sstLogits = DistilBertSst.PredictLogits(sstModel, sstTokenizer, evalSentences[i], maxLen: 128);
                var sstArg = sstLogits.Data[1] > sstLogits.Data[0] ? 1 : 0;
                sstRight += sstArg == gold ? 1 : 0;
                sstLabel = sstArg == 1 ? "positive" : "negative";
            }

            if (teacher == gold) teacherRight++;
            if (studentPred == gold) studentRight++;
            if (baselinePred == gold) baselineRight++;

            Console.WriteLine($"  [{i,-2}] {LabelOf(evalTeacher[i]),-8} {LabelOfOf(studentPred),-8} {LabelOfOf(baselinePred),-9} {sstLabel,-15} {LabelOfOf(gold),-5}");
        }

        Console.WriteLine();
        Console.WriteLine("Accuracy on the 8 SST-2 eval sentences:");
        Console.WriteLine($"  Teacher       {teacherRight,2}/8  ({teacherRight / 8.0:P0})");
        Console.WriteLine($"  Student MLP   {studentRight,2}/8  ({studentRight / 8.0:P0})");
        Console.WriteLine($"  Linear base   {baselineRight,2}/8  ({baselineRight / 8.0:P0})");
        if (sstModel != null)
            Console.WriteLine($"  DistilBERT    {sstRight,2}/8  ({sstRight / 8.0:P0})   [loaded from samples/data/distilbert_sst]");
        else
            Console.WriteLine("  DistilBERT    n/a   [distilbert_sst weights not present; supply samples/data/distilbert_sst to compare]");
        Console.WriteLine();
        Console.WriteLine($"Seed (accepted for future use; Kaiming init is unseeded): {seed}");
        Console.WriteLine();
        return 0;
    }

    static string LabelOf(string teacherLabel) => teacherLabel;
    static string LabelOfOf(int v) => v == 1 ? "positive" : "negative";

    /// <summary>Reads or runs the teacher classify_sentiment tool over the given sentences. Results
    /// are cached in a resumable JSON file; <paramref name="limit"/> before a cached full set runs only
    /// the missing prefix (resumable incremental cache).</summary>
    static string[] LoadOrRunTeacher(
        LlamaForCausalLM<float> model, LlamaConfig config, Gpt2BpeTokenizer tokenizer,
        string cachePath, IReadOnlyList<string> sentences, int limit, bool force)
    {
        var cache = LoadLabelCache(cachePath);
        var labels = new string[sentences.Count];
        int cacheHits = 0;

        for (int i = 0; i < sentences.Count; i++)
        {
            if (!force && cache.TryGetValue(sentences[i], out var cached))
            {
                labels[i] = cached;
                cacheHits++;
                continue;
            }
            if (i >= limit)
            {
                // beyond the requested teacher-examples budget and not cached: leave as a best-effort guess
                labels[i] = "negative";
                continue;
            }

            Console.Write($"  [teacher] classifying [{i}] \"{Shorten(sentences[i])}\"... ");
            var sw = System.Diagnostics.Stopwatch.StartNew();
            string label = RunTeacherClassify(model, config, tokenizer, sentences[i]);
            sw.Stop();
            labels[i] = label;
            cache[sentences[i]] = label;
            Console.WriteLine($"{label} ({sw.Elapsed.TotalSeconds:F0}s)");
            SaveLabelCache(cachePath, cache);
        }

        Console.WriteLine($"  teacher labels: {cacheHits}/{sentences.Count} from cache{(cacheHits > 0 ? "" : " (first pass, all new)")}");
        return labels;
    }

    static string RunTeacherClassify(
        LlamaForCausalLM<float> model, LlamaConfig config, Gpt2BpeTokenizer tokenizer, string sentence)
    {
        string toolJson = QwenChatTemplate.ToolJson(
            ClassifyToolName, ClassifyToolDesc,
            [("text", "string", null, true), ("label", "string", null, true)]);
        string systemMessage = QwenChatTemplate.BuildToolsSystemMessage(toolJson);
        string prompt = QwenChatTemplate.RenderFirstTurn(systemMessage, $"Classify the sentiment of this movie review: \"{sentence}\"");
        var promptIds = tokenizer.Encode(prompt).ToList();

        var turnIds = Generate(model, config, promptIds, MaxNewTokens, useKvCache: true);
        string text = tokenizer.Decode(turnIds);
        var parsed = QwenToolParser.TryParseToolCall(text);
        string? label = QwenToolParser.ArgumentString(parsed, "label");
        if (!string.IsNullOrWhiteSpace(label))
        {
            var norm = label.Trim().ToLowerInvariant();
            if (norm.Contains("positive")) return "positive";
            if (norm.Contains("negative")) return "negative";
        }
        // Fallback: scan the raw generated text for a sentiment word.
        var lower = text.ToLowerInvariant();
        return lower.Contains("positive") ? "positive" : lower.Contains("negative") ? "negative" : "negative";
    }

    static string Shorten(string s) => s.Length <= 48 ? s : s[..45] + "...";

    static Dictionary<string, string> LoadLabelCache(string path)
    {
        if (!File.Exists(path)) return new Dictionary<string, string>(StringComparer.Ordinal);
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            var result = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var prop in doc.RootElement.EnumerateObject())
            {
                var v = prop.Value.ValueKind == JsonValueKind.String ? prop.Value.GetString() : null;
                if (v != null) result[prop.Name] = v;
            }
            return result;
        }
        catch (JsonException)
        {
            return new Dictionary<string, string>(StringComparer.Ordinal);
        }
    }

    static void SaveLabelCache(string path, Dictionary<string, string> cache)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, JsonSerializer.Serialize(cache, new JsonSerializerOptions { WriteIndented = true }));
    }

    // ----------------------------------------------------------------------------------
    // Distill helpers: feature hashing, student module, baseline
    // ----------------------------------------------------------------------------------

    static uint Fnv1a(string s)
    {
        uint hash = 2166136261u;
        foreach (var c in s) { hash ^= c; hash *= 16777619u; }
        return hash;
    }

    /// <summary>Hashes a sentence into a {FeatureDim}-dim word+bigram bag-of-words count vector
    /// using FNV-1a. Lowercased alphanumeric runs become tokens; adjacent pairs become bigrams.</summary>
    static int[] HashSentenceFeatures(string text)
    {
        var features = new int[FeatureDim];
        var words = Tokenize(text);
        foreach (var w in words)
            features[Fnv1a(w) % FeatureDim]++;
        for (int i = 0; i + 1 < words.Count; i++)
            features[Fnv1a(words[i] + " " + words[i + 1]) % FeatureDim]++;
        return features;
    }

    static List<string> Tokenize(string text)
    {
        var words = new List<string>();
        var sb = new StringBuilder();
        foreach (var c in text)
        {
            if (char.IsLetterOrDigit(c)) sb.Append(char.ToLowerInvariant(c));
            else if (sb.Length > 0) { words.Add(sb.ToString()); sb.Clear(); }
        }
        if (sb.Length > 0) words.Add(sb.ToString());
        return words;
    }

    static int ArgMax(ReverseGradTensor<float> logits)
    {
        logits.Data.TryGetSpan(out var span);
        return span[1] > span[0] ? 1 : 0;
    }

    sealed class SentimentMLP : Module<float>
    {
        public Linear<float> Hidden { get; }
        public Linear<float> Head { get; }

        public SentimentMLP()
        {
            Hidden = new Linear<float>(FeatureDim, 64);
            Head = new Linear<float>(64, 2);
            RegisterModules(Hidden, Head);
        }

        public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
            => Head.Forward(Activation.Relu(Hidden.Forward(input)));
    }

    sealed class LinearBaseline : Module<float>
    {
        public Linear<float> Head { get; }

        public LinearBaseline()
        {
            Head = new Linear<float>(FeatureDim, 2);
            RegisterModules(Head);
        }

        public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
            => Head.Forward(input);
    }

    // ----------------------------------------------------------------------------------
    // Fixture readers / diffs
    // ----------------------------------------------------------------------------------

    static void PrintTrainLabels(string[] sentences, string[] labels)
    {
        Console.WriteLine("Train sentences (teacher labels):");
        for (int i = 0; i < sentences.Length; i++)
            Console.WriteLine($"  [{i}] {labels[i],-9}  \"{sentences[i]}\"");
        Console.WriteLine();
    }

    static int[] ReadInt32Fixture(string dir, string name)
        => ReadInt32Array(Path.Combine(dir, name));

    static void CompareInt32Fixture(string dir, string name, IReadOnlyList<int> ids, string label)
    {
        var expected = ReadInt32Fixture(dir, name);
        bool ok = expected.Length == ids.Count;
        if (ok)
            for (int i = 0; i < expected.Length; i++)
                if (ids[i] != expected[i]) { ok = false; break; }

        Console.WriteLine($"  [fixture] {label}: {(ok ? "MATCH" : $"MISMATCH ({Math.Min(expected.Length, ids.Count)} compared, expected {expected.Length}, got {ids.Count})")}");
        if (!ok)
        {
            int n = Math.Min(expected.Length, ids.Count);
            for (int i = 0; i < n; i++)
                if (ids[i] != expected[i])
                {
                    Console.WriteLine($"    first diff @{i}: C#={ids[i]} Py={expected[i]}");
                    break;
                }
        }
    }

    static void CompareGeneratedToolTurn(string idsPath, IReadOnlyList<int> toolTurnIds)
    {
        var expected = ReadInt32Array(idsPath);
        int match = 0;
        for (int i = 0; i < Math.Min(toolTurnIds.Count, expected.Length); i++)
            if (toolTurnIds[i] == expected[i]) match++;

        Console.WriteLine($"  [fixture] tool-call turn ids: {match}/{Math.Min(toolTurnIds.Count, expected.Length)} match " +
                          $"(Py has {expected.Length} total = tool {19} + final {expected.Length - 19})");
        if (match != Math.Min(toolTurnIds.Count, expected.Length))
            Console.WriteLine("    tool-call turn argmax parity differs — see the fixture tool ids above.");
    }

    static int[] ReadInt32Array(string path)
    {
        var raw = File.ReadAllBytes(path);
        var result = new int[raw.Length / 4];
        for (int i = 0; i < result.Length; i++)
            result[i] = BitConverter.ToInt32(raw, i * 4);
        return result;
    }

    static void CompareLogitsFixture(string logitsPath, float[] csLogits)
    {
        var raw = File.ReadAllBytes(logitsPath);
        var refLogits = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, refLogits, 0, raw.Length);

        int len = Math.Min(csLogits.Length, refLogits.Length);
        var diffArr = new float[len];
        TensorPrimitives.Subtract(csLogits.AsSpan(0, len), refLogits.AsSpan(0, len), diffArr);
        var absDiff = new float[len];
        TensorPrimitives.Abs(diffArr.AsSpan(), absDiff);
        float maxAbs = TensorPrimitives.Max(absDiff);
        float meanAbs = TensorPrimitives.Sum(absDiff) / len;
        float cosineSim = TensorPrimitives.CosineSimilarity(csLogits.AsSpan(0, len), refLogits.AsSpan(0, len));

        // Envelope: 3% relative to the reference's max |logit| + 0.5 absolute floor — the same
        // formulation (and observed ~0.4 worst-case) as QwenInstructParityTests.Model_
        // GreedyFinalAnswer_SemanticParityAndFinalLogitsWithinTolerance. Gross numeric errors
        // (rope/transpose/attention bugs) land far outside it.
        float refMaxAbs = 0f;
        for (int i = 0; i < len; i++)
        {
            float a = MathF.Abs(refLogits[i]);
            if (a > refMaxAbs) refMaxAbs = a;
        }
        bool within = maxAbs < 0.03f * refMaxAbs + 0.5f;

        Console.WriteLine($"  [fixture] final-position logits: maxAbs={maxAbs:F5}, meanAbs={meanAbs:F7}, cosine={cosineSim:F6}, within 3%·|ref|max+0.5 envelope: {(within ? "YES" : "NO")}");
    }
}
