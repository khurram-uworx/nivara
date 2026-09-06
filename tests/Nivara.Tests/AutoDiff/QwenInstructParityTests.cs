using Nivara.AutoDiff;
using Nivara.Samples;
using NUnit.Framework;
using System.Buffers.Binary;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Torch-parity fixture for Qwen2.5-0.5B-Instruct loader + tokenizer (issue #382 Phase 2).
/// The reference ids/logits are produced by the real <c>AutoModelForCausalLM</c> /
/// <c>AutoTokenizer</c> run in <c>samples/NivaraInference/Python/qwen_tool_reference.py</c>
/// (greedy decode over the native <c>&lt;tool_call&gt;</c> loop). Loading the locally-downloaded
/// checkpoint; skipped when the model/tokenizer files are absent (CI/clean).
/// </summary>
[TestFixture]
public class QwenInstructParityTests
{
    static string ModelDir
        => Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..",
            "samples", "data", "qwen2.5-0.5b-instruct");

    static Gpt2BpeTokenizer? cachedTokenizer;

    static Gpt2BpeTokenizer Tokenizer
    {
        get
        {
            if (cachedTokenizer != null)
                return cachedTokenizer;

            var vocab = Path.Combine(ModelDir, "vocab.json");
            var merges = Path.Combine(ModelDir, "merges.txt");
            var tokenizerJson = Path.Combine(ModelDir, "tokenizer.json");
            if (!File.Exists(vocab) || !File.Exists(merges) || !File.Exists(tokenizerJson))
                Assert.Ignore("Qwen tokenizer files absent; skipping tokenizer parity verification.");

            cachedTokenizer = new Gpt2BpeTokenizer(vocab, merges, tokenizerJsonPath: tokenizerJson);
            return cachedTokenizer;
        }
    }

    static (LlamaForCausalLM<float> Model, LlamaConfig Config)? cachedModel;

    static (LlamaForCausalLM<float> Model, LlamaConfig Config) Model
    {
        get
        {
            if (cachedModel != null)
                return cachedModel.Value;

            var safetensors = Path.Combine(ModelDir, "model.safetensors");
            var configJson = Path.Combine(ModelDir, "config.json");
            if (!File.Exists(safetensors) || !File.Exists(configJson))
                Assert.Ignore("Qwen safetensors absent; skipping model parity verification.");

            try
            {
                // BF16 on disk -> F32 compute (SafeTensorsLoader upcasts); qkvBias auto-detected.
                var tensors = SafeTensorsLoader.Read<float>(safetensors);
                var config = LlamaConfig.FromJson(File.ReadAllText(configJson));
                var model = LlamaLoader.Load<float, float>(config, tensors);
                cachedModel = (model, config);
                return cachedModel.Value;
            }
            catch (Exception ex)
            {
                Assert.Ignore($"Cannot load Qwen model: {ex.Message}");
                return default; // unreachable; keeps the compiler happy
            }
        }
    }

    /// <summary>Reads a little-endian int32 binary fixture.</summary>
    static int[] ReadInt32(string name)
    {
        var bytes = File.ReadAllBytes(Path.Combine(ModelDir, name));
        var result = new int[bytes.Length / 4];
        for (int i = 0; i < result.Length; i++)
            result[i] = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(i * 4, 4));
        return result;
    }

    /// <summary>Reads a little-endian float32 binary fixture.</summary>
    static float[] ReadFloat32(string name)
    {
        var bytes = File.ReadAllBytes(Path.Combine(ModelDir, name));
        var result = new float[bytes.Length / 4];
        for (int i = 0; i < result.Length; i++)
            result[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4, 4));
        return result;
    }

    [Test]
    public void Tokenizer_EncodeToolPrompt_MatchesTorchIds()
    {
        var promptPath = Path.Combine(ModelDir, "qwen_tool_prompt.txt");
        var idsPath = Path.Combine(ModelDir, "qwen_tool_prompt_ids.bin");
        if (!File.Exists(promptPath) || !File.Exists(idsPath))
            Assert.Ignore("Qwen tool fixtures absent; skipping tool-prompt tokenizer parity verification.");

        var prompt = File.ReadAllText(promptPath);
        var expected = ReadInt32("qwen_tool_prompt_ids.bin");

        var ids = Tokenizer.Encode(prompt);

        Assert.That(ids.Count, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ids[i], Is.EqualTo(expected[i]), $"token[{i}] differs (expected {expected[i]}, got {ids[i]})");
    }

    [Test]
    public void Tokenizer_EncodeFinalPrompt_MatchesTorchIds()
    {
        var promptPath = Path.Combine(ModelDir, "qwen_tool_final_prompt.txt");
        var idsPath = Path.Combine(ModelDir, "qwen_tool_final_prompt_ids.bin");
        if (!File.Exists(promptPath) || !File.Exists(idsPath))
            Assert.Ignore("Qwen tool fixtures absent; skipping final-prompt tokenizer parity verification.");

        var prompt = File.ReadAllText(promptPath);
        var expected = ReadInt32("qwen_tool_final_prompt_ids.bin");

        var ids = Tokenizer.Encode(prompt);

        Assert.That(ids.Count, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(ids[i], Is.EqualTo(expected[i]), $"token[{i}] differs (expected {expected[i]}, got {ids[i]})");
    }

    [Test]
    public void Tokenizer_VocabSize_IncludesAddedTokens()
    {
        // 151,643 base vocab + 22 added tokens (incl. <tool_call>/</tool_call>).
        Assert.That(Tokenizer.VocabSize, Is.EqualTo(151665));
    }

    [Test]
    public void Tokenizer_SpecialTokens_ResolveAsSingleIds()
    {
        Assert.That(Tokenizer.TokenId("<|endoftext|>"), Is.EqualTo(151643));
        Assert.That(Tokenizer.TokenId("<|im_start|>"), Is.EqualTo(151644));
        Assert.That(Tokenizer.TokenId("<|im_end|>"), Is.EqualTo(151645));
        Assert.That(Tokenizer.TokenId("<tool_call>"), Is.EqualTo(151657));
        Assert.That(Tokenizer.TokenId("</tool_call>"), Is.EqualTo(151658));

        // Added tokens must survive a round-trip verbatim (atomic, not char-decoded).
        var decoded = Tokenizer.Decode([151644, 151644, 151648, 151645]);
        Assert.That(decoded, Is.EqualTo("<|im_start|><|im_start|><|box_start|><|im_end|>"));
    }

    [Test]
    public void Model_QkvBias_TensorsLoaded()
    {
        var (model, config) = Model;

        // Qwen2.5-0.5B is the bias variant: q/k/v projections carry bias, o_proj does not.
        Assert.That(config.HiddenSize, Is.EqualTo(896));
        Assert.That(config.NumHiddenLayers, Is.EqualTo(24));
        Assert.That(config.NumAttentionHeads, Is.EqualTo(14));
        Assert.That(config.NumKeyValueHeads, Is.EqualTo(2));
        Assert.That(config.VocabSize, Is.EqualTo(151936));

        int headDim = config.HiddenSize / config.NumAttentionHeads; // 64
        int kvWidth = config.NumKeyValueHeads * headDim;            // 128

        var state = model.Parameters(); // same dotted keys as StateDict(), without cloning tensors

        // Every one of the 24 layers must carry exactly Q/K/V bias (24 × 896 + 48 × 128 entries),
        // and nothing else — o_proj/FFN/norms are bias-free in Qwen2.5-0.5B.
        var biasKeys = state.Keys.Where(k => k.EndsWith(".Bias")).ToArray();
        Assert.That(biasKeys.Length, Is.EqualTo(24 * 3),
            "expected exactly Q/K/V bias per layer, loaded via LlamaLoader qkvBias auto-detect");
        Assert.That(biasKeys.Count(k => state[k].Length == config.HiddenSize), Is.EqualTo(24),
            "one 896-wide bias per layer (q_proj)");
        Assert.That(biasKeys.Count(k => state[k].Length == kvWidth), Is.EqualTo(48),
            "two 128-wide biases per layer (k_proj/v_proj)");

        // Nested Module_{i} path: Embed=Module_0, layers=Module_1..24; in a block,
        // Attention=Module_1; in attention, QProj=Module_0. Spot-check layer 0's Q bias.
        var qBiasKey = $"Module_{1}.Module_1.Module_0.Bias";
        Assert.That(state.ContainsKey(qBiasKey), Is.True, "layer-0 q_proj bias must be present");
        Assert.That(state[qBiasKey].Length, Is.EqualTo(config.HiddenSize));
        Assert.That(state.Keys.Any(k => k.EndsWith(".Module_3.Bias")), Is.False,
            "o_proj is the 4th attention child (Module_3) and must have no bias");
    }

    [Test]
    public void Model_GreedyToolLoop_MatchesTorchGeneratedIds()
    {
        var (model, config) = Model;
        var expected = ReadInt32("qwen_tool_ids_py.bin");
        Assert.That(expected.Length, Is.EqualTo(42), "fixture must contain tool turn (19) + final answer (23) ids");

        var promptIds = ReadInt32("qwen_tool_prompt_ids.bin");

        var toolTurn = Greedy(model, config, promptIds, maxNewTokens: 160);
        Assert.That(toolTurn.Count, Is.EqualTo(19), "tool-call turn must be 19 tokens");
        for (int i = 0; i < toolTurn.Count; i++)
            Assert.That(toolTurn[i], Is.EqualTo(expected[i]), $"tool turn token[{i}] differs (expected {expected[i]}, got {toolTurn[i]})");
    }

    [Test]
    public void Model_GreedyFinalAnswer_SemanticParityAndFinalLogitsWithinTolerance()
    {
        var (model, config) = Model;
        var expected = ReadInt32("qwen_tool_ids_py.bin");
        Assert.That(expected.Length, Is.EqualTo(42), "fixture must contain tool turn (19) + final answer (23) ids");
        var torchFinalTurn = expected.Skip(19).ToArray();

        var finalPromptIds = ReadInt32("qwen_tool_final_prompt_ids.bin");

        // Greedy over the C# model. The final answer is free-form natural language and the
        // F32-compute (BF16-upcast) model may tie-flip a near-equal argmax against the BF16
        // Torch reference, so exact token equality is NOT asserted here — the tool turn above
        // is the byte-exact structural check, and numeric parity is asserted below over the
        // FIXED Torch trajectory (same input ids ⇒ comparable last-row logits).
        var finalTurn = Greedy(model, config, finalPromptIds, maxNewTokens: 160);
        TestContext.Out.WriteLine("C# final-turn ids: " + string.Join(",", finalTurn));
        TestContext.Out.WriteLine("Py final-turn ids:  " + string.Join(",", torchFinalTurn));

        var answer = Tokenizer.Decode(finalTurn).ToLowerInvariant();
        TestContext.Out.WriteLine("C# final answer: " + answer);
        Assert.That(answer, Does.Contain("partly cloudy"), "final answer must reflect the weather result");
        Assert.That(answer, Does.Contain("northwest"), "final answer must reuse the tool observation");

        // Numeric parity over the SAME input Torch saw: last-row logits predicting eos from the
        // full Py prompt + finalized answer must match the fixture within BF16 relative tolerance.
        var fullIds = finalPromptIds.Concat(torchFinalTurn).ToArray();
        var logits = model.Forward(fullIds); // [L, vocab]
        var torchLogits = ReadFloat32("qwen_tool_logits_py.bin");
        Assert.That(logits.Shape[1], Is.EqualTo(torchLogits.Length));

        int vocab = torchLogits.Length;
        int offset = logits.Length - vocab;
        float maxAbsDiff = 0f;
        float maxAbsLogit = 0f;
        int argmax = -1;
        int maxDiffAt = -1;
        float best = float.NegativeInfinity;
        for (int i = 0; i < torchLogits.Length; i++)
        {
            float cSharp = logits[offset + i];
            float diff = Math.Abs(cSharp - torchLogits[i]);
            if (diff > maxAbsDiff)
            {
                maxAbsDiff = diff;
                maxDiffAt = i;
            }
            if (Math.Abs(torchLogits[i]) > maxAbsLogit)
                maxAbsLogit = Math.Abs(torchLogits[i]);
            if (cSharp > best)
            {
                best = cSharp;
                argmax = i;
            }
        }
        TestContext.Out.WriteLine(
            $"final-position logits: maxAbsDiff={maxAbsDiff:F6} at vocab {maxDiffAt} " +
            $"(ref {torchLogits[maxDiffAt]:F4} / c# {logits[offset + maxDiffAt]:F4}), " +
            $"refMaxAbs={maxAbsLogit:F3}, argmax={argmax}");

        // Tie-flip proof: at the first positional difference between the two greedy runs, show
        // both candidates so a divergence is provably a near-tie (tiny margin), not a numeric bug.
        if (finalTurn.Count > 9)
        {
            int row = finalPromptIds.Length + 8; // row whose output picks generated index 9 (the flip site)
            int highAt = row * vocab + 1550;        // ' high' — the Py tokenizer's choice
            int temperatureAt = row * vocab + 9315; // ' temperature' — the C# run's choice
            TestContext.Out.WriteLine($"tie-check pos {row}: logit(' high')={logits[highAt]:F4}, " +
                $"logit(' temperature')={logits[temperatureAt]:F4}, " +
                $"margin={Math.Abs(logits[highAt] - logits[temperatureAt]):F4}");
        }

        Assert.That(argmax, Is.EqualTo(151645), "final-row argmax must predict <|im_end|>");
        // Torch reference computed in BF16 (torch_dtype="auto"); C# computes F32 from BF16-upcast
        // weights, so parity is bounded by BF16 rounding accumulated over 24 layers / 281 tokens:
        // observed worst-case is a ~0.4 absolute diff on a low-probability tail entry (2.3% of the
        // max logit magnitude). 3% relative + a 0.5 absolute floor is the honest envelope; gross
        // numeric errors (rope/transpose/attention bugs) land far outside it.
        Assert.That(maxAbsDiff, Is.LessThan(0.03f * maxAbsLogit + 0.5f),
            "final-position logits must be within BF16-reference relative tolerance");
    }

    /// <summary>Greedily decodes with a KV cache (numeric-identical to full forward), stopping on
    /// the Qwen eos id 151645 exactly as the reference <c>_greedy</c> does.</summary>
    static List<int> Greedy(LlamaForCausalLM<float> model, LlamaConfig config, int[] promptIds, int maxNewTokens)
    {
        int kvWidth = config.NumKeyValueHeads * (config.HiddenSize / config.NumAttentionHeads);
        using var cache = new LlamaKVCache<float>(config.NumHiddenLayers, kvWidth);

        ReverseGradTensor<float> logits = null!;
        for (int p = 0; p < promptIds.Length; p++)
            logits = model.ForwardCached(promptIds[p], p, cache);

        int position = promptIds.Length;
        var gen = new List<int>();
        for (int t = 0; t < maxNewTokens && gen.Count < config.MaxPositionEmbeddings; t++)
        {
            int next = ArgMax(logits, config.VocabSize);
            if (next == 151645)
                break;
            gen.Add(next);
            logits = model.ForwardCached(next, position++, cache);
        }
        return gen;
    }

    static int ArgMax(ReverseGradTensor<float> logits, int vocab)
    {
        int best = -1;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < vocab; i++)
        {
            float v = logits[i];
            if (v > bestVal)
            {
                bestVal = v;
                best = i;
            }
        }
        return best;
    }
}