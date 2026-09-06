using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class LlamaCausalAttentionTests
{
    // No global Grad() scope here deliberately: the inference-path guard is that no graph
    // nodes are built outside Grad(), which is the model's default execution mode.

    [Test]
    public void Inference_OutsideGrad_PreservesShapeAndBuildsNoGraph()
    {
        const int hidden = 768, numHeads = 12, numKvHeads = 4, seqLen = 8;
        using var attn = new LlamaCausalAttention<float>(hidden, numHeads, numKvHeads, maxPositionEmbeddings: 64);

        var inputData = new float[seqLen * hidden];
        var rnd = new Random(7);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rnd.NextDouble() * 2 - 1);
        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, hidden, requiresGrad: false);

        var output = attn.Forward(input);

        Assert.That(output.Rank, Is.EqualTo(2));
        Assert.That(output.shape, Is.EqualTo(new[] { seqLen, hidden }));
        Assert.That(output.IsLeaf, Is.True, "Inference forward outside Grad() must not build graph nodes.");
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsFinite(output[i]), Is.True, $"Output[{i}] must be finite.");
    }

    [Test]
    public void Forward_InsideGrad_AccumulatesGradientsOnAllProjections()
    {
        const int hidden = 192, numHeads = 6, numKvHeads = 3, seqLen = 4;
        using var gradScope = GradientUtils.Grad();
        using var attn = new LlamaCausalAttention<float>(hidden, numHeads, numKvHeads, maxPositionEmbeddings: 32);

        var inputData = new float[seqLen * hidden];
        var rnd = new Random(11);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rnd.NextDouble() * 2 - 1);
        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, hidden, requiresGrad: true);

        var output = attn.Forward(input);
        var loss = Nivara.AutoDiff.Operations.ReverseGradOperations.Sum(output);
        loss.Backward();

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(seqLen * hidden));
        foreach (var g in input.Grad!)
            Assert.That(float.IsNaN(g) || float.IsInfinity(g), Is.False, "Attention input gradient must be finite.");
    }

    [Test]
    public void Constructor_QkvBiasTrue_CreatesQkvBiasesOnly()
    {
        const int hidden = 192, numHeads = 6, numKvHeads = 3;
        using var attn = new LlamaCausalAttention<float>(hidden, numHeads, numKvHeads, qkvBias: true);

        Assert.That(attn.QProj.Bias, Is.Not.Null);
        Assert.That(attn.KProj.Bias, Is.Not.Null);
        Assert.That(attn.VProj.Bias, Is.Not.Null);
        Assert.That(attn.OProj.Bias, Is.Null, "o_proj is bias-free in canonical and Qwen2-style attention");

        Assert.That(attn.QProj.Bias!.Shape, Is.EqualTo(new[] { 1, numHeads * (hidden / numHeads) }));
        Assert.That(attn.KProj.Bias!.Shape, Is.EqualTo(new[] { 1, numKvHeads * (hidden / numHeads) }));
        Assert.That(attn.VProj.Bias!.Shape, Is.EqualTo(new[] { 1, numKvHeads * (hidden / numHeads) }));
    }

    [Test]
    public void Constructor_QkvBiasFalse_AllProjectionsBiasFree()
    {
        const int hidden = 192, numHeads = 6, numKvHeads = 3;
        using var attn = new LlamaCausalAttention<float>(hidden, numHeads, numKvHeads, qkvBias: false);

        Assert.That(attn.QProj.Bias, Is.Null);
        Assert.That(attn.KProj.Bias, Is.Null);
        Assert.That(attn.VProj.Bias, Is.Null);
        Assert.That(attn.OProj.Bias, Is.Null);
    }

    [Test]
    public void Forward_InsideGrad_QkvBiasTrue_AccumulatesFiniteBiasGradients()
    {
        const int hidden = 192, numHeads = 6, numKvHeads = 3, seqLen = 4;
        using var gradScope = GradientUtils.Grad();
        using var attn = new LlamaCausalAttention<float>(hidden, numHeads, numKvHeads, maxPositionEmbeddings: 32, qkvBias: true);

        var inputData = new float[seqLen * hidden];
        var rnd = new Random(21);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rnd.NextDouble() * 2 - 1);
        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, hidden, requiresGrad: true);

        var output = attn.Forward(input);
        var loss = Nivara.AutoDiff.Operations.ReverseGradOperations.Sum(output);
        loss.Backward();

        Assert.That(attn.QProj.Bias!.Tensor.Grad, Is.Not.Null, "q_proj bias gradient must flow");
        Assert.That(attn.KProj.Bias!.Tensor.Grad, Is.Not.Null, "k_proj bias gradient must flow");
        Assert.That(attn.VProj.Bias!.Tensor.Grad, Is.Not.Null, "v_proj bias gradient must flow");
        foreach (var g in new[] { attn.QProj.Bias!.Tensor.Grad!, attn.KProj.Bias!.Tensor.Grad!, attn.VProj.Bias!.Tensor.Grad! })
            foreach (var v in g)
                Assert.That(float.IsNaN(v) || float.IsInfinity(v), Is.False, "Attention bias gradient must be finite.");
    }

    [Test]
    public void ForwardCached_QkvBiasTrue_MatchesFullForward()
    {
        // With qkvBias=true the bias must be applied consistently in the cached single-token
        // path and the full-sequence path, so the two outputs agree.
        const int hidden = 128, numHeads = 8, numKvHeads = 4, seqLen = 6;
        using var attn = new LlamaCausalAttention<float>(hidden, numHeads, numKvHeads, maxPositionEmbeddings: 32, qkvBias: true);

        var inputData = new float[seqLen * hidden];
        var rnd = new Random(33);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rnd.NextDouble() * 2 - 1);

        var fullInput = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, hidden, requiresGrad: false);
        var fullOutput = attn.Forward(fullInput);

        int kvWidth = numKvHeads * (hidden / numHeads);
        var kCache = new float[seqLen * kvWidth];
        var vCache = new float[seqLen * kvWidth];
        var stepOutputs = new ReverseGradTensor<float>[seqLen];
        for (int p = 0; p < seqLen; p++)
        {
            var tokenData = new float[hidden];
            Buffer.BlockCopy(inputData, p * hidden * sizeof(float), tokenData, 0, hidden * sizeof(float));
            var token = ReverseGradTensor<float>.FromArray(tokenData, requiresGrad: false);
            token.Reshape(1, hidden);
            stepOutputs[p] = attn.ForwardCached(token, p, kCache, vCache, p);
        }

        for (int p = 0; p < seqLen; p++)
            for (int d = 0; d < hidden; d++)
            {
                float full = fullOutput[p * hidden + d];
                float step = stepOutputs[p][d];
                Assert.That(step, Is.EqualTo(full).Within(1e-5f), $"Cached vs full mismatch at token {p}, dim {d}.");
            }
    }
}
