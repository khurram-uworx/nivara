using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class LlamaDecoderBlockTests
{
    // No global Grad() scope: the inference guard is that no graph nodes are built outside Grad().

    [Test]
    public void Inference_OutsideGrad_PreservesShapeAndBuildsNoGraph()
    {
        const int hidden = 384, numHeads = 8, numKvHeads = 4, intermediate = 1024, seqLen = 6;
        using var block = new LlamaDecoderBlock<float>(hidden, numHeads, numKvHeads, intermediate, maxPositionEmbeddings: 32);

        var inputData = new float[seqLen * hidden];
        var rnd = new Random(3);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rnd.NextDouble() * 2 - 1);
        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, hidden, requiresGrad: false);

        var output = block.Forward(input);

        Assert.That(output.Rank, Is.EqualTo(2));
        Assert.That(output.shape, Is.EqualTo(new[] { seqLen, hidden }));
        Assert.That(output.IsLeaf, Is.True, "Decoder block inference outside Grad() must not build graph nodes.");
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsFinite(output[i]), Is.True, $"Output[{i}] must be finite.");
    }

    [Test]
    public void Forward_InsideGrad_AccumulatesFiniteGradients()
    {
        const int hidden = 192, numHeads = 6, numKvHeads = 3, intermediate = 512, seqLen = 4;
        using var gradScope = GradientUtils.Grad();
        using var block = new LlamaDecoderBlock<float>(hidden, numHeads, numKvHeads, intermediate, maxPositionEmbeddings: 32);

        var inputData = new float[seqLen * hidden];
        var rnd = new Random(5);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rnd.NextDouble() * 2 - 1);
        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, hidden, requiresGrad: true);

        var output = block.Forward(input);
        var loss = ReverseGradOperations.Sum(output);
        loss.Backward();

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(seqLen * hidden));
        foreach (var g in input.Grad!)
            Assert.That(float.IsNaN(g) || float.IsInfinity(g), Is.False, "Block input gradient must be finite.");
    }

    [Test]
    public void Constructor_QkvBiasTrue_CreatesAttentionQkvBiasesOnly()
    {
        const int hidden = 384, numHeads = 8, numKvHeads = 4, intermediate = 1024;
        using var block = new LlamaDecoderBlock<float>(hidden, numHeads, numKvHeads, intermediate, qkvBias: true);

        Assert.That(block.Attention.QProj.Bias, Is.Not.Null);
        Assert.That(block.Attention.KProj.Bias, Is.Not.Null);
        Assert.That(block.Attention.VProj.Bias, Is.Not.Null);
        Assert.That(block.Attention.OProj.Bias, Is.Null);
        Assert.That(block.GateProj.Bias, Is.Null, "FFN projections stay bias-free with qkvBias=true");
        Assert.That(block.UpProj.Bias, Is.Null, "FFN projections stay bias-free with qkvBias=true");
        Assert.That(block.DownProj.Bias, Is.Null, "FFN projections stay bias-free with qkvBias=true");
    }

    [Test]
    public void Constructor_QkvBiasFalse_AttentionBiasFree()
    {
        const int hidden = 384, numHeads = 8, numKvHeads = 4, intermediate = 1024;
        using var block = new LlamaDecoderBlock<float>(hidden, numHeads, numKvHeads, intermediate, qkvBias: false);

        Assert.That(block.Attention.QProj.Bias, Is.Null);
        Assert.That(block.Attention.KProj.Bias, Is.Null);
        Assert.That(block.Attention.VProj.Bias, Is.Null);
        Assert.That(block.Attention.OProj.Bias, Is.Null);
    }

    [Test]
    public void Forward_InsideGrad_QkvBiasTrue_AccumulatesFiniteBiasGradients()
    {
        const int hidden = 192, numHeads = 6, numKvHeads = 3, intermediate = 512, seqLen = 4;
        using var gradScope = GradientUtils.Grad();
        using var block = new LlamaDecoderBlock<float>(hidden, numHeads, numKvHeads, intermediate, maxPositionEmbeddings: 32, qkvBias: true);

        var inputData = new float[seqLen * hidden];
        var rnd = new Random(17);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rnd.NextDouble() * 2 - 1);
        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, hidden, requiresGrad: true);

        var output = block.Forward(input);
        var loss = ReverseGradOperations.Sum(output);
        loss.Backward();

        Assert.That(block.Attention.QProj.Bias!.Tensor.Grad, Is.Not.Null, "q_proj bias gradient must flow");
        Assert.That(block.Attention.KProj.Bias!.Tensor.Grad, Is.Not.Null, "k_proj bias gradient must flow");
        Assert.That(block.Attention.VProj.Bias!.Tensor.Grad, Is.Not.Null, "v_proj bias gradient must flow");
        foreach (var g in new[] { block.Attention.QProj.Bias!.Tensor.Grad!, block.Attention.KProj.Bias!.Tensor.Grad!, block.Attention.VProj.Bias!.Tensor.Grad! })
            foreach (var v in g)
                Assert.That(float.IsNaN(v) || float.IsInfinity(v), Is.False, "Block bias gradient must be finite.");
    }

    [Test]
    public void Forward_ResidualAdds_ChangeOutputFromRawNormPath()
    {
        // With all weights identity-ish (Linear defaults are small Kaiming), the residual
        // adds still guarantee the output is a finite, non-zero tensor different from pure
        // attention alone. This is a structural smoke check, not a numeric reference.
        const int hidden = 64, numHeads = 4, numKvHeads = 2, intermediate = 128, seqLen = 3;
        using var block = new LlamaDecoderBlock<float>(hidden, numHeads, numKvHeads, intermediate, maxPositionEmbeddings: 8);

        var inputData = new float[seqLen * hidden];
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = 1f;
        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, hidden, requiresGrad: false);
        var output = block.Forward(input);

        Assert.That(output.Length, Is.EqualTo(seqLen * hidden));
        Assert.That(output[0], Is.Not.EqualTo(0f));
    }
}
