using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class LlamaCausalAttentionTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void LlamaCausalAttention_GqaRoPeCausalMask_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("llama_attn_input.bin");
        var qw = TestHelpers.LoadBin("llama_attn_qw.bin");
        var kw = TestHelpers.LoadBin("llama_attn_kw.bin");
        var vw = TestHelpers.LoadBin("llama_attn_vw.bin");
        var ow = TestHelpers.LoadBin("llama_attn_ow.bin");
        var expectedOutput = TestHelpers.LoadBin("llama_attn_output.bin");
        var expectedInputGrad = TestHelpers.LoadBin("llama_attn_input_grad.bin");

        using var attn = new LlamaCausalAttention<float>(
            hiddenSize: 64, numHeads: 4, numKeyValueHeads: 2, maxPositionEmbeddings: 16, ropeTheta: 10000f);
        attn.QProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(qw, 64, 64, requiresGrad: false);
        attn.KProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(kw, 32, 64, requiresGrad: false);
        attn.VProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(vw, 32, 64, requiresGrad: false);
        attn.OProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(ow, 64, 64, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: true);
        inputTensor.Reshape(5, 64);
        var output = attn.Forward(inputTensor);
        ReverseGradOperations.Sum(output).Backward();

        Assert.That(output.Shape, Is.EqualTo(new[] { 5, 64 }));
        TestHelpers.AssertTensorEqual(expectedOutput, TestHelpers.ExtractOutput(output), label: "LlamaCausalAttention_output");
        TestHelpers.AssertTensorEqual(expectedInputGrad, TestHelpers.ExtractGrad(inputTensor), absTol: 1e-4f, relTol: 1e-3f, label: "LlamaCausalAttention_input_grad");
    }

    [Test]
    public void LlamaCausalAttention_QkvBias_GqaRoPeCausalMask_MatchesPyTorch()
    {
        // Qwen2-style variant (#384): q/k/v projections carry a bias vector.
        var input = TestHelpers.LoadBin("llama_attn_bias_input.bin");
        var qw = TestHelpers.LoadBin("llama_attn_bias_qw.bin");
        var kw = TestHelpers.LoadBin("llama_attn_bias_kw.bin");
        var vw = TestHelpers.LoadBin("llama_attn_bias_vw.bin");
        var ow = TestHelpers.LoadBin("llama_attn_bias_ow.bin");
        var qb = TestHelpers.LoadBin("llama_attn_bias_qb.bin");
        var kb = TestHelpers.LoadBin("llama_attn_bias_kb.bin");
        var vb = TestHelpers.LoadBin("llama_attn_bias_vb.bin");
        var expectedOutput = TestHelpers.LoadBin("llama_attn_bias_output.bin");
        var expectedInputGrad = TestHelpers.LoadBin("llama_attn_bias_input_grad.bin");
        var expectedQBiasGrad = TestHelpers.LoadBin("llama_attn_bias_q_bias_grad.bin");
        var expectedKBiasGrad = TestHelpers.LoadBin("llama_attn_bias_k_bias_grad.bin");
        var expectedVBiasGrad = TestHelpers.LoadBin("llama_attn_bias_v_bias_grad.bin");

        using var attn = new LlamaCausalAttention<float>(
            hiddenSize: 64, numHeads: 4, numKeyValueHeads: 2, maxPositionEmbeddings: 16, ropeTheta: 10000f, qkvBias: true);
        attn.QProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(qw, 64, 64, requiresGrad: false);
        attn.KProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(kw, 32, 64, requiresGrad: false);
        attn.VProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(vw, 32, 64, requiresGrad: false);
        attn.OProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(ow, 64, 64, requiresGrad: false);
        attn.QProj.Bias!.Tensor = ReverseGradTensor<float>.FromArray(qb, requiresGrad: true);
        attn.KProj.Bias!.Tensor = ReverseGradTensor<float>.FromArray(kb, requiresGrad: true);
        attn.VProj.Bias!.Tensor = ReverseGradTensor<float>.FromArray(vb, requiresGrad: true);
        Assert.That(attn.OProj.Bias, Is.Null, "o_proj is bias-free in Qwen2-style models");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: true);
        inputTensor.Reshape(5, 64);
        var output = attn.Forward(inputTensor);
        ReverseGradOperations.Sum(output).Backward();

        Assert.That(output.Shape, Is.EqualTo(new[] { 5, 64 }));
        TestHelpers.AssertTensorEqual(expectedOutput, TestHelpers.ExtractOutput(output), label: "LlamaCausalAttention_qkvBias_output");
        TestHelpers.AssertTensorEqual(expectedInputGrad, TestHelpers.ExtractGrad(inputTensor), absTol: 1e-4f, relTol: 1e-3f, label: "LlamaCausalAttention_qkvBias_input_grad");
        TestHelpers.AssertTensorEqual(expectedQBiasGrad, TestHelpers.ExtractGrad(attn.QProj.Bias!.Tensor), absTol: 1e-4f, relTol: 1e-3f, label: "LlamaCausalAttention_qkvBias_q_bias_grad");
        TestHelpers.AssertTensorEqual(expectedKBiasGrad, TestHelpers.ExtractGrad(attn.KProj.Bias!.Tensor), absTol: 1e-4f, relTol: 1e-3f, label: "LlamaCausalAttention_qkvBias_k_bias_grad");
        TestHelpers.AssertTensorEqual(expectedVBiasGrad, TestHelpers.ExtractGrad(attn.VProj.Bias!.Tensor), absTol: 1e-4f, relTol: 1e-3f, label: "LlamaCausalAttention_qkvBias_v_bias_grad");
    }
}