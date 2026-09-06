using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class LlamaDecoderBlockTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void LlamaDecoderBlock_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("llama_decoder_input.bin");
        var inGamma = TestHelpers.LoadBin("llama_decoder_in_gamma.bin");
        var postGamma = TestHelpers.LoadBin("llama_decoder_post_gamma.bin");
        var qw = TestHelpers.LoadBin("llama_decoder_qw.bin");
        var kw = TestHelpers.LoadBin("llama_decoder_kw.bin");
        var vw = TestHelpers.LoadBin("llama_decoder_vw.bin");
        var ow = TestHelpers.LoadBin("llama_decoder_ow.bin");
        var gatew = TestHelpers.LoadBin("llama_decoder_gatew.bin");
        var upw = TestHelpers.LoadBin("llama_decoder_upw.bin");
        var downw = TestHelpers.LoadBin("llama_decoder_downw.bin");
        var expectedOutput = TestHelpers.LoadBin("llama_decoder_output.bin");
        var expectedInputGrad = TestHelpers.LoadBin("llama_decoder_input_grad.bin");

        using var block = new LlamaDecoderBlock<float>(
            hiddenSize: 32, numHeads: 4, numKeyValueHeads: 2, intermediateSize: 48,
            rmsNormEps: 1e-5f, maxPositionEmbeddings: 16, ropeTheta: 10000f);
        block.InputNorm.Weight!.Tensor = ReverseGradTensor<float>.FromArray(inGamma, requiresGrad: false);
        block.PostNorm.Weight!.Tensor = ReverseGradTensor<float>.FromArray(postGamma, requiresGrad: false);
        block.Attention.QProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(qw, 32, 32, requiresGrad: false);
        block.Attention.KProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(kw, 16, 32, requiresGrad: false);
        block.Attention.VProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(vw, 16, 32, requiresGrad: false);
        block.Attention.OProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(ow, 32, 32, requiresGrad: false);
        block.GateProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(gatew, 48, 32, requiresGrad: false);
        block.UpProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(upw, 48, 32, requiresGrad: false);
        block.DownProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(downw, 32, 48, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: true);
        inputTensor.Reshape(4, 32);
        var output = block.Forward(inputTensor);
        ReverseGradOperations.Sum(output).Backward();

        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 32 }));
        TestHelpers.AssertTensorEqual(expectedOutput, TestHelpers.ExtractOutput(output), label: "LlamaDecoderBlock_output");
        TestHelpers.AssertTensorEqual(expectedInputGrad, TestHelpers.ExtractGrad(inputTensor), relTol: 1e-3f, label: "LlamaDecoderBlock_input_grad");
    }

    [Test]
    public void LlamaDecoderBlock_QkvBias_MatchesPyTorch()
    {
        // Qwen2-style variant (#384): q/k/v projections carry a bias vector.
        var input = TestHelpers.LoadBin("llama_decoder_bias_input.bin");
        var inGamma = TestHelpers.LoadBin("llama_decoder_bias_in_gamma.bin");
        var postGamma = TestHelpers.LoadBin("llama_decoder_bias_post_gamma.bin");
        var qw = TestHelpers.LoadBin("llama_decoder_bias_qw.bin");
        var kw = TestHelpers.LoadBin("llama_decoder_bias_kw.bin");
        var vw = TestHelpers.LoadBin("llama_decoder_bias_vw.bin");
        var ow = TestHelpers.LoadBin("llama_decoder_bias_ow.bin");
        var qb = TestHelpers.LoadBin("llama_decoder_bias_qb.bin");
        var kb = TestHelpers.LoadBin("llama_decoder_bias_kb.bin");
        var vb = TestHelpers.LoadBin("llama_decoder_bias_vb.bin");
        var gatew = TestHelpers.LoadBin("llama_decoder_bias_gatew.bin");
        var upw = TestHelpers.LoadBin("llama_decoder_bias_upw.bin");
        var downw = TestHelpers.LoadBin("llama_decoder_bias_downw.bin");
        var expectedOutput = TestHelpers.LoadBin("llama_decoder_bias_output.bin");
        var expectedInputGrad = TestHelpers.LoadBin("llama_decoder_bias_input_grad.bin");
        var expectedQBiasGrad = TestHelpers.LoadBin("llama_decoder_bias_q_bias_grad.bin");
        var expectedKBiasGrad = TestHelpers.LoadBin("llama_decoder_bias_k_bias_grad.bin");
        var expectedVBiasGrad = TestHelpers.LoadBin("llama_decoder_bias_v_bias_grad.bin");

        using var block = new LlamaDecoderBlock<float>(
            hiddenSize: 32, numHeads: 4, numKeyValueHeads: 2, intermediateSize: 48,
            rmsNormEps: 1e-5f, maxPositionEmbeddings: 16, ropeTheta: 10000f, qkvBias: true);
        block.InputNorm.Weight!.Tensor = ReverseGradTensor<float>.FromArray(inGamma, requiresGrad: false);
        block.PostNorm.Weight!.Tensor = ReverseGradTensor<float>.FromArray(postGamma, requiresGrad: false);
        block.Attention.QProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(qw, 32, 32, requiresGrad: false);
        block.Attention.KProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(kw, 16, 32, requiresGrad: false);
        block.Attention.VProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(vw, 16, 32, requiresGrad: false);
        block.Attention.OProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(ow, 32, 32, requiresGrad: false);
        block.Attention.QProj.Bias!.Tensor = ReverseGradTensor<float>.FromArray(qb, requiresGrad: true);
        block.Attention.KProj.Bias!.Tensor = ReverseGradTensor<float>.FromArray(kb, requiresGrad: true);
        block.Attention.VProj.Bias!.Tensor = ReverseGradTensor<float>.FromArray(vb, requiresGrad: true);
        block.GateProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(gatew, 48, 32, requiresGrad: false);
        block.UpProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(upw, 48, 32, requiresGrad: false);
        block.DownProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(downw, 32, 48, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: true);
        inputTensor.Reshape(4, 32);
        var output = block.Forward(inputTensor);
        ReverseGradOperations.Sum(output).Backward();

        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 32 }));
        TestHelpers.AssertTensorEqual(expectedOutput, TestHelpers.ExtractOutput(output), absTol: 1e-4f, relTol: 1e-3f, label: "LlamaDecoderBlock_qkvBias_output");
        TestHelpers.AssertTensorEqual(expectedInputGrad, TestHelpers.ExtractGrad(inputTensor), relTol: 1e-3f, label: "LlamaDecoderBlock_qkvBias_input_grad");
        TestHelpers.AssertTensorEqual(expectedQBiasGrad, TestHelpers.ExtractGrad(block.Attention.QProj.Bias!.Tensor), absTol: 1e-4f, relTol: 1e-3f, label: "LlamaDecoderBlock_qkvBias_q_bias_grad");
        TestHelpers.AssertTensorEqual(expectedKBiasGrad, TestHelpers.ExtractGrad(block.Attention.KProj.Bias!.Tensor), absTol: 1e-4f, relTol: 1e-3f, label: "LlamaDecoderBlock_qkvBias_k_bias_grad");
        TestHelpers.AssertTensorEqual(expectedVBiasGrad, TestHelpers.ExtractGrad(block.Attention.VProj.Bias!.Tensor), absTol: 1e-4f, relTol: 1e-3f, label: "LlamaDecoderBlock_qkvBias_v_bias_grad");
    }
}