using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using Nivara.Primitives;
using Nivara.Samples;
using System.Diagnostics;
using System.Drawing;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NivaraInference;

class Program
{
    static int Main(string[] args)
    {
        Console.WriteLine("=== Nivara HuggingFace Inference ===");
        Console.WriteLine();

        string modelType = args.Length > 0 ? args[0] : "";
        string resolvedType = modelType switch
        {
            "smollm" => "smollm-135m",
            "qwen" => "qwen2.5-0.5b-instruct",
            _ => modelType
        };
        string precision = "f32";
        string mode = "";
        bool simdWiden = false;
        bool noKvCache = false;
        bool force = false;
        int teacherExamples = 0;
        int seed = 42;
        string text = "";
        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--precision" && i + 1 < args.Length)
            {
                precision = args[i + 1].ToLowerInvariant() switch
                {
                    "f32" or "float" => "f32",
                    "bf16" or "bfloat16" => "bf16",
                    "fp16" or "f16" or "half" => "fp16",
                    var other => other
                };
                i++;
            }
            else if (args[i] is "bf16" or "bfloat16")
            {
                precision = "bf16";
            }
            else if (args[i] is "fp16" or "f16" or "half")
            {
                precision = "fp16";
            }
            else if (args[i] == "--simd-widen")
                simdWiden = true;
            else if (args[i] == "--no-kv-cache")
                noKvCache = true;
            else if (args[i] == "--force")
                force = true;
            else if (args[i] == "--teacher-examples" && i + 1 < args.Length)
            {
                int.TryParse(args[i + 1], out teacherExamples);
                i++;
            }
            else if (args[i] == "--seed" && i + 1 < args.Length)
            {
                int.TryParse(args[i + 1], out seed);
                i++;
            }
            else if (args[i] == "--text" && i + 1 < args.Length)
            {
                text = args[i + 1];
                i++;
            }
            else if (mode.Length == 0)
                mode = args[i];
        }

        if (string.IsNullOrEmpty(modelType) || modelType is "-h" or "--help")
        {
            Console.WriteLine("Usage: NivaraInference <mobilenet_v2|resnet18|minilm|distilbert|distilbert_sst|smollm|qwen> [--precision f32|bf16|fp16] [benchmark|similarity|compare|compare_diag|predict|generate|tools|distill|image-path]");
            Console.WriteLine();
            Console.WriteLine("Modes:");
            Console.WriteLine("  benchmark         Run timed inference passes and report median timing");
            Console.WriteLine("  compare           Run forward pass on shared input, print logits for Python comparison");
            Console.WriteLine("  compare_diag      Step-by-step diagnostics, save intermediates to samples/data/diag/");
            Console.WriteLine("  predict           Interactive sentiment REPL (distilbert_sst)");
            Console.WriteLine("  generate          Greedy causal-LM generation (smollm)");
            Console.WriteLine("  ab                A/B scalar vs widen comparison (smollm only)");
            Console.WriteLine("  tools             Native Qwen2.5 function calling (getWeather tool loop)");
            Console.WriteLine("  distill           Teacher distillation into a tiny sentiment classifier");
            Console.WriteLine("  <image-path>      Run inference on a single image");
            Console.WriteLine();
            Console.WriteLine("Qwen options:");
            Console.WriteLine("  --text \"...\"      Override the tools-mode user prompt (default: Paris weather)");
            Console.WriteLine("  --no-kv-cache      Disable the KV cache (re-run full forward each token)");
            Console.WriteLine("  --teacher-examples N  Distill: annotate the first N train sentences (default: all)");
            Console.WriteLine("  --force            Distill: ignore the resumable teacher-label cache and recompute");
            Console.WriteLine("  --seed N           Distill: seed accepted for future use (Kaiming init is unseeded)");
            Console.WriteLine();
            Console.WriteLine("Precision (text models only):");
            Console.WriteLine("  --precision f32   Full float32 weights (qwen default: BF16-on-disk, SIMD-widened to F32 at load).");
            Console.WriteLine("  --precision bf16  BFloat16 (half weight memory). Bare 'bf16' also accepted.");
            Console.WriteLine("  --precision fp16  Half / fp16 (half weight memory). Bare 'fp16'|'half' also accepted.");
            Console.WriteLine("                    (bf16/fp16 are rejected for qwen; student training is always F32).");
            Console.WriteLine();
            Console.WriteLine("SIMD (narrow-float models):");
            Console.WriteLine("  --simd-widen      Enable widen-compute-narrow SIMD kernels for BFloat16/Half");
            Console.WriteLine("                    (NivaraPrimitives.UseWidenSimd). A/B against scalar: run");
            Console.WriteLine("                    once without and once with this flag, or use the 'ab' mode.");
            return 1;
        }

        string modelDir = Path.Combine("samples", "data", resolvedType);
        string modelPath = Path.Combine(modelDir, "model.safetensors");

        if (!File.Exists(modelPath))
        {
            Console.Error.WriteLine($"Model file not found: {modelPath}");
            return 1;
        }

        bool benchmark = mode == "benchmark";
        bool compare = mode == "compare";
        bool compareDiag = mode == "compare_diag";
        bool fp16 = precision == "fp16";
        bool bf16 = precision == "bf16";
        bool isQwen = modelType == "qwen";

        if (isQwen && bf16)
        {
            Console.Error.WriteLine("Qwen2.5 precision error: bf16 is not supported for the qwen mode. Use --precision f32 (BF16-on-disk, widened to F32 at load).");
            return 1;
        }
        if (isQwen && fp16)
        {
            Console.Error.WriteLine("Qwen2.5 precision error: fp16 is not supported for the qwen mode. Use --precision f32 (BF16-on-disk, widened to F32 at load).");
            return 1;
        }

        Console.WriteLine($"Loading weights ({precision}) from {Path.GetFileName(modelPath)}...");
        var loadSw = Stopwatch.StartNew();
        var tensors = SafeTensorsLoader.Read(modelPath);
        loadSw.Stop();
        Console.WriteLine($"  SafeTensors parse (F32): {loadSw.ElapsedMilliseconds} ms ({tensors.Count} tensors)");
        Console.WriteLine();

        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensorsBf16 = null!;
        if (bf16)
        {
            var loadSwBf16 = Stopwatch.StartNew();
            tensorsBf16 = SafeTensorsLoader.Read<BFloat16>(modelPath);
            loadSwBf16.Stop();
            Console.WriteLine($"  SafeTensors parse (BFloat16): {loadSwBf16.ElapsedMilliseconds} ms ({tensorsBf16.Count} tensors)");
            Console.WriteLine();
        }

        Dictionary<string, (Half[] Data, int[] Shape)> tensorsHalf = null!;
        if (fp16)
        {
            var loadSwFp16 = Stopwatch.StartNew();
            tensorsHalf = SafeTensorsLoader.Read<Half>(modelPath);
            loadSwFp16.Stop();
            Console.WriteLine($"  SafeTensors parse (Half): {loadSwFp16.ElapsedMilliseconds} ms ({tensorsHalf.Count} tensors)");
            Console.WriteLine();
        }

        if (simdWiden)
            NivaraPrimitives.UseWidenSimd = true;

        switch (modelType)
        {
            case "mobilenet_v2":
                if (compareDiag) return RunCompareDiag(tensors, "mobilenet_v2");
                if (compare) return RunCompare(tensors, "mobilenet_v2");
                return benchmark ? RunMobileNetV2Benchmark(tensors) : RunMobileNetV2Inference(tensors, mode);
            case "resnet18":
                if (compareDiag) return RunCompareDiag(tensors, "resnet18");
                if (compare) return RunCompare(tensors, "resnet18");
                return benchmark ? RunResNet18Benchmark(tensors) : RunResNet18Inference(tensors, mode);
            case "minilm":
                if (bf16) return benchmark ? BenchmarkMiniLM(tensorsBf16, "BFloat16") : RunMiniLMBFloat16(tensorsBf16);
                if (fp16) return benchmark ? BenchmarkMiniLM(tensorsHalf, "Half") : RunMiniLMHalf(tensorsHalf);
                if (compare) return RunMiniLMCompare(tensors);
                bool similarity = mode == "similarity";
                return similarity ? RunMiniLMSimilarity(tensors) : benchmark ? BenchmarkMiniLM(tensors, "F32") : RunMiniLMInference(tensors);
            case "distilbert":
                if (bf16) return benchmark ? BenchmarkDistilBert(tensorsBf16, "BFloat16") : RunDistilBertBFloat16(tensorsBf16);
                if (fp16) return benchmark ? BenchmarkDistilBert(tensorsHalf, "Half") : RunDistilBertHalf(tensorsHalf);
                if (compare) return RunDistilBertCompare(tensors);
                return benchmark ? BenchmarkDistilBert(tensors, "F32") : RunDistilBertInference(tensors);
            case "distilbert_sst":
                if (bf16) return benchmark ? BenchmarkDistilBertSst(tensorsBf16, "BFloat16") : RunDistilBertSstBFloat16(tensorsBf16);
                if (fp16) return benchmark ? BenchmarkDistilBertSst(tensorsHalf, "Half") : RunDistilBertSstHalf(tensorsHalf);
                if (compare) return RunDistilBertSstCompare(tensors);
                if (mode == "predict") return RunDistilBertSstPredict(tensors);
                return benchmark ? BenchmarkDistilBertSst(tensors, "F32") : RunDistilBertSstInference(tensors);
            case "smollm":
                if (mode == "ab")
                {
                    if (bf16) return SmolLMAb(tensorsBf16);
                    if (fp16) return SmolLMAb(tensorsHalf);
                    return SmolLMAb(tensors);
                }
                if (bf16) return benchmark ? BenchmarkSmolLM(tensorsBf16, simdWiden) : RunSmolLM(tensorsBf16, simdWiden);
                if (fp16) return benchmark ? BenchmarkSmolLM(tensorsHalf, simdWiden) : RunSmolLM(tensorsHalf, simdWiden);
                return benchmark ? BenchmarkSmolLM(tensors, simdWiden) : RunSmolLM(tensors, simdWiden);
            case "qwen":
                if (mode == "distill")
                    return Qwen.RunDistill(tensors, modelDir, teacherExamples, force, seed);
                if (mode == "benchmark")
                    return Qwen.RunBenchmark(tensors, modelDir);
                return Qwen.RunTools(tensors, modelDir, useKvCache: !noKvCache, text);
            default:
                Console.Error.WriteLine($"Unknown model type: {modelType}");
                return 1;
        }
    }

    static int RunMobileNetV2Benchmark(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        Console.WriteLine("=== MobileNetV2 Benchmark ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})");
        Console.WriteLine();

        var buildSw = Stopwatch.StartNew();
        var model = MobileNetV2.LoadWeights(tensors);
        buildSw.Stop();
        Console.WriteLine($"Model build: {buildSw.ElapsedMilliseconds} ms");

        int paramCount = MobileNetV2.CountParameters(tensors);
        Console.WriteLine($"Parameters: {paramCount:N0}");
        Console.WriteLine();

        int n = 1, c = 3, h = 224, w = 224;
        Console.WriteLine("Warmup (3 passes)...");
        for (int i = 0; i < 3; i++)
        {
            var dummy = CreateRandomInput(n, c, h, w);
            model.Forward(dummy);
        }
        Console.WriteLine();

        Console.WriteLine($"Benchmark: synthetic {w}x{h} input (10 passes)...");
        var times = new List<double>();
        for (int i = 0; i < 10; i++)
        {
            var input = CreateRandomInput(n, c, h, w);
            var sw = Stopwatch.StartNew();
            var output = model.Forward(input);
            sw.Stop();
            double ms = sw.ElapsedMilliseconds + sw.ElapsedTicks / (double)Stopwatch.Frequency * 1000.0;
            times.Add(ms);
            Console.WriteLine($"  Run {i + 1,2}: {ms:F1} ms");
        }
        Console.WriteLine($"  Average: {times.Average():F1} ms  (min={times.Min():F1}, max={times.Max():F1})");

        var lastInput = CreateRandomInput(n, c, h, w);
        var lastOutput = model.Forward(lastInput);
        PrintTopK(lastOutput);
        Console.WriteLine();

        RunImageBenchmarks(model);
        return 0;
    }

    static int RunResNet18Benchmark(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        Console.WriteLine("=== ResNet-18 Benchmark ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})");
        Console.WriteLine();

        var buildSw = Stopwatch.StartNew();
        var model = ResNet18.LoadWeights(tensors);
        buildSw.Stop();
        Console.WriteLine($"Model build: {buildSw.ElapsedMilliseconds} ms");

        int paramCount = ResNet18.CountParameters(tensors);
        Console.WriteLine($"Parameters: {paramCount:N0}");
        Console.WriteLine();

        int n = 1, c = 3, h = 224, w = 224;
        Console.WriteLine("Warmup (3 passes)...");
        for (int i = 0; i < 3; i++)
        {
            var dummy = CreateRandomInput(n, c, h, w);
            model.Forward(dummy);
        }
        Console.WriteLine();

        Console.WriteLine($"Benchmark: synthetic {w}x{h} input (10 passes)...");
        var times = new List<double>();
        for (int i = 0; i < 10; i++)
        {
            var input = CreateRandomInput(n, c, h, w);
            var sw = Stopwatch.StartNew();
            var output = model.Forward(input);
            sw.Stop();
            double ms = sw.ElapsedMilliseconds + sw.ElapsedTicks / (double)Stopwatch.Frequency * 1000.0;
            times.Add(ms);
            Console.WriteLine($"  Run {i + 1,2}: {ms:F1} ms");
        }
        Console.WriteLine($"  Average: {times.Average():F1} ms  (min={times.Min():F1}, max={times.Max():F1})");

        var lastInput = CreateRandomInput(n, c, h, w);
        var lastOutput = model.Forward(lastInput);
        PrintTopK(lastOutput);
        Console.WriteLine();

        RunImageBenchmarks(model);
        return 0;
    }

    static void RunImageBenchmarks(Module<float> model)
    {
        string imageDir = Path.Combine("samples", "data", "images");
        if (!Directory.Exists(imageDir))
        {
            Console.WriteLine("No images directory found, skipping image benchmarks.");
            return;
        }

        var imageFiles = Directory.GetFiles(imageDir, "*.jpg").OrderBy(f => f).ToArray();
        if (imageFiles.Length == 0)
        {
            Console.WriteLine("No .jpg images found, skipping image benchmarks.");
            return;
        }

        Console.WriteLine($"Benchmark: real images ({imageFiles.Length} images)...");
        foreach (var path in imageFiles)
        {
            using var img = new Bitmap(path);
            var sw = Stopwatch.StartNew();
            var input = PreprocessImage(img, 224);
            var output = model.Forward(input);
            sw.Stop();
            double ms = sw.ElapsedMilliseconds + sw.ElapsedTicks / (double)Stopwatch.Frequency * 1000.0;
            Console.WriteLine($"  {Path.GetFileName(path)} ({img.Width}x{img.Height}): {ms:F1} ms");
            PrintTopK(output, k: 3);
            Console.WriteLine();
        }
    }

    static int RunMobileNetV2Inference(
        Dictionary<string, (float[] Data, int[] Shape)> tensors, string mode)
    {
        Console.WriteLine("Building MobileNetV2 model...");
        var sw = Stopwatch.StartNew();
        var model = MobileNetV2.LoadWeights(tensors);
        sw.Stop();
        Console.WriteLine($"Model built in {sw.ElapsedMilliseconds} ms");
        Console.WriteLine();

        if (string.IsNullOrEmpty(mode))
            RunInference(model);
        else
            RunImageInference(model, mode);
        return 0;
    }

    static int RunResNet18Inference(
        Dictionary<string, (float[] Data, int[] Shape)> tensors, string mode)
    {
        Console.WriteLine("Building ResNet-18 model...");
        var sw = Stopwatch.StartNew();
        var model = ResNet18.LoadWeights(tensors);
        sw.Stop();
        Console.WriteLine($"Model built in {sw.ElapsedMilliseconds} ms");
        Console.WriteLine();

        if (string.IsNullOrEmpty(mode))
            RunInference(model);
        else
            RunImageInference(model, mode);
        return 0;
    }

    static void RunInference(Module<float> model)
    {
        var input = CreateRandomInput(1, 3, 224, 224);

        Console.WriteLine($"Running forward pass with input [1,3,224,224]...");
        var sw = Stopwatch.StartNew();
        var output = model.Forward(input);
        sw.Stop();
        Console.WriteLine($"Forward pass completed in {sw.ElapsedMilliseconds} ms");
        Console.WriteLine($"Output shape: [{string.Join(", ", output.Shape)}]");

        PrintTopK(output);
    }

    static void RunImageInference(Module<float> model, string imagePath)
    {
        if (!File.Exists(imagePath))
        {
            Console.Error.WriteLine($"Image not found: {imagePath}");
            return;
        }

        Console.WriteLine($"Loading image: {imagePath}");
        using var img = new Bitmap(imagePath);
        Console.WriteLine($"  Original size: {img.Width}x{img.Height}");

        var input = PreprocessImage(img, 224);
        Console.WriteLine($"  Preprocessed to [1,3,224,224]");

        Console.WriteLine($"Running forward pass...");
        var sw = Stopwatch.StartNew();
        var output = model.Forward(input);
        sw.Stop();
        Console.WriteLine($"Forward pass completed in {sw.ElapsedMilliseconds} ms");
        Console.WriteLine($"Output shape: [{string.Join(", ", output.Shape)}]");

        PrintTopK(output);
    }

    static ReverseGradTensor<float> CreateRandomInput(int n, int c, int h, int w)
    {
        int total = n * c * h * w;
        var data = new float[total];
        var rng = new Random(42);
        for (int i = 0; i < total; i++)
            data[i] = rng.NextSingle() * 2f - 1f;

        var input = ReverseGradTensor<float>.FromMatrix(data, n, c * h * w);
        input.Reshape(n, c, h, w);
        return input;
    }

    static ReverseGradTensor<float> PreprocessImage(Bitmap img, int size)
    {
        using var resized = new Bitmap(img, new Size(size, size));

        float[] mean = [0.485f, 0.456f, 0.406f];
        float[] std = [0.229f, 0.224f, 0.225f];
        var data = new float[3 * size * size];

        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                var pixel = resized.GetPixel(x, y);
                int spatialIdx = y * size + x;
                data[0 * size * size + spatialIdx] = (pixel.R / 255.0f - mean[0]) / std[0];
                data[1 * size * size + spatialIdx] = (pixel.G / 255.0f - mean[1]) / std[1];
                data[2 * size * size + spatialIdx] = (pixel.B / 255.0f - mean[2]) / std[2];
            }
        }

        var input = ReverseGradTensor<float>.FromMatrix(data, 1, 3 * size * size);
        input.Reshape(1, 3, size, size);
        return input;
    }

    static int RunCompare(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelType)
    {
        string inputPath = Path.Combine("samples", "data", "compare_input.bin");
        if (!File.Exists(inputPath))
        {
            Console.Error.WriteLine($"Input file not found: {inputPath}");
            Console.Error.WriteLine("Run: python samples/NivaraInference/Python/generate_input.py");
            return 1;
        }

        Console.WriteLine($"Reading input from {inputPath}...");
        var rawBytes = File.ReadAllBytes(inputPath);
        float[] inputData = new float[rawBytes.Length / 4];
        Buffer.BlockCopy(rawBytes, 0, inputData, 0, rawBytes.Length);
        Console.WriteLine($"  {inputData.Length} floats, mean={TensorPrimitives.Average(inputData.AsSpan()):F6}");

        var input = ReverseGradTensor<float>.FromMatrix(inputData, 1, 3 * 224 * 224);
        input.Reshape(1, 3, 224, 224);

        Module<float> model = modelType switch
        {
            "mobilenet_v2" => MobileNetV2.LoadWeights(tensors),
            "resnet18" => ResNet18.LoadWeights(tensors),
            _ => throw new ArgumentException($"Unknown model: {modelType}")
        };

        Console.WriteLine("Running forward pass...");
        var sw = Stopwatch.StartNew();
        var output = model.Forward(input);
        sw.Stop();
        Console.WriteLine($"Forward pass: {sw.ElapsedMilliseconds} ms");
        Console.WriteLine($"Output shape: [{string.Join(", ", output.Shape)}]");

        int numClasses = output.Shape[^1];
        var logits = new float[numClasses];
        output.Data.TryGetSpan(out var outSpan);
        if (!outSpan.IsEmpty)
            outSpan.CopyTo(logits);

        Console.WriteLine($"Raw logits (first 10):");
        Console.Write("  [");
        for (int i = 0; i < Math.Min(10, numClasses); i++)
        {
            Console.Write($"{logits[i]:F6}");
            if (i < 9) Console.Write(", ");
        }
        Console.WriteLine("]");

        Console.WriteLine($"Logits stats: min={TensorPrimitives.Min(logits.AsSpan()):F6}, max={TensorPrimitives.Max(logits.AsSpan()):F6}, mean={TensorPrimitives.Average(logits.AsSpan()):F6}");

        PrintTopK(output);

        var logitsPath = Path.Combine("samples", "data", "compare_logits_cs.bin");
        using (var fs = File.Create(logitsPath))
            fs.Write(MemoryMarshal.AsBytes(logits.AsSpan()));
        Console.WriteLine($"Saved logits to {logitsPath}");

        return 0;
    }

    static int RunCompareDiag(Dictionary<string, (float[] Data, int[] Shape)> tensors, string modelType)
    {
        string inputPath = Path.Combine("samples", "data", "compare_input.bin");
        if (!File.Exists(inputPath))
        {
            Console.Error.WriteLine($"Input file not found: {inputPath}");
            return 1;
        }

        string diagDir = Path.Combine("samples", "data", "diag");
        Directory.CreateDirectory(diagDir);

        Console.WriteLine($"Reading input from {inputPath}...");
        var rawBytes = File.ReadAllBytes(inputPath);
        float[] inputData = new float[rawBytes.Length / 4];
        Buffer.BlockCopy(rawBytes, 0, inputData, 0, rawBytes.Length);
        Console.WriteLine($"  {inputData.Length} floats, mean={TensorPrimitives.Average(inputData.AsSpan()):F6}");

        var input = ReverseGradTensor<float>.FromMatrix(inputData, 1, 3 * 224 * 224);
        input.Reshape(1, 3, 224, 224);

        if (modelType == "resnet18")
            RunResNet18Diag(tensors, input, diagDir);
        else if (modelType == "mobilenet_v2")
            RunMobileNetV2Diag(tensors, input, diagDir);
        else
        {
            Console.Error.WriteLine($"Unknown model type: {modelType}");
            return 1;
        }

        Console.WriteLine($"Saved diagnostics to {diagDir}/");
        return 0;
    }

    static void SaveDiag(string diagDir, string name, ReverseGradTensor<float> tensor)
    {
        int total = tensor.Length;
        var data = new float[total];
        tensor.Data.TryGetSpan(out var span);
        if (!span.IsEmpty)
            span.Slice(0, total).CopyTo(data);
        else
            tensor.Data.CopyTo(data, 0);

        string path = Path.Combine(diagDir, $"{name}.bin");
        using var fs = File.Create(path);
        fs.Write(MemoryMarshal.AsBytes(data.AsSpan()));

        double mean = TensorPrimitives.Average(data.AsSpan());
        float min = TensorPrimitives.Min(data.AsSpan()), max = TensorPrimitives.Max(data.AsSpan());
        string shapeStr = string.Join("x", tensor.Shape);
        Console.WriteLine($"  {name}: [{shapeStr}], mean={mean:F6}, min={min:F6}, max={max:F6}");
        Console.Write($"    first9: [");
        for (int i = 0; i < Math.Min(9, total); i++)
        {
            Console.Write($"{data[i]:F6}");
            if (i < Math.Min(9, total) - 1) Console.Write(", ");
        }
        Console.WriteLine("]");
    }

    static void RunResNet18Diag(
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        ReverseGradTensor<float> input,
        string diagDir)
    {
        Console.WriteLine("=== ResNet-18 Step-by-Step Diagnostics ===");
        Console.WriteLine();

        var stemConv = new Conv2d<float>(3, 64, 7, stride: 2, padding: 3, bias: false);
        var stemBn = new BatchNorm2d<float>(64);
        var stemPool = new MaxPool2d<float>(kernelSize: 3, stride: 2, padding: 1);

        ResNet18.LoadConv(stemConv,
            tensors["resnet.embedder.embedder.convolution.weight"].Data,
            tensors["resnet.embedder.embedder.convolution.weight"].Shape);
        ResNet18.LoadBn(stemBn,
            tensors.TryGetValue("resnet.embedder.embedder.normalization.weight", out var sw0) ? sw0.Data : null,
            tensors.TryGetValue("resnet.embedder.embedder.normalization.bias", out var sb0) ? sb0.Data : null,
            tensors.TryGetValue("resnet.embedder.embedder.normalization.running_mean", out var sm0) ? sm0.Data : null,
            tensors.TryGetValue("resnet.embedder.embedder.normalization.running_var", out var sv0) ? sv0.Data : null);
        stemBn.Eval();

        Console.WriteLine("--- Step 1: Stem Conv ---");
        var x = stemConv.Forward(input);
        SaveDiag(diagDir, "cs_step1_stem_conv", x);
        Console.WriteLine();

        Console.WriteLine("--- Step 2: Stem BN (eval) ---");
        x = stemBn.Forward(x);
        SaveDiag(diagDir, "cs_step2_stem_bn", x);
        Console.WriteLine();

        Console.WriteLine("--- Step 3: Stem ReLU ---");
        x = ReverseGradOperations.Relu(x);
        SaveDiag(diagDir, "cs_step3_stem_relu", x);
        Console.WriteLine();

        Console.WriteLine("--- Step 4: Stem Pool ---");
        x = stemPool.Forward(x);
        SaveDiag(diagDir, "cs_step4_stem_pool", x);
        Console.WriteLine();

        string[] stagePrefixes = [
            "resnet.encoder.stages.0.layers.0",
            "resnet.encoder.stages.0.layers.1",
            "resnet.encoder.stages.1.layers.0",
            "resnet.encoder.stages.1.layers.1",
            "resnet.encoder.stages.2.layers.0",
            "resnet.encoder.stages.2.layers.1",
            "resnet.encoder.stages.3.layers.0",
            "resnet.encoder.stages.3.layers.1",
        ];
        int[] inChannels = [64, 64, 64, 128, 128, 256, 256, 512];
        int[] outChannels = [64, 64, 128, 128, 256, 256, 512, 512];
        int[] strides = [1, 1, 2, 1, 2, 1, 2, 1];

        for (int i = 0; i < 8; i++)
        {
            bool hasDownsample = inChannels[i] != outChannels[i] || strides[i] != 1;

            var conv1 = new Conv2d<float>(inChannels[i], outChannels[i], 3, stride: strides[i], padding: 1, bias: false);
            var bn1 = new BatchNorm2d<float>(outChannels[i]);
            var conv2 = new Conv2d<float>(outChannels[i], outChannels[i], 3, padding: 1, bias: false);
            var bn2 = new BatchNorm2d<float>(outChannels[i]);

            Conv2d<float>? dsConv = null;
            BatchNorm2d<float>? dsBn = null;
            if (hasDownsample)
            {
                dsConv = new Conv2d<float>(inChannels[i], outChannels[i], 1, stride: strides[i], bias: false);
                dsBn = new BatchNorm2d<float>(outChannels[i]);
            }

            ResNet18.LoadConv(conv1, tensors[$"{stagePrefixes[i]}.layer.0.convolution.weight"].Data,
                tensors[$"{stagePrefixes[i]}.layer.0.convolution.weight"].Shape);
            ResNet18.LoadBn(bn1,
                tensors.TryGetValue($"{stagePrefixes[i]}.layer.0.normalization.weight", out var w1) ? w1.Data : null,
                tensors.TryGetValue($"{stagePrefixes[i]}.layer.0.normalization.bias", out var b1) ? b1.Data : null,
                tensors.TryGetValue($"{stagePrefixes[i]}.layer.0.normalization.running_mean", out var m1) ? m1.Data : null,
                tensors.TryGetValue($"{stagePrefixes[i]}.layer.0.normalization.running_var", out var v1) ? v1.Data : null);

            ResNet18.LoadConv(conv2, tensors[$"{stagePrefixes[i]}.layer.1.convolution.weight"].Data,
                tensors[$"{stagePrefixes[i]}.layer.1.convolution.weight"].Shape);
            ResNet18.LoadBn(bn2,
                tensors.TryGetValue($"{stagePrefixes[i]}.layer.1.normalization.weight", out var w2) ? w2.Data : null,
                tensors.TryGetValue($"{stagePrefixes[i]}.layer.1.normalization.bias", out var b2) ? b2.Data : null,
                tensors.TryGetValue($"{stagePrefixes[i]}.layer.1.normalization.running_mean", out var m2) ? m2.Data : null,
                tensors.TryGetValue($"{stagePrefixes[i]}.layer.1.normalization.running_var", out var v2) ? v2.Data : null);

            if (hasDownsample && dsConv != null && dsBn != null)
            {
                ResNet18.LoadConv(dsConv, tensors[$"{stagePrefixes[i]}.shortcut.convolution.weight"].Data,
                    tensors[$"{stagePrefixes[i]}.shortcut.convolution.weight"].Shape);
                ResNet18.LoadBn(dsBn,
                    tensors.TryGetValue($"{stagePrefixes[i]}.shortcut.normalization.weight", out var sw) ? sw.Data : null,
                    tensors.TryGetValue($"{stagePrefixes[i]}.shortcut.normalization.bias", out var sb) ? sb.Data : null,
                    tensors.TryGetValue($"{stagePrefixes[i]}.shortcut.normalization.running_mean", out var sm) ? sm.Data : null,
                    tensors.TryGetValue($"{stagePrefixes[i]}.shortcut.normalization.running_var", out var sv) ? sv.Data : null);
            }

            conv1.Eval(); bn1.Eval(); conv2.Eval(); bn2.Eval();
            dsConv?.Eval(); dsBn?.Eval();

            Console.WriteLine($"--- Stage {i / 2}, Block {i % 2} (conv1) ---");
            var cx = conv1.Forward(x);
            SaveDiag(diagDir, $"cs_stage{i}_conv1", cx);

            Console.WriteLine($"--- Stage {i / 2}, Block {i % 2} (bn1) ---");
            cx = bn1.Forward(cx);
            SaveDiag(diagDir, $"cs_stage{i}_bn1", cx);

            Console.WriteLine($"--- Stage {i / 2}, Block {i % 2} (relu1) ---");
            cx = ReverseGradOperations.Relu(cx);
            SaveDiag(diagDir, $"cs_stage{i}_relu1", cx);

            Console.WriteLine($"--- Stage {i / 2}, Block {i % 2} (conv2) ---");
            cx = conv2.Forward(cx);
            SaveDiag(diagDir, $"cs_stage{i}_conv2", cx);

            Console.WriteLine($"--- Stage {i / 2}, Block {i % 2} (bn2) ---");
            cx = bn2.Forward(cx);
            SaveDiag(diagDir, $"cs_stage{i}_bn2", cx);

            var residual = hasDownsample && dsConv != null && dsBn != null
                ? dsBn.Forward(dsConv.Forward(x))
                : x;

            Console.WriteLine($"--- Stage {i / 2}, Block {i % 2} (residual) ---");
            SaveDiag(diagDir, $"cs_stage{i}_residual", residual);

            cx = cx + residual;
            cx = ReverseGradOperations.Relu(cx);
            x = cx;

            Console.WriteLine($"--- After stage {i} ---");
            SaveDiag(diagDir, $"cs_step{5 + i}_stage{i / 2}{'a' + i % 2}", x);
            Console.WriteLine();
        }

        var avgPool = new AdaptiveAvgPool2d<float>(1);
        avgPool.Eval();
        x = avgPool.Forward(x);
        SaveDiag(diagDir, "cs_step9_avgpool", x);

        int n = x.Shape[0], c = x.Shape[1];
        x.Reshape(n, c);
        SaveDiag(diagDir, "cs_step9b_flattened", x);

        var fc = new Linear<float>(512, 1000, bias: true);
        ResNet18.LoadLinear(fc,
            tensors["classifier.1.weight"].Data,
            tensors["classifier.1.weight"].Shape,
            tensors.TryGetValue("classifier.1.bias", out var bias) ? bias.Data : null);
        fc.Eval();
        x = fc.Forward(x);
        SaveDiag(diagDir, "cs_step10_logits", x);
    }

    static void RunMobileNetV2Diag(
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        ReverseGradTensor<float> input,
        string diagDir)
    {
        Console.WriteLine("=== MobileNetV2 Step-by-Step Diagnostics ===");
        Console.WriteLine();

        var model = MobileNetV2.LoadWeights(tensors);
        var x = model.Forward(input);
        SaveDiag(diagDir, "cs_final_logits", x);
    }

    static void PrintTopK(Nivara.AutoDiff.ReverseGradTensor<float> output, int k = 5)
    {
        int numClasses = output.Shape[^1];
        k = Math.Min(k, numClasses);
        var scores = new float[numClasses];
        output.Data.TryGetSpan(out var outSpan);
        if (!outSpan.IsEmpty)
            outSpan.CopyTo(scores);

        var topIndices = Enumerable.Range(0, numClasses)
            .OrderByDescending(i => scores[i])
            .Take(k)
            .ToArray();

        Console.WriteLine($"Top-{k} predictions:");
        for (int i = 0; i < k; i++)
        {
            int idx = topIndices[i];
            Console.WriteLine($"  #{i + 1}: class {idx,5}  score={scores[idx]:F6}");
        }
    }

    static int RunMiniLMInference(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        Console.WriteLine("=== MiniLM Inference ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})");
        Console.WriteLine();

        var config = BertConfig.FromJson(File.ReadAllText(Path.Combine("samples", "data", "minilm", "config.json")));

        var buildSw = Stopwatch.StartNew();
        var model = MiniLMDistilled<float>.LoadWeights(tensors, config);
        buildSw.Stop();
        Console.WriteLine($"Model build: {buildSw.ElapsedMilliseconds} ms");

        int totalParams = tensors.Values.Sum(t => t.Data.Length);
        Console.WriteLine($"Parameters: {totalParams:N0}");
        Console.WriteLine();

        var tokenizer = MiniLMTokenizer.Load(Path.Combine("samples", "data", "minilm", "vocab.txt"));
        string text = "This is a test sentence.";
        var (input, mask) = MiniLMTokenizer.TokenizeWithMask(tokenizer, text, maxLen: 128);

        Console.WriteLine($"Input text: \"{text}\"");
        var inputData = new float[input.Length];
        input.Data.TryGetSpan(out var inSpan);
        if (!inSpan.IsEmpty) inSpan.CopyTo(inputData);
        Console.WriteLine($"Input tokens (first 10): [{string.Join(", ", inputData.Take(10).Select(x => (int)x))}] (seqLen={input.Length})");

        model.Eval();
        var fwdSw = Stopwatch.StartNew();
        var output = mask != null ? model.ForwardWithMask(input, mask) : model.Forward(input);
        fwdSw.Stop();

        var outputData = new float[output.Length];
        output.Data.TryGetSpan(out var outSpan);
        if (!outSpan.IsEmpty) outSpan.CopyTo(outputData);

        Console.WriteLine($"Forward: {fwdSw.ElapsedMilliseconds} ms");
        Console.WriteLine($"Output shape: [{output.Shape[^1]}]");
        Console.WriteLine($"Output stats: min={TensorPrimitives.Min(outputData.AsSpan()):F6}, max={TensorPrimitives.Max(outputData.AsSpan()):F6}, mean={TensorPrimitives.Average(outputData.AsSpan()):F6}");
        Console.Write("Output[:10]: [");
        for (int i = 0; i < Math.Min(10, outputData.Length); i++)
        {
            Console.Write($"{outputData[i]:F6}");
            if (i < Math.Min(10, outputData.Length) - 1) Console.Write(", ");
        }
        Console.WriteLine("]");

        float norm = TensorPrimitives.Norm(outputData.AsSpan());
        Console.WriteLine($"L2 norm: {norm:F6} (should be ~1.0 for normalized embeddings)");
        Console.WriteLine();

        return 0;
    }

    static int RunMiniLMCompare(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        Console.WriteLine("=== MiniLM Compare ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})");
        Console.WriteLine();

        var config = BertConfig.FromJson(File.ReadAllText(Path.Combine("samples", "data", "minilm", "config.json")));

        var buildSw = Stopwatch.StartNew();
        var model = MiniLMDistilled<float>.LoadWeights(tensors, config);
        buildSw.Stop();
        Console.WriteLine($"Model build: {buildSw.ElapsedMilliseconds} ms");

        int totalParams = tensors.Values.Sum(t => t.Data.Length);
        Console.WriteLine($"Parameters: {totalParams:N0}");
        Console.WriteLine();

        var tokenizer = MiniLMTokenizer.Load(Path.Combine("samples", "data", "minilm", "vocab.txt"));

        var sentences = new[]
        {
            "This is a cat.",
            "This is a dog.",
            "I love programming.",
            "The weather is nice today.",
            "I love coding."
        };

        Console.WriteLine($"Sentences ({sentences.Length}):");
        for (int i = 0; i < sentences.Length; i++)
            Console.WriteLine($"  [{i}] {sentences[i]}");
        Console.WriteLine();

        model.Eval();

        var embeddings = new float[sentences.Length][];
        var fwdSw = Stopwatch.StartNew();
        for (int s = 0; s < sentences.Length; s++)
        {
            var (input, mask) = MiniLMTokenizer.TokenizeWithMask(tokenizer, sentences[s], maxLen: 128);
            var output = mask != null ? model.ForwardWithMask(input, mask) : model.Forward(input);
            var outputData = new float[output.Length];
            output.Data.TryGetSpan(out var outSpan);
            if (!outSpan.IsEmpty) outSpan.CopyTo(outputData);
            embeddings[s] = outputData;
        }
        fwdSw.Stop();

        double avgMs = fwdSw.ElapsedMilliseconds / (double)sentences.Length;
        Console.WriteLine($"Forward total: {fwdSw.ElapsedMilliseconds} ms across {sentences.Length} sentences ({avgMs:F1} ms/sentence)");
        Console.WriteLine();

        for (int i = 0; i < sentences.Length; i++)
        {
            var emb = embeddings[i];
            float norm = TensorPrimitives.Norm(emb.AsSpan());

            Console.WriteLine($"[{i}] {sentences[i]}");
            Console.Write($"    first 10: [");
            for (int j = 0; j < Math.Min(10, emb.Length); j++)
            {
                Console.Write($"{emb[j]:F6}");
                if (j < Math.Min(10, emb.Length) - 1) Console.Write(", ");
            }
            Console.WriteLine("]");
            Console.WriteLine($"    stats: min={TensorPrimitives.Min(emb.AsSpan()):F6}, max={TensorPrimitives.Max(emb.AsSpan()):F6}, mean={TensorPrimitives.Average(emb.AsSpan()):F6}, L2 norm={norm:F6}");
            Console.WriteLine();
        }

        Console.WriteLine("Cosine Similarity Matrix:");
        Console.Write("       ");
        for (int i = 0; i < sentences.Length; i++)
            Console.Write($"  [{i}]   ");
        Console.WriteLine();
        for (int i = 0; i < sentences.Length; i++)
        {
            Console.Write($"  [{i}]  ");
            for (int j = 0; j < sentences.Length; j++)
            {
                float sim = TensorPrimitives.CosineSimilarity(embeddings[i].AsSpan(), embeddings[j].AsSpan());
                Console.Write($"{sim,7:F4} ");
            }
            Console.WriteLine();
        }
        Console.WriteLine();

        var savePath = Path.Combine("samples", "data", "compare_minilm_embeddings_cs.bin");
        using (var fs = File.Create(savePath))
        {
            int totalFloats = embeddings.Length * embeddings[0].Length;
            byte[] raw = new byte[totalFloats * 4];
            int offset = 0;
            for (int i = 0; i < embeddings.Length; i++)
            {
                Buffer.BlockCopy(embeddings[i], 0, raw, offset, embeddings[i].Length * 4);
                offset += embeddings[i].Length * 4;
            }
            fs.Write(raw);
        }
        Console.WriteLine($"Saved embeddings to {savePath}");

        return 0;
    }

    static void ReportTiming<T>(Func<ReverseGradTensor<T>> forward, int warmup = 3, int passes = 10)
        where T : struct, IFloatingPointIeee754<T>
    {
        Console.WriteLine($"Warmup ({warmup} passes)...");
        for (int i = 0; i < warmup; i++)
            forward();

        Console.WriteLine($"Benchmarking ({passes} passes)...");
        var times = new List<long>();
        for (int i = 0; i < passes; i++)
        {
            var sw = Stopwatch.StartNew();
            forward();
            sw.Stop();
            times.Add(sw.ElapsedMilliseconds);
        }

        Console.WriteLine($"  Average: {times.Average():F1} ms");
        Console.WriteLine($"  Min:     {times.Min()} ms");
        Console.WriteLine($"  Max:     {times.Max()} ms");
        Console.WriteLine();
    }

    static int BenchmarkMiniLM<T>(Dictionary<string, (T[] Data, int[] Shape)> tensors, string precision)
        where T : struct, IFloatingPointIeee754<T>
    {
        Console.WriteLine("=== MiniLM Benchmark ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})  Precision: {precision}");
        Console.WriteLine();

        var config = BertConfig.FromJson(File.ReadAllText(Path.Combine("samples", "data", "minilm", "config.json")));
        var model = MiniLMDistilled<T>.LoadWeights<T, T>(tensors, config);

        int totalParams = tensors.Values.Sum(t => t.Data.Length);
        Console.WriteLine($"Parameters: {totalParams:N0}");
        Console.WriteLine($"Weights: {totalParams * Unsafe.SizeOf<T>() / (1024.0 * 1024.0):F1} MB");
        Console.WriteLine();

        var tokenizer = MiniLMTokenizer.Load(Path.Combine("samples", "data", "minilm", "vocab.txt"));
        string text = "This is a long test sentence that will be tokenized to demonstrate the performance of the MiniLM model inference across multiple tokens for benchmarking purposes.";
        var (tokenIds, attnMask, _) = MiniLMTokenizer.Encode(tokenizer, text, maxLen: 128);
        var intIds = Array.ConvertAll(tokenIds, x => (int)x);
        var mask = GradientUtils.Constant(Array.ConvertAll(attnMask, x => T.CreateChecked(x)));
        model.Eval();

        Console.WriteLine($"Input text length: {text.Split(' ').Length} words");
        Console.WriteLine($"Input tokens: {intIds.Length}");
        Console.WriteLine();

        ReportTiming<T>(() => model.ForwardWithMask(intIds, mask));
        return 0;
    }

    static int RunDistilBertInference(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        Console.WriteLine("=== DistilBERT Inference ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})");
        Console.WriteLine();

        var config = DistilBertConfig.FromJson(File.ReadAllText(Path.Combine("samples", "data", "distilbert", "config.json")));
        Console.WriteLine($"Config: dim={config.Dim}, layers={config.NLayers}, heads={config.NHeads}, hidden={config.HiddenDim}");

        var buildSw = Stopwatch.StartNew();
        var encoder = DistilBertLoader.LoadEncoder(tensors, config.ToBertConfig());
        buildSw.Stop();
        Console.WriteLine($"Model build: {buildSw.ElapsedMilliseconds} ms");

        int totalParams = tensors.Values.Sum(t => t.Data.Length);
        double weightMb = tensors.Values.Sum(t => t.Data.Length * 4.0) / (1024.0 * 1024.0);
        Console.WriteLine($"Parameters: {totalParams:N0}");
        Console.WriteLine($"Weights: {weightMb:F1} MB");
        Console.WriteLine();

        var tokenizer = MiniLMTokenizer.Load(Path.Combine("samples", "data", "distilbert", "vocab.txt"));
        string text = "This is a test sentence.";
        var (input, mask) = MiniLMTokenizer.TokenizeWithMask(tokenizer, text, maxLen: 128);

        Console.WriteLine($"Input text: \"{text}\"");
        var inputData = new float[input.Length];
        input.Data.TryGetSpan(out var inSpan);
        if (!inSpan.IsEmpty) inSpan.CopyTo(inputData);
        Console.WriteLine($"Input tokens (first 10): [{string.Join(", ", inputData.Take(10).Select(x => (int)x))}] (seqLen={input.Length})");

        encoder.Eval();
        var fwdSw = Stopwatch.StartNew();
        var output = mask != null ? encoder.ForwardWithMask(input, mask) : encoder.Forward(input);
        fwdSw.Stop();

        var outputData = new float[output.Length];
        output.Data.TryGetSpan(out var outSpan);
        if (!outSpan.IsEmpty) outSpan.CopyTo(outputData);

        Console.WriteLine($"Forward: {fwdSw.ElapsedMilliseconds} ms");
        Console.WriteLine($"Output shape: [{string.Join(", ", output.Shape)}]");
        Console.WriteLine($"Output stats: min={TensorPrimitives.Min(outputData.AsSpan()):F6}, max={TensorPrimitives.Max(outputData.AsSpan()):F6}, mean={TensorPrimitives.Average(outputData.AsSpan()):F6}, std={StdDev(outputData):F6}");
        Console.Write("Output[:10]: [");
        for (int i = 0; i < Math.Min(10, outputData.Length); i++)
        {
            Console.Write($"{outputData[i]:F6}");
            if (i < Math.Min(10, outputData.Length) - 1) Console.Write(", ");
        }
        Console.WriteLine("]");

        return 0;
    }

    static int BenchmarkDistilBert<T>(Dictionary<string, (T[] Data, int[] Shape)> tensors, string precision)
        where T : struct, IFloatingPointIeee754<T>
    {
        Console.WriteLine("=== DistilBERT Benchmark ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})  Precision: {precision}");
        Console.WriteLine();

        var config = DistilBertConfig.FromJson(File.ReadAllText(Path.Combine("samples", "data", "distilbert", "config.json")));
        var buildSw = Stopwatch.StartNew();
        var encoder = new BertEncoder<T>(config.ToBertConfig(), includeTokenTypeEmbedding: false);
        DistilBertLoader.LoadEncoderWeights<T, T>(encoder, tensors, "distilbert");
        encoder.Eval();
        buildSw.Stop();
        Console.WriteLine($"Model build: {buildSw.ElapsedMilliseconds} ms");

        int totalParams = tensors.Values.Sum(t => t.Data.Length);
        Console.WriteLine($"Parameters: {totalParams:N0}");
        Console.WriteLine($"Weights: {totalParams * Unsafe.SizeOf<T>() / (1024.0 * 1024.0):F1} MB");
        Console.WriteLine();

        var tokenizer = MiniLMTokenizer.Load(Path.Combine("samples", "data", "distilbert", "vocab.txt"));
        string text = "This is a long test sentence that will be tokenized to demonstrate the performance of the DistilBERT model inference across multiple tokens for benchmarking purposes.";
        var (tokenIds, attnMask, _) = MiniLMTokenizer.Encode(tokenizer, text, maxLen: 128);
        var intIds = Array.ConvertAll(tokenIds, x => (int)x);
        var mask = GradientUtils.Constant(Array.ConvertAll(attnMask, x => T.CreateChecked(x)));

        Console.WriteLine($"Input text length: {text.Split(' ').Length} words");
        Console.WriteLine($"Input tokens: {intIds.Length}");
        Console.WriteLine();

        ReportTiming<T>(() => encoder.ForwardWithMask(intIds, mask));
        return 0;
    }

    static double StdDev(float[] data)
    {
        if (data.Length == 0) return 0;
        double mean = TensorPrimitives.Average(data.AsSpan());
        var diff = new float[data.Length];
        TensorPrimitives.Add(data.AsSpan(), (float)-mean, diff);
        double sumSq = TensorPrimitives.Dot(diff, diff);
        return Math.Sqrt(sumSq / data.Length);
    }

    static int RunDistilBertCompare(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        string refPath = Path.Combine("samples", "data", "distilbert", "last_hidden_state_py.bin");
        if (!File.Exists(refPath))
        {
            Console.Error.WriteLine($"Reference file not found: {refPath}");
            Console.Error.WriteLine("Run: python samples/NivaraInference/Python/distilbert_compare.py");
            return 1;
        }

        Console.WriteLine("=== DistilBERT Compare ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})");
        Console.WriteLine();

        var config = DistilBertConfig.FromJson(File.ReadAllText(Path.Combine("samples", "data", "distilbert", "config.json")));
        var encoder = DistilBertLoader.LoadEncoder(tensors, config.ToBertConfig());

        var tokenizer = MiniLMTokenizer.Load(Path.Combine("samples", "data", "distilbert", "vocab.txt"));
        string text = "This is a test sentence.";
        var (input, mask) = MiniLMTokenizer.TokenizeWithMask(tokenizer, text, maxLen: 128);

        encoder.Eval();
        var output = mask != null ? encoder.ForwardWithMask(input, mask) : encoder.Forward(input);

        var outputData = new float[output.Length];
        output.Data.TryGetSpan(out var outSpan);
        if (!outSpan.IsEmpty) outSpan.CopyTo(outputData);

        var rawBytes = File.ReadAllBytes(refPath);
        float[] refData = new float[rawBytes.Length / 4];
        Buffer.BlockCopy(rawBytes, 0, refData, 0, rawBytes.Length);

        int len = Math.Min(outputData.Length, refData.Length);
        var outputSpan = outputData.AsSpan(0, len);
        var refSpan = refData.AsSpan(0, len);

        var diffArr = new float[len];
        TensorPrimitives.Subtract(outputSpan, refSpan, diffArr);
        var absDiff = new float[len];
        TensorPrimitives.Abs(diffArr.AsSpan(), absDiff);
        float maxAbs = TensorPrimitives.Max(absDiff);
        float sumAbs = TensorPrimitives.Sum(absDiff);
        float cosineSim = TensorPrimitives.CosineSimilarity(outputSpan, refSpan);

        Console.WriteLine($"Input text: \"{text}\"");
        Console.WriteLine($"Output shape: [{string.Join(", ", output.Shape)}]");
        Console.WriteLine($"  C# stats: min={TensorPrimitives.Min(outputData.AsSpan()):F6}, max={TensorPrimitives.Max(outputData.AsSpan()):F6}, mean={TensorPrimitives.Average(outputData.AsSpan()):F6}, std={StdDev(outputData):F6}");
        Console.WriteLine($"  Py stats: min={TensorPrimitives.Min(refData.AsSpan()):F6}, max={TensorPrimitives.Max(refData.AsSpan()):F6}, mean={TensorPrimitives.Average(refData.AsSpan()):F6}, std={StdDev(refData):F6}");
        Console.WriteLine($"  max abs diff: {maxAbs:F6}");
        Console.WriteLine($"  mean abs diff: {sumAbs / len:F8}");
        Console.WriteLine($"  cosine similarity: {cosineSim:F8}");
        Console.WriteLine();

        Console.Write("C# Output[:10]: [");
        for (int i = 0; i < Math.Min(10, len); i++)
        {
            Console.Write($"{outputData[i]:F6}");
            if (i < Math.Min(10, len) - 1) Console.Write(", ");
        }
        Console.WriteLine("]");
        Console.Write("Py Output[:10]: [");
        for (int i = 0; i < Math.Min(10, len); i++)
        {
            Console.Write($"{refData[i]:F6}");
            if (i < Math.Min(10, len) - 1) Console.Write(", ");
        }
        Console.WriteLine("]");

        return 0;
    }

    static int RunDistilBertSstInference(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        Console.WriteLine("=== DistilBERT SST-2 Inference ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})");
        Console.WriteLine();

        string modelDir = Path.Combine("samples", "data", "distilbert_sst");
        var config = DistilBertConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json")));
        Console.WriteLine($"Config: dim={config.Dim}, layers={config.NLayers}, heads={config.NHeads}, hidden={config.HiddenDim}");

        var buildSw = Stopwatch.StartNew();
        var model = DistilBertSst.Load(tensors, modelDir);
        var tokenizer = DistilBertSst.LoadTokenizer(modelDir);
        buildSw.Stop();
        Console.WriteLine($"Model build: {buildSw.ElapsedMilliseconds} ms");

        int totalParams = DistilBertSst.CountParameters(tensors);
        double weightMb = DistilBertSst.WeightMb(tensors);
        Console.WriteLine($"Parameters: {totalParams:N0}");
        Console.WriteLine($"Weights: {weightMb:F1} MB");
        Console.WriteLine();

        string text = "This is a test sentence.";
        Console.WriteLine($"Input text: \"{text}\"");

        model.Eval();
        var fwdSw = Stopwatch.StartNew();
        var logits = DistilBertSst.PredictLogits(model, tokenizer, text, maxLen: 128);
        fwdSw.Stop();
        var (argMax, probs) = DistilBertSst.Softmax(logits);

        Console.WriteLine($"Forward: {fwdSw.ElapsedMilliseconds} ms");
        Console.WriteLine($"Logits: [{logits.Data[0]:F6}, {logits.Data[1]:F6}]");
        Console.WriteLine($"Softmax: [NEGATIVE {probs[0] * 100:F1}%, POSITIVE {probs[1] * 100:F1}%]");
        Console.WriteLine($"Sentiment: {DistilBertSst.Label(argMax)} ({probs[argMax] * 100:F1}%)");
        Console.WriteLine();

        return 0;
    }

    static int RunDistilBertSstPredict(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        Console.WriteLine("=== DistilBERT SST-2 Interactive Sentiment ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})");
        Console.WriteLine();

        string modelDir = Path.Combine("samples", "data", "distilbert_sst");
        var buildSw = Stopwatch.StartNew();
        var model = DistilBertSst.Load(tensors, modelDir);
        var tokenizer = DistilBertSst.LoadTokenizer(modelDir);
        buildSw.Stop();
        Console.WriteLine($"Model build: {buildSw.ElapsedMilliseconds} ms");
        Console.WriteLine();
        Console.WriteLine("Type a movie review and press Enter (or 'quit' to exit).\n");

        string? line;
        while ((line = Console.ReadLine()) != null)
        {
            line = line.Trim();
            if (string.IsNullOrEmpty(line) || line.Equals("quit", StringComparison.OrdinalIgnoreCase))
                break;

            var sw = Stopwatch.StartNew();
            var logits = DistilBertSst.PredictLogits(model, tokenizer, line, maxLen: 128);
            var (argMax, probs) = DistilBertSst.Softmax(logits);
            sw.Stop();

            Console.WriteLine($"  Sentiment: {DistilBertSst.Label(argMax)} ({probs[argMax] * 100:F1}%)  [{sw.ElapsedMilliseconds} ms]");
        }

        return 0;
    }

    static int RunDistilBertSstCompare(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        Console.WriteLine("=== DistilBERT SST-2 Compare ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})");
        Console.WriteLine();

        string modelDir = Path.Combine("samples", "data", "distilbert_sst");
        string csPath = DistilBertSst.LabelsPath;
        string pyPath = Path.Combine("samples", "data", "compare_distilbert_sst_py.bin");

        Console.WriteLine($"Sentences ({DistilBertSst.CompareSentences.Length}):");
        for (int i = 0; i < DistilBertSst.CompareSentences.Length; i++)
            Console.WriteLine($"  [{i}] {DistilBertSst.CompareSentences[i]}");
        Console.WriteLine();

        DistilBertSst.SaveCompareOutput(tensors, modelDir, csPath);
        DistilBertSst.PrintCompareDiff(pyPath, csPath, DistilBertSst.CompareSentences.Length);

        return 0;
    }

    static int RunSmolLM(Dictionary<string, (float[] Data, int[] Shape)> tensors, bool simdWiden)
        => RunSmolLMCore(tensors, simdWiden);

    static int RunSmolLM(Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors, bool simdWiden)
        => RunSmolLMCore(tensors, simdWiden);

    static int RunSmolLM(Dictionary<string, (Half[] Data, int[] Shape)> tensors, bool simdWiden)
        => RunSmolLMCore(tensors, simdWiden);

    static string PrecisionName(Type t)
        => t == typeof(BFloat16) ? "BFloat16"
         : t == typeof(Half) ? "Half"
         : "F32";

    static int RunSmolLMCore<T>(Dictionary<string, (T[] Data, int[] Shape)> tensors, bool simdWiden)
        where T : struct, IFloatingPointIeee754<T>
    {
        string precisionName = PrecisionName(typeof(T));

        // The narrow 16-bit paths (BFloat16/Half) run through the Phase-1 SIMD
        // widen-compute-narrow kernels by default; without them BF16 matmul falls back
        // to a ~100x-slower scalar dot. --simd-widen opts additional precisions in, and
        // setting it off explicitly (scalar A/B) is done by the 'ab' mode. Save and
        // restore the prior global value so other model modes are unaffected.
        bool narrow = typeof(T) == typeof(BFloat16) || typeof(T) == typeof(Half);
        bool priorWiden = NivaraPrimitives.UseWidenSimd;
        if (simdWiden || narrow)
            NivaraPrimitives.UseWidenSimd = true;
        try
        {
            RunSmolLMGenerate(tensors, precisionName);
        }
        finally
        {
            NivaraPrimitives.UseWidenSimd = priorWiden;
        }
        return 0;
    }

    static int BenchmarkSmolLM<T>(Dictionary<string, (T[] Data, int[] Shape)> tensors, bool simdWiden)
        where T : struct, IFloatingPointIeee754<T>
    {
        bool narrow = typeof(T) == typeof(BFloat16) || typeof(T) == typeof(Half);
        bool priorWiden = NivaraPrimitives.UseWidenSimd;
        if (simdWiden || narrow)
            NivaraPrimitives.UseWidenSimd = true;
        try
        {
            return BenchmarkSmolLMCore(tensors);
        }
        finally
        {
            NivaraPrimitives.UseWidenSimd = priorWiden;
        }
    }

    static int BenchmarkSmolLMCore<T>(Dictionary<string, (T[] Data, int[] Shape)> tensors)
        where T : struct, IFloatingPointIeee754<T>
    {
        string precisionName = PrecisionName(typeof(T));
        Console.WriteLine($"=== SmolLM-135M Benchmark ({precisionName}) ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})  Precision: {precisionName}" +
                          $"  SIMD-widen: {(NivaraPrimitives.UseWidenSimd ? "on" : "off")}");
        Console.WriteLine();

        var modelDir = Path.Combine("samples", "data", "smollm-135m");
        var config = LlamaConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json")));

        var buildSw = Stopwatch.StartNew();
        var model = LlamaLoader.Load<T, T>(config, tensors);
        buildSw.Stop();
        Console.WriteLine($"Model build: {buildSw.ElapsedMilliseconds} ms");

        int totalParams = tensors.Values.Sum(t => t.Data.Length);
        int bytesPerWeight = typeof(T) == typeof(BFloat16) || typeof(T) == typeof(Half) ? 2 : 4;
        double weightMb = tensors.Values.Sum(t => t.Data.Length * (long)bytesPerWeight) / (1024.0 * 1024.0);
        Console.WriteLine($"Parameters: {totalParams:N0}");
        Console.WriteLine($"Weights ({precisionName}): {weightMb:F1} MB");
        Console.WriteLine();

        var tokenizer = new Gpt2BpeTokenizer(
            Path.Combine(modelDir, "vocab.json"), Path.Combine(modelDir, "merges.txt"));
        const string Prompt = "The capital of France is";
        var promptIds = tokenizer.Encode(Prompt);
        const int MaxNewTokens = 32;

        // Warmup pass (discarded — JIT + cache warming).
        var warmup = RunSmolLMGeneration(model, promptIds, config, MaxNewTokens);
        Console.WriteLine($"Warmup: {warmup.Milliseconds} ms ({warmup.Generated} tokens)");

        // Median-of-3 timed runs (matches samples/NivaraInference/README.md methodology).
        var runs = new (long Ms, int Generated)[3];
        for (int i = 0; i < 3; i++)
        {
            var timing = RunSmolLMGeneration(model, promptIds, config, MaxNewTokens);
            runs[i] = (timing.Milliseconds, timing.Generated);
            Console.WriteLine($"Run {i + 1}: {timing.Milliseconds} ms ({timing.Generated} tokens, " +
                              $"{timing.Milliseconds / Math.Max(1, timing.Generated)} ms/token)");
        }
        Array.Sort(runs, (a, b) => a.Ms.CompareTo(b.Ms));
        long medianMs = runs[1].Ms;
        int medianTokens = runs[1].Generated;
        Console.WriteLine();
        Console.WriteLine($"Median: {medianMs} ms ({medianTokens} tokens, " +
                          $"{medianMs / Math.Max(1, medianTokens):F1} ms/token)");

        return 0;
    }

    /// <summary>
    /// SmolLM A/B comparison: runs one full generation with <c>UseWidenSimd = false</c>
    /// (scalar) then one with <c>UseWidenSimd = true</c> (widen), and prints a
    /// side-by-side table of timing and generated-token counts. Also diffs the
    /// generated-token streams to confirm numerical equivalence. For BF16 this
    /// measures the actual widen-compute-narrow cost; for F32 it is effectively a
    /// no-op control (the toggle is transparent to float).
    /// </summary>
    static int SmolLMAb<T>(Dictionary<string, (T[] Data, int[] Shape)> tensors)
        where T : struct, IFloatingPointIeee754<T>
    {
        string precisionName = PrecisionName(typeof(T));
        bool narrow = typeof(T) == typeof(BFloat16) || typeof(T) == typeof(Half);
        Console.WriteLine($"=== SmolLM-135M A/B: Scalar vs SIMD Widen ({precisionName}) ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})  Precision: {precisionName}");
        if (!narrow)
            Console.WriteLine("Note: F32 widening is transparent (no-op). Timing differences are noise.");
        Console.WriteLine();

        var modelDir = Path.Combine("samples", "data", "smollm-135m");
        var config = LlamaConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json")));

        var buildSw = Stopwatch.StartNew();
        var model = LlamaLoader.Load<T, T>(config, tensors);
        buildSw.Stop();
        Console.WriteLine($"Model build: {buildSw.ElapsedMilliseconds} ms");

        int totalParams = tensors.Values.Sum(t => t.Data.Length);
        int bytesPerWeight = typeof(T) == typeof(BFloat16) || typeof(T) == typeof(Half) ? 2 : 4;
        double weightMb = tensors.Values.Sum(t => t.Data.Length * (long)bytesPerWeight) / (1024.0 * 1024.0);
        Console.WriteLine($"Parameters: {totalParams:N0}");
        Console.WriteLine($"Weights ({precisionName}): {weightMb:F1} MB");
        Console.WriteLine();

        var tokenizer = new Gpt2BpeTokenizer(
            Path.Combine(modelDir, "vocab.json"), Path.Combine(modelDir, "merges.txt"));
        const string Prompt = "The capital of France is";
        var promptIds = tokenizer.Encode(Prompt);
        const int MaxNewTokens = 32;

        // Half (fp16) is unsupported for SmolLM generation: the 16-bit weights produce
        // NaN logits (see README numerical caveats), so an A/B comparison would be
        // meaningless. Fall back to a single timed generation reporting timing only.
        if (typeof(T) == typeof(Half))
        {
            Console.WriteLine("Note: Half (fp16) is unsupported for SmolLM generation (NaN");
            Console.WriteLine("logits — see README numerical caveats). The A/B comparison is");
            Console.WriteLine("only meaningful for BF16; running a single timed run instead.");
            Console.WriteLine();
            NivaraPrimitives.UseWidenSimd = true; // narrow auto-enable
            var fallback = RunSmolLMGeneration(model, promptIds, config, MaxNewTokens);
            Console.WriteLine($"Generated {fallback.Generated} tokens in {fallback.Milliseconds} ms " +
                              $"({fallback.Milliseconds / Math.Max(1, fallback.Generated)} ms/token)");
            return 0;
        }

        // Warmup pass (discarded — JIT + cache warming).
        NivaraPrimitives.UseWidenSimd = false;
        RunSmolLMGeneration(model, promptIds, config, MaxNewTokens);

        // Side A: scalar (UseWidenSimd = false).
        NivaraPrimitives.UseWidenSimd = false;
        var scalar = RunSmolLMGeneration(model, promptIds, config, MaxNewTokens);
        Console.WriteLine($"Scalar:  {scalar.Milliseconds} ms  ({scalar.Generated} tokens, " +
                          $"{scalar.Milliseconds / Math.Max(1, scalar.Generated)} ms/token)");

        // Side B: widen (UseWidenSimd = true).
        NivaraPrimitives.UseWidenSimd = true;
        var widened = RunSmolLMGeneration(model, promptIds, config, MaxNewTokens);
        Console.WriteLine($"Widen:   {widened.Milliseconds} ms  ({widened.Generated} tokens, " +
                          $"{widened.Milliseconds / Math.Max(1, widened.Generated)} ms/token)");
        NivaraPrimitives.UseWidenSimd = false; // restore safe default

        // Summary.
        Console.WriteLine();
        double ratio = scalar.Milliseconds > 0
            ? (double)scalar.Milliseconds / widened.Milliseconds
            : 0;
        Console.WriteLine($"Scalar/Widen ratio: {ratio:F2}x  " +
                          $"(values > 1 mean widen is faster)");
        Console.WriteLine();

        // Correctness: compare generated-token streams. For narrow types (BFloat16/Half)
        // the scalar fallback computes in reduced precision and legitimately diverges
        // from the numerically-correct widen path, so a match here is not required —
        // it is informational. The widen side's correctness against PyTorch is pinned
        // separately by the compare/generate modes (compare_smollm_py.bin fixtures).
        var scalarIds = scalar.Sequence.Skip(promptIds.Count).ToList();
        var widenIds = widened.Sequence.Skip(promptIds.Count).ToList();
        int match = 0;
        for (int i = 0; i < Math.Min(scalarIds.Count, widenIds.Count); i++)
            if (scalarIds[i] == widenIds[i]) match++;

        Console.WriteLine("Token equivalence (scalar vs widen):");
        Console.WriteLine($"  Generated: scalar={scalarIds.Count}  widen={widenIds.Count}  " +
                          $"match={match}/{Math.Min(scalarIds.Count, widenIds.Count)}");
        if (scalarIds.Count == widenIds.Count && match == scalarIds.Count)
            Console.WriteLine("  ✓ Scalar and widen produced identical token streams");
        else if (!narrow)
            Console.WriteLine($"  ✗ {scalarIds.Count - match} positions differ");
        else
            Console.WriteLine("  (informational) divergence is expected for the reduced-" +
                              "precision scalar fallback; the widen side is the reference-verified path");
        Console.WriteLine();

        Console.WriteLine($"Scalar tokens: [{string.Join(", ", scalarIds)}]");
        Console.WriteLine($"Widen tokens:  [{string.Join(", ", widenIds)}]");

        return 0;
    }

    static (int Generated, long Milliseconds, List<int> Sequence) RunSmolLMGeneration<T>(
        LlamaForCausalLM<T> model, IReadOnlyList<int> promptIds, LlamaConfig config, int maxNewTokens)
        where T : struct, IFloatingPointIeee754<T>
    {
        var sequence = new List<int>(promptIds);
        int generated = 0;
        var genSw = Stopwatch.StartNew();
        while (generated < maxNewTokens)
        {
            var logits = model.Forward(sequence.ToArray());
            int vocab = config.VocabSize;
            int next = ArgMaxLastRow(logits, logits.Shape[0], vocab);
            sequence.Add(next);
            generated++;
            if (next == config.EosTokenId)
                break;
        }
        genSw.Stop();
        return (generated, genSw.ElapsedMilliseconds, sequence);
    }

    static void RunSmolLMGenerate<T>(Dictionary<string, (T[] Data, int[] Shape)> tensors, string precisionName)
        where T : struct, IFloatingPointIeee754<T>
    {
        Console.WriteLine($"=== SmolLM-135M Causal LM ({precisionName}) ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})  Precision: {precisionName}");
        Console.WriteLine();

        var modelDir = Path.Combine("samples", "data", "smollm-135m");
        var config = LlamaConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json")));

        var buildSw = Stopwatch.StartNew();
        var model = LlamaLoader.Load<T, T>(config, tensors);
        buildSw.Stop();
        Console.WriteLine($"Config: hidden={config.HiddenSize}, layers={config.NumHiddenLayers}, " +
                          $"heads={config.NumAttentionHeads}, kvHeads={config.NumKeyValueHeads}, " +
                          $"intermediate={config.IntermediateSize}");
        Console.WriteLine($"Model build: {buildSw.ElapsedMilliseconds} ms");

        int totalParams = tensors.Values.Sum(t => t.Data.Length);
        int bytesPerWeight = typeof(T) == typeof(BFloat16) || typeof(T) == typeof(Half) ? 2 : 4;
        double weightMb = tensors.Values.Sum(t => t.Data.Length * (long)bytesPerWeight) / (1024.0 * 1024.0);
        Console.WriteLine($"Parameters: {totalParams:N0}");
        Console.WriteLine($"Weights ({precisionName}): {weightMb:F1} MB");
        Console.WriteLine();

        var tokenizer = new Gpt2BpeTokenizer(
            Path.Combine(modelDir, "vocab.json"), Path.Combine(modelDir, "merges.txt"));
        ConsumeGenerated(model, modelDir, tokenizer, config, precisionName);
    }

    /// <summary>
    /// Greedy-generates up to <c>maxNewTokens</c> tokens from the fixed SmolLM prompt and,
    /// when the PyTorch reference fixtures are present, diffs the token-id stream and the
    /// final-position logits. Inference-by-default: never enters a <c>GradientUtils.Grad()</c>
    /// scope, so no graph nodes are built.
    /// </summary>
    static void ConsumeGenerated<T>(
        LlamaForCausalLM<T> model,
        string modelDir,
        Gpt2BpeTokenizer tokenizer,
        LlamaConfig config,
        string precisionName)
        where T : struct, IFloatingPointIeee754<T>
    {
        const string Prompt = "The capital of France is";
        const int MaxNewTokens = 32;

        var promptIds = tokenizer.Encode(Prompt);
        Console.WriteLine($"Prompt: \"{Prompt}\"");
        Console.WriteLine($"Prompt token ids ({promptIds.Count}): [{string.Join(", ", promptIds)}]");
        Console.WriteLine();

        var (generated, genMs, sequence) = RunSmolLMGeneration(model, promptIds, config, MaxNewTokens);
        var generatedIds = sequence.Skip(promptIds.Count).ToList();

        Console.WriteLine($"Generated {generated} tokens in {genMs} ms " +
                          $"({genMs / Math.Max(1, generated)} ms/token)");
        Console.WriteLine($"Generated token ids: [{string.Join(", ", generatedIds)}]");
        Console.WriteLine($"Decoded: \"{tokenizer.Decode(sequence)}\"");
        Console.WriteLine();

        // Final-position logits for the numeric precision diff: feed the full prefix
        // (prompt + all but the last generated token) so the model predicts the last
        // generated token, matching the Python reference generator.
        var lastLogits = LastPositionLogits(model, sequence, config.VocabSize);

        CompareSmolLmFixtures(modelDir, promptIds.Count, sequence, lastLogits, precisionName);
    }

    /// <summary>Returns the argmax vocabulary index of the model's final-position logits.</summary>
    static int ArgMaxLastRow<T>(ReverseGradTensor<T> logits, int rows, int vocab)
        where T : struct, IFloatingPointIeee754<T>
    {
        logits.Data.TryGetSpan(out var span);
        int offset = (rows - 1) * vocab;
        int best = 0;
        double bestVal = double.CreateChecked(span[offset]);
        for (int i = 1; i < vocab; i++)
        {
            double v = double.CreateChecked(span[offset + i]);
            if (v > bestVal)
            {
                bestVal = v;
                best = i;
            }
        }
        return best;
    }

    /// <summary>Runs the full prefix through the model and returns the last-row logits as float[].</summary>
    static float[] LastPositionLogits<T>(
        LlamaForCausalLM<T> model, IReadOnlyList<int> sequence, int vocab)
        where T : struct, IFloatingPointIeee754<T>
    {
        var prefix = new int[sequence.Count - 1];
        for (int i = 0; i < prefix.Length; i++)
            prefix[i] = sequence[i];

        var logits = model.Forward(prefix); // [L, vocab]
        logits.Data.TryGetSpan(out var span);
        int rows = logits.Shape[0];
        int offset = (rows - 1) * vocab;

        var result = new float[vocab];
        for (int i = 0; i < vocab; i++)
            result[i] = (float)double.CreateChecked(span[offset + i]);
        return result;
    }

    static void CompareSmolLmFixtures(
        string modelDir, int inputLen, IReadOnlyList<int> fullPrefix, float[] lastLogits, string precisionName)
    {
        string tokenPath = Path.Combine(modelDir, "..", "compare_smollm_py.bin");
        string logitsPath = Path.Combine(modelDir, "..", "compare_smollm_logits_py.bin");

        if (!File.Exists(tokenPath) || !File.Exists(logitsPath))
        {
            Console.WriteLine("Reference fixtures (compare_smollm_py.bin / compare_smollm_logits_py.bin) not found;");
            Console.WriteLine("skipping the PyTorch diff. Run:");
            Console.WriteLine("  python samples/NivaraInference/Python/smollm_generate_reference.py");
            return;
        }

        // Token-id stream: [int32 input_len][int32 full prefix = prompt + generated].
        using (var br = new BinaryReader(File.OpenRead(tokenPath)))
        {
            int refInputLen = br.ReadInt32();
            int refCount = (int)(new FileInfo(tokenPath).Length - 4) / 4;
            var refIds = new int[refCount];
            for (int i = 0; i < refCount; i++)
                refIds[i] = br.ReadInt32();

            int streamLen = Math.Min(fullPrefix.Count, refIds.Length);

            int genMatch = 0;
            var mismatches = new List<string>();
            for (int i = inputLen; i < streamLen; i++)
            {
                if (fullPrefix[i] == refIds[i])
                    genMatch++;
                else if (mismatches.Count < 8)
                    mismatches.Add($"@{i}: C#={fullPrefix[i]} Py={refIds[i]}");
            }
            int genCount = Math.Max(0, streamLen - inputLen);

            Console.WriteLine($"=== PyTorch diff ({precisionName}) ===");
            Console.WriteLine($"  Reference stream: {refInputLen} prompt + {refCount - refInputLen} generated (total {refCount})");
            Console.WriteLine($"  C# stream length: {fullPrefix.Count} (prompt {inputLen} + {fullPrefix.Count - inputLen} generated)");
            Console.WriteLine($"  Generated-token argmax match: {genMatch}/{genCount}");
            if (mismatches.Count > 0)
                Console.WriteLine("  Mismatches (C# vs Py): " + string.Join(", ", mismatches));
        }

        // Final-position logits: [float32 vocab_size].
        var rawBytes = File.ReadAllBytes(logitsPath);
        float[] refLogits = new float[rawBytes.Length / 4];
        Buffer.BlockCopy(rawBytes, 0, refLogits, 0, rawBytes.Length);

        int len = Math.Min(lastLogits.Length, refLogits.Length);
        var diffArr = new float[len];
        TensorPrimitives.Subtract(lastLogits.AsSpan(0, len), refLogits.AsSpan(0, len), diffArr);
        var absDiff = new float[len];
        TensorPrimitives.Abs(diffArr.AsSpan(), absDiff);
        float maxAbs = TensorPrimitives.Max(absDiff);
        float sumAbs = TensorPrimitives.Sum(absDiff);
        float cosineSim = TensorPrimitives.CosineSimilarity(lastLogits.AsSpan(0, len), refLogits.AsSpan(0, len));

        int csArgmax = ArgMax(lastLogits);
        int pyArgmax = ArgMax(refLogits);

        Console.WriteLine($"  C# final-logits argmax: {csArgmax}");
        Console.WriteLine($"  Py final-logits argmax: {pyArgmax}");
        Console.WriteLine($"  max abs diff: {maxAbs:F6}");
        Console.WriteLine($"  mean abs diff: {sumAbs / len:F8}");
        Console.WriteLine($"  cosine similarity: {cosineSim:F8}");
        Console.WriteLine();
    }

    static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best])
                best = i;
        return best;
    }

    static int BenchmarkDistilBertSst<T>(Dictionary<string, (T[] Data, int[] Shape)> tensors, string precision)
        where T : struct, IFloatingPointIeee754<T>
    {
        Console.WriteLine("=== DistilBERT SST-2 Benchmark ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})  Precision: {precision}");
        Console.WriteLine();

        string modelDir = Path.Combine("samples", "data", "distilbert_sst");
        var buildSw = Stopwatch.StartNew();
        var model = DistilBertSst.Load<T>(tensors, modelDir);
        var tokenizer = DistilBertSst.LoadTokenizer(modelDir);
        buildSw.Stop();
        Console.WriteLine($"Model build: {buildSw.ElapsedMilliseconds} ms");

        int totalParams = DistilBertSst.CountParameters(tensors);
        Console.WriteLine($"Parameters: {totalParams:N0}");
        Console.WriteLine($"Weights: {totalParams * Unsafe.SizeOf<T>() / (1024.0 * 1024.0):F1} MB");
        Console.WriteLine();

        string text = "This is a long test sentence that will be tokenized to demonstrate the performance of the DistilBERT SST-2 model inference across multiple tokens for benchmarking purposes.";
        Console.WriteLine($"Input text length: {text.Split(' ').Length} words");
        Console.WriteLine($"Input tokens: 128");
        Console.WriteLine();

        model.Eval();
        ReportTiming<T>(() => DistilBertSst.PredictLogits<T>(model, tokenizer, text, maxLen: 128));

        var (argMax, probs) = DistilBertSst.Softmax(DistilBertSst.PredictLogits<T>(model, tokenizer, text, maxLen: 128));
        Console.WriteLine($"Last pass sentiment: {DistilBertSst.Label(argMax)} ({probs[argMax] * 100:F1}%)");
        Console.WriteLine();

        return 0;
    }

    static int RunMiniLMSimilarity(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        Console.WriteLine("=== MiniLM Cosine Similarity Demo ===");
        Console.WriteLine();

        var config = BertConfig.FromJson(File.ReadAllText(Path.Combine("samples", "data", "minilm", "config.json")));
        var model = MiniLMDistilled<float>.LoadWeights(tensors, config);

        var sentences = new[]
        {
            "This is a cat.",
            "This is a dog.",
            "I love programming.",
            "The weather is nice today.",
            "I love coding."
        };

        Console.WriteLine($"Sentences ({sentences.Length}):");
        for (int i = 0; i < sentences.Length; i++)
            Console.WriteLine($"  [{i}] {sentences[i]}");
        Console.WriteLine();

        var tokenizer = MiniLMTokenizer.Load(Path.Combine("samples", "data", "minilm", "vocab.txt"));
        var embeddings = new float[sentences.Length][];
        for (int s = 0; s < sentences.Length; s++)
        {
            var (input, mask) = MiniLMTokenizer.TokenizeWithMask(tokenizer, sentences[s], maxLen: 128);

            var output = mask != null ? model.ForwardWithMask(input, mask) : model.Forward(input);
            var outputData = new float[output.Length];
            output.Data.TryGetSpan(out var outSpan);
            if (!outSpan.IsEmpty) outSpan.CopyTo(outputData);
            embeddings[s] = outputData;
        }

        Console.WriteLine("Cosine Similarity Matrix:");
        Console.Write("       ");
        for (int i = 0; i < sentences.Length; i++)
            Console.Write($"  [{i}]   ");
        Console.WriteLine();

        for (int i = 0; i < sentences.Length; i++)
        {
            Console.Write($"  [{i}]  ");
            for (int j = 0; j < sentences.Length; j++)
            {
                float sim = TensorPrimitives.CosineSimilarity(embeddings[i].AsSpan(), embeddings[j].AsSpan());
                Console.Write($"{sim,7:F4} ");
            }
            Console.WriteLine();
        }
        Console.WriteLine();

        return 0;
    }

    static int RunDistilBertSstBFloat16(Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors)
    {
        Console.WriteLine("=== DistilBERT SST-2 BFloat16 Compare ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})  Precision: BFloat16");
        Console.WriteLine();

        string modelDir = Path.Combine("samples", "data", "distilbert_sst");
        string csPath = Path.Combine("samples", "data", "compare_distilbert_sst_bf16_cs.bin");
        string pyPath = Path.Combine("samples", "data", "compare_distilbert_sst_py.bin");

        double mbBf16 = tensors.Values.Sum(t => t.Data.Length) * 2.0 / (1024.0 * 1024.0);
        Console.WriteLine($"Weight memory (BFloat16): {mbBf16:F1} MB  (half of F32 = {mbBf16 * 2:F1} MB)");
        Console.WriteLine();

        Console.WriteLine($"Sentences ({DistilBertSst.CompareSentences.Length}):");
        for (int i = 0; i < DistilBertSst.CompareSentences.Length; i++)
            Console.WriteLine($"  [{i}] {DistilBertSst.CompareSentences[i]}");
        Console.WriteLine();

        DistilBertSst.SaveBFloat16CompareOutput(tensors, modelDir, csPath);
        DistilBertSst.PrintCompareDiff(pyPath, csPath, DistilBertSst.CompareSentences.Length);

        return 0;
    }

    static int RunDistilBertBFloat16(Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors)
    {
        string refPath = Path.Combine("samples", "data", "distilbert", "last_hidden_state_py.bin");
        if (!File.Exists(refPath))
        {
            Console.Error.WriteLine($"Reference file not found: {refPath}");
            Console.Error.WriteLine("Run: python samples/NivaraInference/Python/distilbert_compare.py");
            return 1;
        }

        Console.WriteLine("=== DistilBERT BFloat16 Compare ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})  Precision: BFloat16");
        Console.WriteLine();

        var config = DistilBertConfig.FromJson(File.ReadAllText(Path.Combine("samples", "data", "distilbert", "config.json")));
        var encoder = new BertEncoder<BFloat16>(config.ToBertConfig(), includeTokenTypeEmbedding: false);
        DistilBertLoader.LoadEncoderWeights<BFloat16, BFloat16>(encoder, tensors, "distilbert");
        encoder.Eval();

        var tokenizer = MiniLMTokenizer.Load(Path.Combine("samples", "data", "distilbert", "vocab.txt"));
        string text = "This is a test sentence.";
        var (tokenIds, attnMask, _) = MiniLMTokenizer.Encode(tokenizer, text, maxLen: 128);
        var intIds = Array.ConvertAll(tokenIds, x => (int)x);
        var mask = GradientUtils.Constant(Array.ConvertAll(attnMask, x => (BFloat16)x));

        var output = encoder.ForwardWithMask(intIds, mask);

        var outputData = new float[output.Length];
        output.Data.TryGetSpan(out var outSpan);
        if (!outSpan.IsEmpty)
        {
            int take = Math.Min(outputData.Length, outSpan.Length);
            for (int i = 0; i < take; i++)
                outputData[i] = (float)outSpan[i];
        }

        var rawBytes = File.ReadAllBytes(refPath);
        float[] refData = new float[rawBytes.Length / 4];
        Buffer.BlockCopy(rawBytes, 0, refData, 0, rawBytes.Length);

        int len = Math.Min(outputData.Length, refData.Length);
        var outputSpan = outputData.AsSpan(0, len);
        var refSpan = refData.AsSpan(0, len);

        var diffArr = new float[len];
        TensorPrimitives.Subtract(outputSpan, refSpan, diffArr);
        var absDiff = new float[len];
        TensorPrimitives.Abs(diffArr.AsSpan(), absDiff);
        float maxAbs = TensorPrimitives.Max(absDiff);
        float sumAbs = TensorPrimitives.Sum(absDiff);
        float cosineSim = TensorPrimitives.CosineSimilarity(outputSpan, refSpan);

        Console.WriteLine($"Input text: \"{text}\"");
        Console.WriteLine($"Output shape: [{string.Join(", ", output.Shape)}]");
        Console.WriteLine($"  C# stats: min={TensorPrimitives.Min(outputData.AsSpan()):F6}, max={TensorPrimitives.Max(outputData.AsSpan()):F6}, mean={TensorPrimitives.Average(outputData.AsSpan()):F6}, std={StdDev(outputData):F6}");
        Console.WriteLine($"  Py stats: min={TensorPrimitives.Min(refData.AsSpan()):F6}, max={TensorPrimitives.Max(refData.AsSpan()):F6}, mean={TensorPrimitives.Average(refData.AsSpan()):F6}, std={StdDev(refData):F6}");
        Console.WriteLine($"  max abs diff: {maxAbs:F6}");
        Console.WriteLine($"  mean abs diff: {sumAbs / len:F8}");
        Console.WriteLine($"  cosine similarity: {cosineSim:F8}");
        Console.WriteLine();
        Console.WriteLine($"Weight memory (BFloat16): {tensors.Values.Sum(t => t.Data.Length) * 2.0 / (1024.0 * 1024.0):F1} MB  (half of F32)");

        return 0;
    }

    static int RunMiniLMBFloat16(Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors)
    {
        Console.WriteLine("=== MiniLM BFloat16 Compare ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})  Precision: BFloat16");
        Console.WriteLine();

        var config = BertConfig.FromJson(File.ReadAllText(Path.Combine("samples", "data", "minilm", "config.json")));

        var buildSw = Stopwatch.StartNew();
        var model = MiniLMDistilled<BFloat16>.LoadWeights<BFloat16, BFloat16>(tensors, config);
        buildSw.Stop();
        Console.WriteLine($"Model build: {buildSw.ElapsedMilliseconds} ms");

        int totalParams = tensors.Values.Sum(t => t.Data.Length);
        Console.WriteLine($"Parameters: {totalParams:N0}");
        Console.WriteLine($"Weight memory (BFloat16): {totalParams * 2.0 / (1024.0 * 1024.0):F1} MB  (half of F32)");
        Console.WriteLine();

        var tokenizer = MiniLMTokenizer.Load(Path.Combine("samples", "data", "minilm", "vocab.txt"));

        var sentences = new[]
        {
            "This is a cat.",
            "This is a dog.",
            "I love programming.",
            "The weather is nice today.",
            "I love coding."
        };

        Console.WriteLine($"Sentences ({sentences.Length}):");
        for (int i = 0; i < sentences.Length; i++)
            Console.WriteLine($"  [{i}] {sentences[i]}");
        Console.WriteLine();

        model.Eval();

        var embeddings = new float[sentences.Length][];
        var fwdSw = Stopwatch.StartNew();
        for (int s = 0; s < sentences.Length; s++)
        {
            var (tokenIds, attnMask, _) = MiniLMTokenizer.Encode(tokenizer, sentences[s], maxLen: 128);
            var intIds = Array.ConvertAll(tokenIds, x => (int)x);
            var mask = GradientUtils.Constant(Array.ConvertAll(attnMask, x => (BFloat16)x));
            var output = model.ForwardWithMask(intIds, mask);
            var outputData = new float[output.Length];
            output.Data.TryGetSpan(out var outSpan);
            if (!outSpan.IsEmpty)
            {
                int take = Math.Min(outputData.Length, outSpan.Length);
                for (int i = 0; i < take; i++)
                    outputData[i] = (float)outSpan[i];
            }
            embeddings[s] = outputData;
        }
        fwdSw.Stop();

        double avgMs = fwdSw.ElapsedMilliseconds / (double)sentences.Length;
        Console.WriteLine($"Forward total: {fwdSw.ElapsedMilliseconds} ms across {sentences.Length} sentences ({avgMs:F1} ms/sentence)");
        Console.WriteLine();

        for (int i = 0; i < sentences.Length; i++)
        {
            var emb = embeddings[i];
            float norm = TensorPrimitives.Norm(emb.AsSpan());
            Console.WriteLine($"[{i}] {sentences[i]}");
            Console.Write($"    first 10: [");
            for (int j = 0; j < Math.Min(10, emb.Length); j++)
            {
                Console.Write($"{emb[j]:F6}");
                if (j < Math.Min(10, emb.Length) - 1) Console.Write(", ");
            }
            Console.WriteLine("]");
            Console.WriteLine($"    stats: min={TensorPrimitives.Min(emb.AsSpan()):F6}, max={TensorPrimitives.Max(emb.AsSpan()):F6}, mean={TensorPrimitives.Average(emb.AsSpan()):F6}, L2 norm={norm:F6}");
            Console.WriteLine();
        }

        Console.WriteLine("Cosine Similarity Matrix:");
        Console.Write("       ");
        for (int i = 0; i < sentences.Length; i++)
            Console.Write($"  [{i}]   ");
        Console.WriteLine();
        for (int i = 0; i < sentences.Length; i++)
        {
            Console.Write($"  [{i}]  ");
            for (int j = 0; j < sentences.Length; j++)
            {
                float sim = TensorPrimitives.CosineSimilarity(embeddings[i].AsSpan(), embeddings[j].AsSpan());
                Console.Write($"{sim,7:F4} ");
            }
            Console.WriteLine();
        }
        Console.WriteLine();

        string pyPath = Path.Combine("samples", "data", "compare_minilm_embeddings_py.bin");
        if (File.Exists(pyPath))
        {
            var rawBytes = File.ReadAllBytes(pyPath);
            float[] refData = new float[rawBytes.Length / 4];
            Buffer.BlockCopy(rawBytes, 0, refData, 0, rawBytes.Length);
            int rows = sentences.Length;
            int dim = refData.Length / rows;
            Console.WriteLine($"Cosine similarity vs F32 reference ({rows} sentences, dim {dim}):");
            for (int i = 0; i < rows; i++)
            {
                var csSpan = embeddings[i].AsSpan(0, Math.Min(embeddings[i].Length, dim));
                var pySpan = refData.AsSpan(i * dim, dim);
                float sim = TensorPrimitives.CosineSimilarity(csSpan, pySpan);
                Console.WriteLine($"  [{i}] cosine(C#, F32 reference) = {sim:F6}");
            }
            Console.WriteLine();
        }
        else
        {
            Console.WriteLine("F32 reference (compare_minilm_embeddings_py.bin) not found; skipping diff.");
            Console.WriteLine();
        }

        return 0;
    }

    static int RunDistilBertSstHalf(Dictionary<string, (Half[] Data, int[] Shape)> tensors)
    {
        Console.WriteLine("=== DistilBERT SST-2 Half Compare ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})  Precision: Half (fp16)");
        Console.WriteLine();

        string modelDir = Path.Combine("samples", "data", "distilbert_sst");
        string csPath = Path.Combine("samples", "data", "compare_distilbert_sst_fp16_cs.bin");
        string pyPath = Path.Combine("samples", "data", "compare_distilbert_sst_py.bin");

        double mbHalf = tensors.Values.Sum(t => t.Data.Length) * 2.0 / (1024.0 * 1024.0);
        Console.WriteLine($"Weight memory (Half): {mbHalf:F1} MB  (half of F32 = {mbHalf * 2:F1} MB)");
        Console.WriteLine();

        Console.WriteLine($"Sentences ({DistilBertSst.CompareSentences.Length}):");
        for (int i = 0; i < DistilBertSst.CompareSentences.Length; i++)
            Console.WriteLine($"  [{i}] {DistilBertSst.CompareSentences[i]}");
        Console.WriteLine();

        DistilBertSst.SaveHalfCompareOutput(tensors, modelDir, csPath);
        DistilBertSst.PrintCompareDiff(pyPath, csPath, DistilBertSst.CompareSentences.Length);

        return 0;
    }

    static int RunDistilBertHalf(Dictionary<string, (Half[] Data, int[] Shape)> tensors)
    {
        string refPath = Path.Combine("samples", "data", "distilbert", "last_hidden_state_py.bin");
        if (!File.Exists(refPath))
        {
            Console.Error.WriteLine($"Reference file not found: {refPath}");
            Console.Error.WriteLine("Run: python samples/NivaraInference/Python/distilbert_compare.py");
            return 1;
        }

        Console.WriteLine("=== DistilBERT Half Compare ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})  Precision: Half (fp16)");
        Console.WriteLine();

        var config = DistilBertConfig.FromJson(File.ReadAllText(Path.Combine("samples", "data", "distilbert", "config.json")));
        var encoder = new BertEncoder<Half>(config.ToBertConfig(), includeTokenTypeEmbedding: false);
        DistilBertLoader.LoadEncoderWeights<Half, Half>(encoder, tensors, "distilbert");
        encoder.Eval();

        var tokenizer = MiniLMTokenizer.Load(Path.Combine("samples", "data", "distilbert", "vocab.txt"));
        string text = "This is a test sentence.";
        var (tokenIds, attnMask, _) = MiniLMTokenizer.Encode(tokenizer, text, maxLen: 128);
        var intIds = Array.ConvertAll(tokenIds, x => (int)x);
        var mask = GradientUtils.Constant(Array.ConvertAll(attnMask, x => (Half)x));

        var output = encoder.ForwardWithMask(intIds, mask);

        var outputData = new float[output.Length];
        output.Data.TryGetSpan(out var outSpan);
        if (!outSpan.IsEmpty)
        {
            int take = Math.Min(outputData.Length, outSpan.Length);
            for (int i = 0; i < take; i++)
                outputData[i] = (float)outSpan[i];
        }

        var rawBytes = File.ReadAllBytes(refPath);
        float[] refData = new float[rawBytes.Length / 4];
        Buffer.BlockCopy(rawBytes, 0, refData, 0, rawBytes.Length);

        int len = Math.Min(outputData.Length, refData.Length);
        var outputSpan = outputData.AsSpan(0, len);
        var refSpan = refData.AsSpan(0, len);

        var diffArr = new float[len];
        TensorPrimitives.Subtract(outputSpan, refSpan, diffArr);
        var absDiff = new float[len];
        TensorPrimitives.Abs(diffArr.AsSpan(), absDiff);
        float maxAbs = TensorPrimitives.Max(absDiff);
        float sumAbs = TensorPrimitives.Sum(absDiff);
        float cosineSim = TensorPrimitives.CosineSimilarity(outputSpan, refSpan);

        Console.WriteLine($"Input text: \"{text}\"");
        Console.WriteLine($"Output shape: [{string.Join(", ", output.Shape)}]");
        Console.WriteLine($"  C# stats: min={TensorPrimitives.Min(outputData.AsSpan()):F6}, max={TensorPrimitives.Max(outputData.AsSpan()):F6}, mean={TensorPrimitives.Average(outputData.AsSpan()):F6}, std={StdDev(outputData):F6}");
        Console.WriteLine($"  Py stats: min={TensorPrimitives.Min(refData.AsSpan()):F6}, max={TensorPrimitives.Max(refData.AsSpan()):F6}, mean={TensorPrimitives.Average(refData.AsSpan()):F6}, std={StdDev(refData):F6}");
        Console.WriteLine($"  max abs diff: {maxAbs:F6}");
        Console.WriteLine($"  mean abs diff: {sumAbs / len:F8}");
        Console.WriteLine($"  cosine similarity: {cosineSim:F8}");
        Console.WriteLine();
        Console.WriteLine($"Weight memory (Half): {tensors.Values.Sum(t => t.Data.Length) * 2.0 / (1024.0 * 1024.0):F1} MB  (half of F32)");

        return 0;
    }

    static int RunMiniLMHalf(Dictionary<string, (Half[] Data, int[] Shape)> tensors)
    {
        Console.WriteLine("=== MiniLM Half Compare ===");
        Console.WriteLine($"Device: CPU (.NET {Environment.Version})  Precision: Half (fp16)");
        Console.WriteLine();

        var config = BertConfig.FromJson(File.ReadAllText(Path.Combine("samples", "data", "minilm", "config.json")));

        var buildSw = Stopwatch.StartNew();
        var model = MiniLMDistilled<Half>.LoadWeights<Half, Half>(tensors, config);
        buildSw.Stop();
        Console.WriteLine($"Model build: {buildSw.ElapsedMilliseconds} ms");

        int totalParams = tensors.Values.Sum(t => t.Data.Length);
        Console.WriteLine($"Parameters: {totalParams:N0}");
        Console.WriteLine($"Weight memory (Half): {totalParams * 2.0 / (1024.0 * 1024.0):F1} MB  (half of F32)");
        Console.WriteLine();

        var tokenizer = MiniLMTokenizer.Load(Path.Combine("samples", "data", "minilm", "vocab.txt"));

        var sentences = new[]
        {
            "This is a cat.",
            "This is a dog.",
            "I love programming.",
            "The weather is nice today.",
            "I love coding."
        };

        Console.WriteLine($"Sentences ({sentences.Length}):");
        for (int i = 0; i < sentences.Length; i++)
            Console.WriteLine($"  [{i}] {sentences[i]}");
        Console.WriteLine();

        model.Eval();

        var embeddings = new float[sentences.Length][];
        var fwdSw = Stopwatch.StartNew();
        for (int s = 0; s < sentences.Length; s++)
        {
            var (tokenIds, attnMask, _) = MiniLMTokenizer.Encode(tokenizer, sentences[s], maxLen: 128);
            var intIds = Array.ConvertAll(tokenIds, x => (int)x);
            var mask = GradientUtils.Constant(Array.ConvertAll(attnMask, x => (Half)x));
            var output = model.ForwardWithMask(intIds, mask);
            var outputData = new float[output.Length];
            output.Data.TryGetSpan(out var outSpan);
            if (!outSpan.IsEmpty)
            {
                int take = Math.Min(outputData.Length, outSpan.Length);
                for (int i = 0; i < take; i++)
                    outputData[i] = (float)outSpan[i];
            }
            embeddings[s] = outputData;
        }
        fwdSw.Stop();

        double avgMs = fwdSw.ElapsedMilliseconds / (double)sentences.Length;
        Console.WriteLine($"Forward total: {fwdSw.ElapsedMilliseconds} ms across {sentences.Length} sentences ({avgMs:F1} ms/sentence)");
        Console.WriteLine();

        for (int i = 0; i < sentences.Length; i++)
        {
            var emb = embeddings[i];
            float norm = TensorPrimitives.Norm(emb.AsSpan());
            Console.WriteLine($"[{i}] {sentences[i]}");
            Console.Write($"    first 10: [");
            for (int j = 0; j < Math.Min(10, emb.Length); j++)
            {
                Console.Write($"{emb[j]:F6}");
                if (j < Math.Min(10, emb.Length) - 1) Console.Write(", ");
            }
            Console.WriteLine("]");
            Console.WriteLine($"    stats: min={TensorPrimitives.Min(emb.AsSpan()):F6}, max={TensorPrimitives.Max(emb.AsSpan()):F6}, mean={TensorPrimitives.Average(emb.AsSpan()):F6}, L2 norm={norm:F6}");
            Console.WriteLine();
        }

        Console.WriteLine("Cosine Similarity Matrix:");
        Console.Write("       ");
        for (int i = 0; i < sentences.Length; i++)
            Console.Write($"  [{i}]   ");
        Console.WriteLine();
        for (int i = 0; i < sentences.Length; i++)
        {
            Console.Write($"  [{i}]  ");
            for (int j = 0; j < sentences.Length; j++)
            {
                float sim = TensorPrimitives.CosineSimilarity(embeddings[i].AsSpan(), embeddings[j].AsSpan());
                Console.Write($"{sim,7:F4} ");
            }
            Console.WriteLine();
        }
        Console.WriteLine();

        string pyPath = Path.Combine("samples", "data", "compare_minilm_embeddings_py.bin");
        if (File.Exists(pyPath))
        {
            var rawBytes = File.ReadAllBytes(pyPath);
            float[] refData = new float[rawBytes.Length / 4];
            Buffer.BlockCopy(rawBytes, 0, refData, 0, rawBytes.Length);
            int rows = sentences.Length;
            int dim = refData.Length / rows;
            Console.WriteLine($"Cosine similarity vs F32 reference ({rows} sentences, dim {dim}):");
            for (int i = 0; i < rows; i++)
            {
                var csSpan = embeddings[i].AsSpan(0, Math.Min(embeddings[i].Length, dim));
                var pySpan = refData.AsSpan(i * dim, dim);
                float sim = TensorPrimitives.CosineSimilarity(csSpan, pySpan);
                Console.WriteLine($"  [{i}] cosine(C#, F32 reference) = {sim:F6}");
            }
            Console.WriteLine();
        }
        else
        {
            Console.WriteLine("F32 reference (compare_minilm_embeddings_py.bin) not found; skipping diff.");
            Console.WriteLine();
        }

        return 0;
    }
}
