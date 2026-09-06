using Nivara.Samples;
using NUnit.Framework;
using System.Buffers.Binary;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Kernel + read-path verification for the BF16→F32 loader. The SIMD widening must be
/// bit-exact against the scalar BF16→F32 rule (<c>float bits = ushortBits &lt;&lt; 16</c>) for
/// every possible 16-bit pattern, and the fused <c>SafeTensorsLoader.Read&lt;float&gt;</c> read
/// must reproduce the same tensors from a real BF16 checkpoint (skipped when the model
/// files are absent).
/// </summary>
[TestFixture]
public class SafeTensorsLoaderBf16Tests
{
    static string ModelDir
        => Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..",
            "samples", "data", "qwen2.5-0.5b-instruct");

    [Test]
    public void WidenBf16ToF32_AllBitPatterns_MatchesScalarReference()
    {
        var src = new ushort[65536];
        for (int i = 0; i < src.Length; i++)
            src[i] = (ushort)i;

        var dst = new float[src.Length];
        SafeTensorsLoader.WidenBf16ToF32(src, dst);

        for (int i = 0; i < src.Length; i++)
        {
            uint expectedBits = (uint)i << 16;
            Assert.That(BitConverter.SingleToUInt32Bits(dst[i]), Is.EqualTo(expectedBits),
                $"pattern 0x{i:X4}: expect bit-exact 0x{expectedBits:X8}, got 0x{BitConverter.SingleToUInt32Bits(dst[i]):X8}");
        }

        // Spot truth-checks for representative BF16 values.
        Assert.That(dst[0x3F80], Is.EqualTo(1f));       // 1.0
        Assert.That(dst[0x4000], Is.EqualTo(2f));       // 2.0
        Assert.That(dst[0xC000], Is.EqualTo(-2f));      // -2.0
        Assert.That(dst[0x0000], Is.EqualTo(0f));       // 0.0
        Assert.That(float.IsPositiveInfinity(dst[0x7F80]), Is.True); // +inf
    }

    [Test]
    public void WidenBf16ToF32_VariousLengths_MatchesScalarReference()
    {
        var rng = new Random(1337);
        foreach (int length in new[] { 0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 100, 10001 })
        {
            var src = new ushort[length];
            var expected = new float[length];
            for (int i = 0; i < length; i++)
            {
                src[i] = (ushort)rng.Next(0, 65536);
                expected[i] = BitConverter.UInt32BitsToSingle((uint)src[i] << 16);
            }

            var dst = new float[length];
            SafeTensorsLoader.WidenBf16ToF32(src, dst);

            Assert.That(dst, Is.EqualTo(expected), $"length {length} must match scalar reference");
        }
    }

    [Test]
    public void WidenBf16ToF32_LengthMismatch_Throws()
    {
        var src = new ushort[4];
        var dst = new float[3];
        Assert.Throws<ArgumentException>(() => SafeTensorsLoader.WidenBf16ToF32(src, dst));
    }

    [Test]
    public void ReadFloat_OnBf16Fixture_MatchesScalarReference()
    {
        (byte[] file, string[] tensorNames) = BuildBf16Fixture();

        var direct = SafeTensorsLoader.Read<float>(file);

        Assert.That(direct.Keys, Is.EquivalentTo(tensorNames));
        foreach (var name in tensorNames)
        {
            // Fused Read<float> must produce the scalar BF16->F32 widen (left-shift by 16) elementwise.
            var expected = BuildBf16ScalarF32(name);
            Assert.That(direct[name].Data, Is.EqualTo(expected), $"{name}: fused BF16->F32 must match the scalar reference");
        }
    }

    [Test]
    public void ReadFloat_OnQwenCheckpoint_LoadsAll290TensorsWithExpectedShapes()
    {
        var safetensors = Path.Combine(ModelDir, "model.safetensors");
        if (!File.Exists(safetensors))
            Assert.Ignore("Qwen safetensors absent; skipping fused BF16->F32 checkpoint verification.");

        var stopwatch = Stopwatch.StartNew();
        var direct = SafeTensorsLoader.Read<float>(safetensors);
        stopwatch.Stop();

        // Structural parity against Qwen2.5-0.5B config values.
        Assert.That(direct.Count, Is.EqualTo(290));
        Assert.That(direct["model.embed_tokens.weight"].Shape, Is.EqualTo(new[] { 151936, 896 }));
        Assert.That(direct["model.norm.weight"].Shape, Is.EqualTo(new[] { 896 }));
        Assert.That(direct["model.layers.0.self_attn.q_proj.weight"].Shape, Is.EqualTo(new[] { 896, 896 }));
        Assert.That(direct["model.layers.0.self_attn.q_proj.bias"].Shape, Is.EqualTo(new[] { 896 }));
        Assert.That(direct["model.layers.0.self_attn.v_proj.weight"].Shape, Is.EqualTo(new[] { 128, 896 }));

        TestContext.Out.WriteLine(
            $"Qwen BF16 load: Read<float> (fused) {stopwatch.ElapsedMilliseconds} ms ({direct.Count} tensors).");
    }

    /// <summary>Builds a small, valid safetensors BF16 file with three tensors (mixed ranks).</summary>
    static (byte[] File, string[] Names) BuildBf16Fixture() => BuildBf16Fixture(BuildBf16Sections());

    static (byte[] File, string[] Names) BuildBf16Fixture(
        (string Name, int[] Shape, ushort[] Patterns)[] sections)
    {
        var names = sections.Select(s => s.Name).ToArray();

        int offset = 0;
        var builder = new StringBuilder("{");
        foreach (var (name, shape, patterns) in sections)
        {
            int end = offset + patterns.Length * sizeof(ushort);
            if (builder.Length > 1) builder.Append(',');
            builder.Append($"\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[{string.Join(",", shape)}],\"data_offsets\":[{offset},{end}]}}");
            offset = end;
        }
        builder.Append(",\"__metadata__\":{}");
        builder.Append('}');

        var headerBytes = Encoding.UTF8.GetBytes(builder.ToString());
        var file = new byte[8 + headerBytes.Length + offset];
        BinaryPrimitives.WriteUInt64LittleEndian(file, (ulong)headerBytes.Length);
        headerBytes.CopyTo(file.AsSpan(8, headerBytes.Length));

        int dataStart = 8 + headerBytes.Length;
        int cursor = 0;
        foreach (var (_, _, patterns) in sections)
        {
            int len = patterns.Length * sizeof(ushort);
            MemoryMarshal.AsBytes(patterns.AsSpan()).CopyTo(file.AsSpan(dataStart + cursor, len));
            cursor += len;
        }

        return (file, names);
    }

    /// <summary>Shared 3-tensor BF16 fixture (weights, bias, vector) with raw ushort patterns.</summary>
    static (string Name, int[] Shape, ushort[] Patterns)[] BuildBf16Sections()
    {
        var weightP = new ushort[] { 0x3F80, 0x4000, 0xC000, 0x3F00 }; // 1, 2, -2, 0.5
        var biasP = new ushort[] { 0xBF80, 0x3F80 };                  // -1, 1
        var vecP = new ushort[] { 0x0000, 0x7F80, 0x7FC0 };           // 0, +inf, NaN

        return new (string Name, int[] Shape, ushort[] Patterns)[]
        {
            ("w", new[] { 2, 2 }, weightP),
            ("b", new[] { 2 }, biasP),
            ("v", new[] { 3 }, vecP),
        };
    }

    /// <summary>Scalar BF16→F32 reference for a named fixture tensor (float bits = ushort &lt;&lt; 16).</summary>
    static float[] BuildBf16ScalarF32(string name)
    {
        var patterns = BuildBf16Sections().First(s => s.Name == name).Patterns;
        var result = new float[patterns.Length];
        for (int i = 0; i < patterns.Length; i++)
            result[i] = BitConverter.UInt32BitsToSingle((uint)patterns[i] << 16);
        return result;
    }
}