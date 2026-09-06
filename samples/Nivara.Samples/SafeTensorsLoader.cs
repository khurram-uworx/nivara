using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Nivara.Samples;

public static class SafeTensorsLoader
{
    public static Dictionary<string, (float[] Data, int[] Shape)> Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"SafeTensors file not found: {path}", path);

        return Read(File.ReadAllBytes(path));
    }

    public static Dictionary<string, (float[] Data, int[] Shape)> Read(byte[] bytes)
        => Read<float>(bytes).ToDictionary(kvp => kvp.Key, kvp => (kvp.Value.Data, kvp.Value.Shape));

    public static Dictionary<string, (T[] Data, int[] Shape)> Read<T>(string path)
        where T : struct, IFloatingPointIeee754<T>
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"SafeTensors file not found: {path}", path);

        return Read<T>(File.ReadAllBytes(path));
    }

    public static Dictionary<string, (T[] Data, int[] Shape)> Read<T>(byte[] bytes)
        where T : struct, IFloatingPointIeee754<T>
    {
        var entries = ParseHeader(bytes, out int dataOffset);
        var dataBuffer = bytes.AsSpan(dataOffset);
        var result = new Dictionary<string, (T[] Data, int[] Shape)>(StringComparer.Ordinal);

        foreach (var (name, dtype, shape, begin, end) in entries)
        {
            ReadOnlySpan<byte> tensorBytes = dataBuffer.Slice(begin, end - begin);
            T[] data = DtypeToArray<T>(tensorBytes, dtype, name);

            result[name] = (data, shape);
        }

        return result;
    }

    /// <summary>
    /// Widens raw BF16 bit patterns to float32. BF16 is the high 16 bits of float32, so the
    /// widening is a pure left-shift by 16 of the bit pattern — SIMD-over
    /// <see cref="Vector{ushort}"/> (hardware-accelerated) via WidenLower/Upper + shift, with a
    /// scalar tail for the remainder. Element count must match.
    /// </summary>
    public static void WidenBf16ToF32(ReadOnlySpan<ushort> source, Span<float> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Source and destination lengths must match.", nameof(destination));

        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int ushortsPerVector = Vector<ushort>.Count;
            int floatsPerVector = Vector<float>.Count; // ushortsPerVector / 2
            int simdLimit = source.Length - (source.Length % ushortsPerVector);

            for (; i < simdLimit; i += ushortsPerVector)
            {
                var packed = new Vector<ushort>(source.Slice(i, ushortsPerVector));
                var lower = Vector.WidenLower(packed); // Vector<uint>
                var upper = Vector.WidenUpper(packed);
                lower <<= 16;
                upper <<= 16;
                var lowerF = Unsafe.As<Vector<uint>, Vector<float>>(ref lower);
                var upperF = Unsafe.As<Vector<uint>, Vector<float>>(ref upper);
                lowerF.CopyTo(destination.Slice(i, floatsPerVector));
                upperF.CopyTo(destination.Slice(i + floatsPerVector, floatsPerVector));
            }
        }

        for (; i < source.Length; i++)
            destination[i] = BitConverter.UInt32BitsToSingle((uint)source[i] << 16);
    }

    static (string Name, string Dtype, int[] Shape, int Begin, int End)[] ParseHeader(byte[] bytes, out int dataOffset)
    {
        if (bytes.Length < 8)
            throw new InvalidDataException("SafeTensors file is too small to contain a header.");

        ulong headerSize = BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(0, 8));

        if (8 + headerSize > (ulong)bytes.Length)
            throw new InvalidDataException(
                $"Header size ({headerSize}) exceeds file size ({bytes.Length}).");

        dataOffset = 8 + (int)headerSize;

        var headerJson = System.Text.Encoding.UTF8.GetString(bytes, 8, (int)headerSize);
        using var doc = JsonDocument.Parse(headerJson);

        var root = doc.RootElement;
        var entries = new List<(string, string, int[], int, int)>();

        foreach (var property in root.EnumerateObject())
        {
            if (property.Name == "__metadata__")
                continue;

            var tensor = property.Value;
            string dtype = tensor.GetProperty("dtype").GetString()!;
            string name = property.Name;
            int elementSize = DtypeByteSize(dtype);

            var shapeArray = tensor.GetProperty("shape");
            var offsets = tensor.GetProperty("data_offsets");
            int begin = offsets[0].GetInt32();
            int end = offsets[1].GetInt32();

            int[] shape = new int[shapeArray.GetArrayLength()];
            for (int i = 0; i < shape.Length; i++)
                shape[i] = shapeArray[i].GetInt32();

            int byteLength = end - begin;
            int elementCount = 1;
            foreach (var d in shape)
                elementCount *= d;

            int expectedBytes = elementCount * elementSize;
            if (byteLength != expectedBytes)
                throw new InvalidDataException(
                    $"Tensor '{name}': expected {expectedBytes} bytes ({elementCount} × {elementSize}), got {byteLength} bytes.");

            entries.Add((name, dtype, shape, begin, end));
        }

        return entries.ToArray();
    }

    static int DtypeByteSize(string dtype) => dtype switch
    {
        "F32" or "I32" => 4,
        "F16" or "BF16" => 2,
        "I64" => 8,
        _ => throw new NotSupportedException($"Unsupported dtype '{dtype}'.")
    };

    static T[] DtypeToArray<T>(ReadOnlySpan<byte> tensorBytes, string dtype, string name)
        where T : struct, IFloatingPointIeee754<T> => dtype switch
        {
            "F32" => ConvertF32<T>(tensorBytes),
            "I32" => ConvertI32<T>(tensorBytes),
            "I64" => ConvertI64<T>(tensorBytes),
            "F16" => ConvertF16<T>(tensorBytes),
            "BF16" => typeof(T) == typeof(BFloat16)
                ? (T[])(object)ConvertBF16ToBFloat16(tensorBytes)
                : ConvertBF16<T>(tensorBytes),
            _ => throw new NotSupportedException($"Tensor '{name}' has unsupported dtype '{dtype}'. " +
                "Supported dtypes: F32, I32, I64, F16, BF16.")
        };

    static T[] ConvertF32<T>(ReadOnlySpan<byte> bytes)
        where T : struct, IFloatingPointIeee754<T>
    {
        var src = MemoryMarshal.Cast<byte, float>(bytes);
        var result = new T[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = T.CreateChecked(src[i]);
        return result;
    }

    static T[] ConvertI32<T>(ReadOnlySpan<byte> bytes)
        where T : struct, IFloatingPointIeee754<T>
    {
        var src = MemoryMarshal.Cast<byte, int>(bytes);
        var result = new T[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = T.CreateChecked(src[i]);
        return result;
    }

    static T[] ConvertI64<T>(ReadOnlySpan<byte> bytes)
        where T : struct, IFloatingPointIeee754<T>
    {
        var src = MemoryMarshal.Cast<byte, long>(bytes);
        var result = new T[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = T.CreateChecked(src[i]);
        return result;
    }

    static T[] ConvertF16<T>(ReadOnlySpan<byte> bytes)
        where T : struct, IFloatingPointIeee754<T>
    {
        var src = MemoryMarshal.Cast<byte, Half>(bytes);
        var result = new T[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = T.CreateChecked(src[i]);
        return result;
    }

    static T[] ConvertBF16<T>(ReadOnlySpan<byte> bytes)
        where T : struct, IFloatingPointIeee754<T>
    {
        var src = MemoryMarshal.Cast<byte, ushort>(bytes);
        var result = new T[src.Length];

        if (typeof(T) == typeof(float))
        {
            // Eager BF16 -> F32 translation at load time, SIMD via Vector<ushort>.
            WidenBf16ToF32(src, MemoryMarshal.Cast<T, float>(result));
            return result;
        }

        for (int i = 0; i < src.Length; i++)
        {
            uint bits = (uint)src[i] << 16;
            float f = Unsafe.As<uint, float>(ref bits);
            result[i] = T.CreateChecked(f);
        }
        return result;
    }

    // Zero-hop path for the native BF16 read: when the on-disk dtype is BF16 and
    // the target is BFloat16, the raw 16-bit patterns already *are* the BFloat16
    // memory layout, so reinterpret the bytes directly. This avoids the generic
    // ConvertBF16<T> BF16 -> F32 -> BF16 round-trip (lossless, but redundant work).
    static BFloat16[] ConvertBF16ToBFloat16(ReadOnlySpan<byte> bytes)
    {
        var src = MemoryMarshal.Cast<byte, BFloat16>(bytes);
        var result = new BFloat16[src.Length];
        src.CopyTo(result);
        return result;
    }
}
