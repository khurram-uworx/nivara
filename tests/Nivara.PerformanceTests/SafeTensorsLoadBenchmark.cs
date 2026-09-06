using Nivara.Samples;
using System.Diagnostics;

namespace Nivara.PerformanceTests;

/// <summary>
/// On-demand A/B for the safetensors string-path load (#392): the memory-mapped read
/// (<c>SafeTensorsLoader.Read(path)</c>) vs the copy-into-<c>byte[]</c> read
/// (<c>SafeTensorsLoader.Read(File.ReadAllBytes(path))</c>). Reports per-load ms and
/// managed-heap high-water via a <c>GC.GetTotalMemory</c> background sampler. Run only when
/// explicitly requested (<c>--safetensors-mmap</c>) — never part of the default scenario suite,
/// since a Qwen-sized load takes ~1 s and dominates the gate runs.
/// </summary>
static class SafeTensorsLoadBenchmark
{
    const int Rounds = 3;

    public static void Run(string[] args)
    {
        string path = args.Length > 1 && !args[1].StartsWith('-')
            ? args[1]
            : Path.Combine(Environment.CurrentDirectory, "samples", "data", "qwen2.5-0.5b-instruct", "model.safetensors");

        if (!File.Exists(path))
        {
            Console.Error.WriteLine($"Safetensors file not found: {path}");
            Console.Error.WriteLine("Usage: Nivara.PerformanceTests --safetensors-mmap [<path>]");
            return;
        }

        long fileLength = new FileInfo(path).Length;
        Console.WriteLine("SafeTensors load A/B (#392): memory-mapped vs byte[]");
        Console.WriteLine($"  Runtime: {Environment.Version}, {Environment.ProcessorCount} logical processors, {(Environment.Is64BitProcess ? "x64" : "x86")}");
        Console.WriteLine($"  File   : {path} ({fileLength / (1024.0 * 1024.0):F0} MB)");
        Console.WriteLine($"  Rounds : {Rounds} (alternating; both paths JIT-warmed first)");
        Console.WriteLine();

        // JIT-warm both paths (not counted).
        SafeTensorsLoader.Read(File.ReadAllBytes(path));
        SafeTensorsLoader.Read(path);
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();

        var bufferedMs = new List<long>();
        var mappedMs = new List<long>();
        long bufferedPeak = 0, mappedPeak = 0;

        for (int round = 0; round < Rounds; round++)
        {
            var (ms, peak) = Measure("byte[] (ReadAllBytes)", path, mapped: false);
            bufferedMs.Add(ms);
            bufferedPeak = Math.Max(bufferedPeak, peak);

            (ms, peak) = Measure("memory-mapped       ", path, mapped: true);
            mappedMs.Add(ms);
            mappedPeak = Math.Max(mappedPeak, peak);
        }

        Console.WriteLine();
        Console.WriteLine($"byte[] load: {string.Join(", ", bufferedMs)} ms; median {Median(bufferedMs)} ms; managed-heap high-water max {bufferedPeak / (1024.0 * 1024.0):F0} MB");
        Console.WriteLine($"mmap   load: {string.Join(", ", mappedMs)} ms; median {Median(mappedMs)} ms; managed-heap high-water max {mappedPeak / (1024.0 * 1024.0):F0} MB");
        Console.WriteLine();
        Console.WriteLine($"Managed-heap high-water delta: {(bufferedPeak - mappedPeak) / (1024.0 * 1024.0):F0} MB (mmap saves the full-file byte[] copy; physical working set is similar either way)");
    }

    static (long Ms, long PeakManaged) Measure(string label, string path, bool mapped)
    {
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();

        long localPeak = GC.GetTotalMemory(false);
        bool stop = false;
        var sampler = new Thread(() =>
        {
            while (!Volatile.Read(ref stop))
            {
                long m = GC.GetTotalMemory(false);
                if (m > Volatile.Read(ref localPeak)) Volatile.Write(ref localPeak, m);
                Thread.SpinWait(1000);
            }
        });
        sampler.IsBackground = true;
        sampler.Start();

        var sw = Stopwatch.StartNew();
        var tensors = mapped
            ? SafeTensorsLoader.Read(path)
            : SafeTensorsLoader.Read(File.ReadAllBytes(path));
        sw.Stop();
        Volatile.Write(ref stop, true);
        sampler.Join();

        long retained = GC.GetTotalMemory(forceFullCollection: true) / (1024 * 1024);
        Console.WriteLine($"  {label}: {sw.ElapsedMilliseconds} ms | sampled peak {localPeak / (1024.0 * 1024.0):F0} MB | retained after GC {retained} MB | {tensors.Count} tensors");
        return (sw.ElapsedMilliseconds, localPeak);
    }

    static long Median(List<long> values)
    {
        var copy = values.OrderBy(v => v).ToList();
        return copy[copy.Count / 2];
    }
}