using System.Buffers;
using System.Diagnostics;
using SolTTS.Core.Core;
using NAudio.Wave;

namespace SolTTS.AudioBench;

/// <summary>
/// Real-time audio path benchmark: managed implementation vs native/sol_audio.
/// Measures (a) AudioPipeline per-chunk processing latency and allocations, (b) SPSC ring buffer throughput.
/// Outputs managed data only when the native DLL is missing.
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        int chunkSamples = int.TryParse(GetArg(args, "--chunk") ?? "1024", out var cs) ? cs : 1024;
        int iters = int.TryParse(GetArg(args, "--iters") ?? "20000", out var it) ? it : 20000;

        Console.WriteLine($"[audio] native available={NativeAudio.IsAvailable}" +
                          (NativeAudio.IsAvailable ? $" version=0x{NativeAudio.Version:X}" : $" ({NativeAudio.UnavailableReason})"));
        Console.WriteLine($"[audio] chunk={chunkSamples} samples ({chunkSamples * 1000.0 / 32000:F1} ms) iters={iters}");

        RunPipelineBench("managed", useNative: false, chunkSamples, iters);
        if (NativeAudio.IsAvailable)
            RunPipelineBench("native ", useNative: true, chunkSamples, iters);

        RunRingBench(iters);

        return 0;
    }

    private static void RunPipelineBench(string label, bool useNative, int chunkSamples, int iters)
    {
        using var pipeline = new AudioPipeline(32000, 20, 20, useNative);
        var rng = new Random(42);
        var input = new float[chunkSamples];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(Math.Sin(i * 0.01) * 0.3);

        // Warm-up
        for (int i = 0; i < 200; i++)
        {
            var (buf, _) = pipeline.ProcessChunk(input, chunkSamples, false);
            ArrayPool<byte>.Shared.Return(buf);
        }

        // Steady-state latency: reuse the same input (filtered in place, but shape and branches are unchanged)
        var blockTimes = new double[iters];
        for (int i = 0; i < iters; i++)
        {
            var sw = Stopwatch.StartNew();
            var (buf, len) = pipeline.ProcessChunk(input, chunkSamples, false);
            sw.Stop();
            blockTimes[i] = sw.Elapsed.TotalMicroseconds;
            ArrayPool<byte>.Shared.Return(buf);
        }

        Array.Sort(blockTimes);
        double mean = blockTimes.Average();
        double p50 = blockTimes[iters / 2];
        double p99 = blockTimes[(int)(iters * 0.99)];
        double p999 = blockTimes[(int)(iters * 0.999)];

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++)
        {
            var (buf, _) = pipeline.ProcessChunk(input, chunkSamples, false);
            ArrayPool<byte>.Shared.Return(buf);
        }
        long allocPerCall = (GC.GetAllocatedBytesForCurrentThread() - before) / 1000;

        Console.WriteLine($"[audio] pipeline {label}: mean {mean,8:F1} us  p50 {p50,8:F1}  p99 {p99,8:F1}  " +
                          $"p99.9 {p999,8:F1}  alloc {allocPerCall} B/call");
    }

    private static void RunRingBench(int iters)
    {
        const int chunk = 1024; // bytes
        var format = new WaveFormat(32000, 16, 1);
        var payload = new byte[chunk];
        Random.Shared.NextBytes(payload);
        var outBuf = new byte[chunk];

        foreach (var useNative in NativeAudio.IsAvailable ? new[] { false, true } : new[] { false })
        {
            using var ring = new LockFreeWaveProvider(format, 65536, useNative);
            ring.ReadFully = false;

            // One round: fill, then drain
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
            {
                ring.AddSamples(payload, 0, chunk);
                int got = ring.Read(outBuf, 0, chunk);
                if (got != chunk) throw new Exception($"ring underrun: {got}");
            }
            sw.Stop();
            double usPerPair = sw.Elapsed.TotalMicroseconds / iters;
            Console.WriteLine($"[audio] ring {(useNative ? "native " : "managed")}: " +
                              $"{usPerPair:F3} us per write+read pair ({chunk}B)");
        }
    }

    private static string? GetArg(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase)) return args[i + 1];
        return null;
    }
}
