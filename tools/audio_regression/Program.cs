using System.Text;
using SolTTS.Core.Config;
using SolTTS.Core.Core;

namespace SolTTS.AudioRegression;

/// <summary>
/// Structural regression check: for the same text under serial (psent=1) and parallel (psent=N),
/// the sentence count must match, the sentence order must be consistent, and each sentence's token/sample length must stay within sampling noise.
///
/// Note: the T2S graph contains RandomNormalLike/TopK sampling, so two runs cannot be bit-identical;
/// this therefore validates structure rather than waveform. For stricter audio comparison, use manual A/B.
///
/// Exit code: 0 = pass; 1 = structural mismatch.
/// </summary>
internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        string resources = GetArg(args, "--resources") ?? "resources";
        int degree = int.TryParse(GetArg(args, "--degree") ?? "3", out var d) ? d : 3;
        int tolerancePct = int.TryParse(GetArg(args, "--tolerance") ?? "25", out var t) ? t : 25;

        var config = TTSConfig.Load();
        config.ResourcesDir = Path.GetFullPath(resources);
        config.UseEmotionControl = true;
        config.PoolCapacity = 1;

        string multiText = "今天天气真好。我们一起去公园散步吧。晚上吃点什么好呢？";

        PerfStats.Enabled = true;
        bool ok = true;

        Console.WriteLine("[regress] serial pass 1...");
        var serial = await RunOnce(config, multiText, parallelDegree: 1);

        Console.WriteLine("[regress] serial pass 2 (noise baseline)...");
        var serial2 = await RunOnce(config, multiText, parallelDegree: 1);

        Console.WriteLine("[regress] parallel pass...");
        var parallel = await RunOnce(config, multiText, parallelDegree: degree);

        // Sampling-noise baseline: serial vs serial length difference. Parallel-vs-serial difference should not significantly exceed this baseline.
        double baseline = MaxTokenDelta(serial, serial2);
        double actual = MaxTokenDelta(serial, parallel);
        Console.WriteLine($"[regress] sampling-noise baseline (serial vs serial): max Δ={baseline:F1}%");
        Console.WriteLine($"[regress] parallel vs serial:                        max Δ={actual:F1}%");
        if (actual > baseline * 1.5 + 5)
        {
            Console.WriteLine("[regress] FAIL parallel divergence exceeds sampling-noise baseline");
            ok = false;
        }

        if (serial.Count != parallel.Count)
        {
            Console.WriteLine($"[regress] FAIL sentence count: serial={serial.Count} parallel={parallel.Count}");
            ok = false;
        }

        int n = Math.Min(serial.Count, parallel.Count);
        for (int i = 0; i < n; i++)
        {
            var s = serial[i];
            var p = parallel[i];
            Console.WriteLine($"[regress] sentence {i}: serial tokens={s.Tokens,4} samples={s.Samples,7} | " +
                              $"parallel tokens={p.Tokens,4} samples={p.Samples,7} | token Δ={Pct(s.Tokens, p.Tokens):F1}%");

            if (s.Tokens <= 0 || p.Tokens <= 0)
            {
                Console.WriteLine($"[regress] FAIL sentence {i} produced no audio");
                ok = false;
                continue;
            }

            double delta = Pct(s.Tokens, p.Tokens);
            if (delta > tolerancePct)
            {
                Console.WriteLine($"[regress] FAIL sentence {i} token length differs by {delta:F1}% (>{tolerancePct}%)");
                ok = false;
            }
        }

        // Sentence-order check: the parallel implementation's AddSentence indices must strictly increase 0..n-1
        var parallelIdx = parallel.Select(x => x.Index).ToArray();
        for (int i = 0; i < parallelIdx.Length; i++)
        {
            if (parallelIdx[i] != i)
            {
                Console.WriteLine($"[regress] FAIL ordering: expected index {i}, got {parallelIdx[i]}");
                ok = false;
            }
        }

        Console.WriteLine(ok ? "[regress] PASS" : "[regress] FAILED");
        return ok ? 0 : 1;
    }

    private static async Task<List<(int Index, int Tokens, int Samples)>> RunOnce(
        TTSConfig config, string text, int parallelDegree)
    {
        config.ParallelSentenceDegree = parallelDegree;
        using var sdk = new SolTtsSdk(config);
        await sdk.InitializeAsync();

        // Warm-up (load reference features / JIT), not counted in stats
        PerfStats.DrainSentences();
        await sdk.PredictAsync("预热。", new TtsOptions(), config.DefaultAvatarId, null);
        PerfStats.DrainSentences();

        await sdk.PredictAsync(text, new TtsOptions(), config.DefaultAvatarId, null);
        var sentences = PerfStats.DrainSentences();
        await Task.CompletedTask;
        return sentences;
    }

    private static double MaxTokenDelta(
        List<(int Index, int Tokens, int Samples)> a,
        List<(int Index, int Tokens, int Samples)> b)
    {
        int n = Math.Min(a.Count, b.Count);
        double max = 0;
        for (int i = 0; i < n; i++) max = Math.Max(max, Pct(a[i].Tokens, b[i].Tokens));
        return max;
    }

    private static double Pct(int a, int b)
    {
        if (a == 0) return b == 0 ? 0 : 100;
        return Math.Abs(a - b) * 100.0 / a;
    }

    private static string? GetArg(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase)) return args[i + 1];
        return null;
    }
}
