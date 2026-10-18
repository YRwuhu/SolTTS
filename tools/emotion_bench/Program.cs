using System.Diagnostics;
using System.Globalization;
using SolTTS.Core.Config;
using SolTTS.Core.Core;

namespace SolTTSEmotionBench;

/// <summary>
/// Emotion/performance optimization benchmark tool.
///   A. Single-sentence synthesis (steady state, no emotion switching)
///   B. Multi-sentence synthesis (verifies the cross-sentence pipeline)
///   C. Emotion switching (switch emotion per sentence, verifies the lightweight recompute path)
/// Use --gol to override GraphOptimizationLevel for comparison experiments.
/// </summary>
internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        string? configPath = GetArg(args, "--config");
        string? resources = GetArg(args, "--resources");
        int nSingle = int.TryParse(GetArg(args, "--n-single") ?? "5", out var a) ? a : 5;
        int nMulti = int.TryParse(GetArg(args, "--n-multi") ?? "3", out var b) ? b : 3;
        int nSwitch = int.TryParse(GetArg(args, "--n-switch") ?? "6", out var c) ? c : 6;
        int? gol = int.TryParse(GetArg(args, "--gol") ?? "", out var g) ? g : null;
        int? intra = int.TryParse(GetArg(args, "--intra") ?? "", out var it) ? it : null;
        int? inter = int.TryParse(GetArg(args, "--inter") ?? "", out var ir) ? ir : null;
        bool? spin = bool.TryParse(GetArg(args, "--spin") ?? "", out var sp) ? sp : null;
        bool? parallel = bool.TryParse(GetArg(args, "--parallel") ?? "", out var pl) ? pl : null;
        int sessions = int.TryParse(GetArg(args, "--sessions") ?? "", out var ss) ? ss : 0;
        int? parallelSentences = int.TryParse(GetArg(args, "--psent") ?? "", out var ps) ? ps : null;
        float alpha = float.TryParse(GetArg(args, "--alpha") ?? "0.3",
            NumberStyles.Float, CultureInfo.InvariantCulture, out var al) ? al : 0.3f;
        string avatarId = GetArg(args, "--avatar") ?? "chenxing";

        const string singleText = "今天天气真好，我们一起去公园散步吧。";
        const string multiText = "今天天气真好。我们一起去公园散步吧。晚上吃点什么好呢？";

        var config = configPath != null ? TTSConfig.Load(configPath) : TTSConfig.Load();
        config.UseEmotionControl = true;
        if (resources != null) config.ResourcesDir = Path.GetFullPath(resources);
        if (gol.HasValue) config.GraphOptimizationLevel = gol.Value;
        if (intra.HasValue) config.IntraOpNumThreads = intra.Value;
        if (inter.HasValue) config.InterOpNumThreads = inter.Value;
        if (spin.HasValue) config.IntraOpAllowSpinning = spin.Value;
        if (parallel.HasValue) config.ParallelExecution = parallel.Value;
        if (parallelSentences.HasValue) config.ParallelSentenceDegree = parallelSentences.Value;
        if (sessions > 0)
        {
            config.PoolCapacity = sessions;
            // For true multi-session parallelism, each must load independently and cannot share a Session
            config.ReuseMemory = false;
        }

        Console.WriteLine($"[bench] GOL={config.GraphOptimizationLevel} threads={config.IntraOpNumThreads} " +
                          $"inter={config.InterOpNumThreads} spin={config.IntraOpAllowSpinning} parallel={config.ParallelExecution} " +
                          $"vocoderGgml={config.UseVocoderGgml} emotion={config.UseEmotionControl} " +
                          $"pool={config.PoolCapacity} psent={config.ParallelSentenceDegree} " +
                          $"perfStats={PerfStats.Enabled}");

        using var sdk = new SolTtsSdk(config);
        var initSw = Stopwatch.StartNew();
        await sdk.InitializeAsync();
        initSw.Stop();
        Console.WriteLine($"[bench] init {initSw.ElapsedMilliseconds} ms");

        async Task<(double ms, int len, int tokens)> Timed(string text, string? emo)
        {
            var opt = new TtsOptions { EmotionId = emo, EmotionAlpha = emo == null ? null : alpha };
            var sw = Stopwatch.StartNew();
            var r = await sdk.PredictAsync(text, opt, avatarId, null);
            sw.Stop();
            return (sw.Elapsed.TotalMilliseconds, r.Audio.Length, r.TokenCount);
        }

        // Warm-up (includes first-sentence JIT / memory allocation)
        await Timed(singleText, null);

        if (PerfStats.Enabled) PerfStats.Reset();
        var results = new List<(string Case, double Ms, int Len)>();

        Console.WriteLine($"\n== A. Single-sentence steady state x{nSingle} ==");
        for (int i = 0; i < nSingle; i++)
        {
            var (ms, len, _) = await Timed(singleText, null);
            Console.WriteLine($"  A[{i}] {ms,8:F0} ms  ({len / 32000.0:F2}s audio)");
            results.Add(("A_single", ms, len));
        }

        Console.WriteLine($"\n== B. Multi-sentence (3 sentences) x{nMulti} ==");
        for (int i = 0; i < nMulti; i++)
        {
            var (ms, len, _) = await Timed(multiText, null);
            Console.WriteLine($"  B[{i}] {ms,8:F0} ms  ({len / 32000.0:F2}s audio)");
            results.Add(("B_multi", ms, len));
        }

        Console.WriteLine($"\n== C. Emotion switching (change emotion on every call) x{nSwitch} ==");
        string[] cycle = { "happy", "angry", "sad", "surprise", "fearful", "playfulness" };
        for (int i = 0; i < nSwitch; i++)
        {
            var emo = cycle[i % cycle.Length];
            var (ms, len, _) = await Timed(singleText, emo);
            Console.WriteLine($"  C[{i}] {emo,-12} {ms,8:F0} ms  ({len / 32000.0:F2}s)");
            results.Add(("C_switch_" + emo, ms, len));
        }

        Console.WriteLine("\n==== summary (ms) ====");
        foreach (var grp in results.GroupBy(r => r.Case.Split('_')[0] + "_" + (r.Case.StartsWith("C") ? "switch" : r.Case.Split('_')[1])))
        {
            var list = grp.Select(x => x.Ms).OrderBy(x => x).ToList();
            double mean = list.Average();
            Console.WriteLine($"  {grp.Key,-14} n={list.Count}  mean {mean,7:F0}  median {list[list.Count / 2],7:F0}  min {list[0],7:F0}  max {list[^1],7:F0}");
        }

        // Phase timings: the overlap between single-sentence phase A and multi-sentence phase B is directly observable here
        var stats = PerfStats.Snapshot();
        if (PerfStats.Enabled)
        {
            Console.WriteLine("\n==== phase totals (ms) ====");
            Console.WriteLine($"  frontend  {stats.FrontendMs,9:F0}  calls={stats.FrontendCalls}");
            Console.WriteLine($"  t2s       {stats.T2sMs,9:F0}  calls={stats.T2sCalls}  phones/call={stats.T2sPhonesPerCall:F1}  stageSteps/call={stats.T2sStageStepsPerCall:F1}");
            Console.WriteLine($"  vocoder   {stats.VocMs,9:F0}  calls={stats.VocCalls}  samples/call={stats.VocSamplesPerCall:F0}");
        }
        return 0;
    }

    private static string? GetArg(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase)) return args[i + 1];
        return null;
    }
}
