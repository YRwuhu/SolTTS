using System.Diagnostics;
using SolTTS.Core.Config;
using SolTTS.Core.Core;
using SolTTS.Core.Frontend.BERT;

namespace SolTTS.BertBench;

/// <summary>
/// BERT backend comparison: CPU vs ggml/Vulkan.
/// With caching disabled, repeatedly runs the same input to measure cache-miss latency and compares outputs element by element.
/// </summary>
internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        string resources = GetArg(args, "--resources") ?? "resources";
        int iters = int.TryParse(GetArg(args, "--iters") ?? "5", out var it) ? it : 5;

        var config = TTSConfig.Load();
        config.ResourcesDir = Path.GetFullPath(resources);

        string model = config.EffectiveBertModelPath;
        string tok = config.TokenizerJsonPath;
        Console.WriteLine($"[bert] model={Path.GetFileName(model)} iters={iters}");

        var engine = new InferenceEngineV1();
        var cpuOpts = engine.GetSessionOptions(config);

        using var cpu = new RobertaFeatureExtractor(model, tok, cpuOpts, useCache: false);
        var text = "今天天气真好，我们一起去公园散步吧";
        var word2ph = new int[text.Length];
        Array.Fill(word2ph, 1);

        // Warm-up
        var cpuRef = cpu.Extract(text, word2ph);

        var cpuTimes = new List<double>();
        for (int i = 0; i < iters; i++)
        {
            var sw = Stopwatch.StartNew();
            cpu.Extract(text, word2ph);
            cpuTimes.Add(sw.Elapsed.TotalMilliseconds);
        }
        Console.WriteLine($"[bert] CPU      median {Median(cpuTimes):F1} ms  min {cpuTimes.Min():F1}");

        var gpuOpts = GgmlExecutionProvider.TryCreateSessionOptions(device: "gpu", gpuMinWeightMb: 0, partial: false);
        if (gpuOpts == null)
        {
            Console.WriteLine("[bert] ggml unavailable — CPU only");
            return 0;
        }

        RobertaFeatureExtractor? gpu = null;
        try
        {
            gpu = new RobertaFeatureExtractor(model, tok, gpuOpts, useCache: false);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[bert] ggml session failed: {ex.Message}");
            return 0;
        }
        finally
        {
            gpuOpts.Dispose();
        }

        using (gpu)
        {
            var gpuRef = gpu.Extract(text, word2ph);
            int rows = Math.Min(cpuRef.GetLength(0), gpuRef.GetLength(0));
            int cols = Math.Min(cpuRef.GetLength(1), gpuRef.GetLength(1));
            double dot = 0, na = 0, nb = 0, maxAbs = 0;
            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
            {
                double a = cpuRef[r, c], b = gpuRef[r, c];
                dot += a * b; na += a * a; nb += b * b;
                maxAbs = Math.Max(maxAbs, Math.Abs(a - b));
            }
            double cosine = dot / (Math.Sqrt(na) * Math.Sqrt(nb));
            Console.WriteLine($"[bert] shape cpu=[{cpuRef.GetLength(0)},{cpuRef.GetLength(1)}] gpu=[{gpuRef.GetLength(0)},{gpuRef.GetLength(1)}] " +
                              $"cosine={cosine:F6} maxAbs={maxAbs:E3}");

            var gpuTimes = new List<double>();
            for (int i = 0; i < iters; i++)
            {
                var sw = Stopwatch.StartNew();
                gpu.Extract(text, word2ph);
                gpuTimes.Add(sw.Elapsed.TotalMilliseconds);
            }
            Console.WriteLine($"[bert] ggml/VK  median {Median(gpuTimes):F1} ms  min {gpuTimes.Min():F1}  " +
                              $"speedup {Median(cpuTimes) / Median(gpuTimes):F2}x");
        }

        await Task.CompletedTask;
        return 0;
    }

    private static double Median(List<double> xs)
    {
        var s = xs.OrderBy(x => x).ToList();
        return s[s.Count / 2];
    }

    private static string? GetArg(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase)) return args[i + 1];
        return null;
    }
}
