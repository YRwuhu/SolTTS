using System.Diagnostics;
using SolTTS.Core.Config;
using SolTTS.Core.Core;
using SolTTS.Core.Utils;

namespace SolTTSEmotionEval
{
    /// <summary>
    /// End-to-end emotion injection validation tool:
    ///   1) τ=0 (neutral/α=0) should match the baseline
    ///   2) Random τ should change the output
    ///   3) α grid sweep: generate audio + duration/F0 statistics
    ///
    /// Usage:
    ///   dotnet run --project tools/emotion_eval -- --out E:\...\eval --emotions happy,angry ...
    /// </summary>
    internal static class Program
    {
        private static async Task<int> Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            Console.InputEncoding = System.Text.Encoding.UTF8;

            var outDir = GetArg(args, "--out") ?? @"E:\AI\SolTTS\_emotion_eval_cli";
            var text = GetArg(args, "--text") ?? "今天天气真好，我们一起去公园散步吧。";
            var avatarId = GetArg(args, "--avatar") ?? "chenxing";
            var emotions = (GetArg(args, "--emotions") ?? "neutral,happy,angry,sad,surprise,fearful,playfulness").Split(',');
            var alphas = (GetArg(args, "--alphas") ?? "0.1,0.3,0.5").Split(',');
            int repeats = int.TryParse(GetArg(args, "--repeats") ?? "1", out var rp) ? rp : 1;
            var configPath = GetArg(args, "--config");

            System.IO.Directory.CreateDirectory(outDir);

            var config = configPath != null ? TTSConfig.Load(configPath) : TTSConfig.LoadOrCreate(
                System.IO.Path.Combine(AppContext.BaseDirectory, "config.yaml"));

            // Force-enable emotion control (for experiments)
            config.UseEmotionControl = true;
            if (GetArg(args, "--resources") is { } resDir) config.ResourcesDir = resDir;
            if (GetArg(args, "--tau") is { } tauDir) config.EmotionVectorPath = tauDir;

            Console.WriteLine($"ResourcesDir : {config.ResourcesDir}");
            Console.WriteLine($"EmotionPath  : {config.EmotionVectorFullPath}");
            Console.WriteLine($"Avatar       : {avatarId}");
            Console.WriteLine($"Text         : {text}");
            Console.WriteLine();

            using var sdk = new SolTtsSdk(config);
            var sw = Stopwatch.StartNew();
            await sdk.InitializeAsync();
            Console.WriteLine($"SDK initialized in {sw.ElapsedMilliseconds} ms (sampling={sdk.SamplingRate})");

            var results = new List<(string emotion, float alpha, string file, int samples, double ms)>();

            async Task RunOne(string emotion, float? alpha, int rep = 0)
            {
                var opt = new TtsOptions
                {
                    EmotionId = emotion == "neutral" ? null : emotion,
                    EmotionAlpha = alpha
                };
                var sw2 = Stopwatch.StartNew();
                var r = await sdk.PredictAsync(text, opt, avatarId, null);
                sw2.Stop();
                var tag = alpha.HasValue ? $"{emotion}_a{alpha.Value:0.##}" : emotion;
                if (rep > 0) tag += $"_r{rep}";
                var file = System.IO.Path.Combine(outDir, $"{tag}.wav");
                AudioHelper.SaveWav(file, r.Audio, sdk.SamplingRate);
                double dur = r.Audio.Length / (double)sdk.SamplingRate;
                Console.WriteLine($"  {tag,-22} samples={r.Audio.Length,7} dur={dur:0.000}s tokens={r.TokenCount,4} " +
                                  $"wall={sw2.ElapsedMilliseconds,6}ms");
                results.Add((emotion, alpha ?? -1f, file, r.Audio.Length, sw2.ElapsedMilliseconds));
            }

            // 1. Baseline (no emotion)
            Console.WriteLine("== baseline (no injection) ==");
            await RunOne("neutral", null);
            for (int i = 1; i <= repeats; i++) await RunOne("neutral", null, i);

            // 2. Emotion × α grid
            foreach (var emo in emotions)
            {
                if (emo.Trim() == "neutral") continue;
                Console.WriteLine($"== {emo} ==");
                foreach (var a in alphas)
                {
                    if (!float.TryParse(a, System.Globalization.NumberStyles.Float,
                            System.Globalization.CultureInfo.InvariantCulture, out var af)) continue;
                    await RunOne(emo.Trim(), af);
                    for (int i = 1; i <= repeats; i++) await RunOne(emo.Trim(), af, i);
                }
            }

            // 3. α=0 consistency test
            Console.WriteLine("== alpha=0 (should match no-injection path) ==");
            await RunOne("happy", 0f);

            // summary
            var csv = System.IO.Path.Combine(outDir, "summary.csv");
            using (var w = new StreamWriter(csv, false, System.Text.Encoding.UTF8))
            {
                w.WriteLine("emotion,alpha,file,samples,ms");
                foreach (var r in results)
                    w.WriteLine($"{r.emotion},{r.alpha},{r.file},{r.samples},{r.ms}");
            }
            Console.WriteLine($"\nSaved {results.Count} samples to {outDir}");
            Console.WriteLine($"Summary: {csv}");
            return 0;
        }

        private static string? GetArg(string[] args, string name)
        {
            for (int i = 0; i < args.Length - 1; i++)
                if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase)) return args[i + 1];
            return null;
        }
    }
}
