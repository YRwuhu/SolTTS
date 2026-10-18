using SolTTS.Core.Config;
using SolTTS.Core.Core;

namespace SolTTS.EmotionInvariant;

/// <summary>
/// Verifies the α=0 invariant: when emotion intensity is 0, or neutral/not enabled, the injected speaker embedding
/// must be bit-identical to the original embedding (max|Δ| = 0), and prompt_encoder output must remain consistent.
///
/// This is the safety boundary of emotion optimization: the lightweight recompute path (ApplyEmotionOnly) must not change the output of emotion-free requests.
/// </summary>
internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        string resources = GetArg(args, "--resources") ?? "resources";

        var config = TTSConfig.Load();
        config.ResourcesDir = Path.GetFullPath(resources);
        config.UseEmotionControl = true;

        using var sdk = new SolTtsSdk(config);
        await sdk.InitializeAsync();

        var emotion = new EmotionDirectionController(config.EffectiveEmotionVectorPath);
        if (!emotion.Loaded)
        {
            Console.WriteLine("[invariant] emotion vectors unavailable — skipped");
            return 0;
        }

        bool ok = true;

        // 1. Apply(x, tau, 0) must return x bit-for-bit
        var rng = new Random(2024);
        var xBase = new float[emotion.Dim];
        for (int i = 0; i < xBase.Length; i++) xBase[i] = (float)(rng.NextDouble() * 2 - 1);

        foreach (var name in emotion.Emotions)
        {
            var tau = emotion.GetTau(name);
            if (tau == null) continue;

            var zero = EmotionDirectionController.Apply(xBase, tau, 0f);
            double maxAbs = 0;
            for (int i = 0; i < xBase.Length; i++)
            {
                if (BitConverter.SingleToInt32Bits(xBase[i]) != BitConverter.SingleToInt32Bits(zero[i]))
                    maxAbs = Math.Max(maxAbs, Math.Abs(xBase[i] - zero[i]));
            }

            if (maxAbs != 0)
            {
                Console.WriteLine($"[invariant] FAIL α=0 changed embedding for '{name}': max|Δ|={maxAbs:E3}");
                ok = false;
            }
            else
            {
                Console.WriteLine($"[invariant] α=0 bit-exact for '{name}'");
            }
        }

        // 2. End-to-end: neutral and α=0 emotion requests should produce the same token-count distribution (structurally consistent)
        string text = "今天天气真好。";
        var neutral = await sdk.PredictAsync(text, new TtsOptions { EmotionId = null }, config.DefaultAvatarId, null);
        var omega = await sdk.PredictAsync(text, new TtsOptions { EmotionId = "happy", EmotionAlpha = 0f }, config.DefaultAvatarId, null);

        if (neutral.Audio.Length == 0 || omega.Audio.Length == 0)
        {
            Console.WriteLine("[invariant] FAIL empty audio");
            ok = false;
        }
        else
        {
            Console.WriteLine($"[invariant] neutral audio={neutral.Audio.Length} samples, tokens={neutral.TokenCount}");
            Console.WriteLine($"[invariant] α=0     audio={omega.Audio.Length} samples, tokens={omega.TokenCount}");
            // Sampling is random, so lengths will not be identical; only require the same order of magnitude (no anomalies caused by the α=0 path)
            double ratio = (double)omega.Audio.Length / neutral.Audio.Length;
            if (ratio < 0.6 || ratio > 1.6)
            {
                Console.WriteLine($"[invariant] FAIL α=0 audio length ratio {ratio:F2} out of range");
                ok = false;
            }
        }

        Console.WriteLine(ok ? "[invariant] PASS" : "[invariant] FAILED");
        return ok ? 0 : 1;
    }

    private static string? GetArg(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase)) return args[i + 1];
        return null;
    }
}
