using SolTTS.Core.Config;
using SolTTS.Core.Core;

// End-to-end smoke check: non-streaming multi-sentence + streaming chunks, confirming both paths produce non-empty audio.
// Used for quick regression of issues like "front-end prefetch results discarded, yielding 0-duration audio".
// Usage: dotnet run --project tools/stream_check -c Release [-- --resources resources]

string resources = "resources";
for (int i = 0; i < args.Length - 1; i++)
    if (string.Equals(args[i], "--resources", StringComparison.OrdinalIgnoreCase)) resources = args[i + 1];

var config = TTSConfig.Load();
config.ResourcesDir = Path.GetFullPath(resources);

using var sdk = new SolTtsSdk(config);
await sdk.InitializeAsync();

const string multiText = "今天天气真好。我们一起去公园散步吧。晚上吃点什么好呢。";
const string streamText = "今天天气真好。我们一起去公园散步吧。";

var full = await sdk.PredictAsync(multiText, new TtsOptions(), config.DefaultAvatarId, null);
Console.WriteLine($"[check] non-streaming samples={full.Audio.Length} tokens={full.TokenCount} " +
                  $"duration={full.Audio.Length / 32000.0:F2}s");

int chunks = 0;
long samples = 0;
await foreach (var c in sdk.PredictStreamAsync(streamText, new TtsOptions(), config.DefaultAvatarId, null))
{
    chunks++;
    samples += c.Length;
}
Console.WriteLine($"[check] streaming chunks={chunks} samples={samples} duration={samples / 32000.0:F2}s");
Console.WriteLine($"[check] native audio={(NativeAudio.IsAvailable ? "available" : "unavailable (managed fallback)")}, " +
                  $"useNativeAudio={config.UseNativeAudio}, parallelSentences={config.ParallelSentenceDegree}");

// 3-sentence text should produce at least 1 second of audio; streaming should have at least 1 chunk
bool ok = full.Audio.Length > 32000 && full.TokenCount > 0 && chunks > 0 && samples > 32000;
Console.WriteLine(ok ? "[check] PASS" : "[check] FAIL");
return ok ? 0 : 1;
