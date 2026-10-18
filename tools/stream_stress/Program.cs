using SolTTS.Core.Config;
using SolTTS.Core.Core;

// Non-streaming degradation check: mult=6 forces "insufficient generation" to confirm it no longer throws and audio is normal.
string resources = "resources";
int reps = 2;
float mult = 0;
for (int i = 0; i < args.Length - 1; i++)
{
    if (args[i] == "--resources") resources = args[i + 1];
    if (args[i] == "--reps") reps = int.Parse(args[i + 1]);
    if (args[i] == "--mult") mult = float.Parse(args[i + 1]);
}

var config = TTSConfig.Load();
config.ResourcesDir = Path.GetFullPath(resources);
config.PoolCapacity = 1;
if (mult > 0) config.Inference.MinTokenMultiplierChinese = mult;

using var sdk = new SolTtsSdk(config);
await sdk.InitializeAsync();
Console.WriteLine($"[batch] mult={config.Inference.MinTokenMultiplierChinese}");

string[] cases =
{
    "今天天气真好。",
    "今天天气真好。我们一起去公园散步吧。晚上吃点什么好呢？",
    "人工智能正在改变我们的生活方式，从语音识别到自然语言处理，技术进步的速度令人惊叹。",
};

int total = 0, failed = 0;
foreach (var text in cases)
for (int r = 0; r < reps; r++)
{
    total++;
    try
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var res = await sdk.PredictAsync(text, new TtsOptions(), "chenxing", null);
        sw.Stop();
        if (res.Audio.Length < 8000) { failed++; Console.WriteLine($"[batch] SHORT samples={res.Audio.Length}"); }
        else Console.WriteLine($"[batch] ok {sw.ElapsedMilliseconds,5}ms {res.Audio.Length / 32000.0:F2}s tokens={res.TokenCount} : {text[..Math.Min(18, text.Length)]}");
    }
    catch (Exception ex) { failed++; Console.WriteLine($"[batch] FAIL {ex.Message}"); }
}
Console.WriteLine($"[batch] total={total} failed={failed}");
return failed == 0 ? 0 : 1;
