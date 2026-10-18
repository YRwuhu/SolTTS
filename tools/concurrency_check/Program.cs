using SolTTS.Core.Config;
using SolTTS.Core.Core;

// Concurrency stress check: multiple requests enter the engine pool simultaneously, confirming no deadlock, empty audio, or exceptions.
// Usage: dotnet run --project tools/concurrency_check -c Release [-- --requests 8 --concurrency 3]

int requests = 8, concurrency = 3;
string resources = "resources";
for (int i = 0; i < args.Length - 1; i++)
{
    if (string.Equals(args[i], "--requests", StringComparison.OrdinalIgnoreCase)) requests = int.Parse(args[i + 1]);
    if (string.Equals(args[i], "--concurrency", StringComparison.OrdinalIgnoreCase)) concurrency = int.Parse(args[i + 1]);
    if (string.Equals(args[i], "--resources", StringComparison.OrdinalIgnoreCase)) resources = args[i + 1];
}

var config = TTSConfig.Load();
config.ResourcesDir = Path.GetFullPath(resources);
config.PoolCapacity = concurrency;
config.ReuseMemory = false; // Concurrency requires independent sessions

Console.WriteLine($"[conc] requests={requests} concurrency={concurrency} psent={config.ParallelSentenceDegree}");

using var sdk = new SolTtsSdk(config);
await sdk.InitializeAsync();

string[] texts =
{
    "今天天气真好。我们一起去公园散步吧。",
    "晚上吃点什么好呢？要不我们去那家新开的面馆。",
    "良宵方始，不必心急。",
};

var failures = 0;
var sw = System.Diagnostics.Stopwatch.StartNew();
var tasks = Enumerable.Range(0, requests).Select(async i =>
{
    var text = texts[i % texts.Length];
    var r = await sdk.PredictAsync(text, new TtsOptions(), config.DefaultAvatarId, null);
    bool ok = r.Audio.Length > 16000 && r.TokenCount > 0;
    if (!ok) Interlocked.Increment(ref failures);
    Console.WriteLine($"[conc] req {i}: samples={r.Audio.Length} tokens={r.TokenCount} {(ok ? "OK" : "EMPTY")}");
    return ok;
}).ToArray();

try
{
    await Task.WhenAll(tasks);
}
catch (Exception ex)
{
    Console.WriteLine($"[conc] EXCEPTION: {ex.GetType().Name}: {ex.Message}");
    return 1;
}
sw.Stop();

Console.WriteLine($"[conc] elapsed={sw.ElapsedMilliseconds} ms failures={failures}");
Console.WriteLine(failures == 0 ? "[conc] PASS" : "[conc] FAIL");
return failures == 0 ? 0 : 1;
