using System.Diagnostics;
using SolTTS.Core.Config;
using SolTTS.Core.Core;
using SolTTS.Core.Frontend.BERT;
using Microsoft.ML.OnnxRuntime;

string resources = "resources";
int iters = 5;
for (int i = 0; i < args.Length - 1; i++)
{
    if (args[i] == "--resources") resources = args[i + 1];
    if (args[i] == "--iters") iters = int.Parse(args[i + 1]);
}

var config = TTSConfig.Load();
config.ResourcesDir = Path.GetFullPath(resources);
string model = config.EffectiveBertModelPath;
string tok = config.TokenizerJsonPath;
Console.WriteLine($"[bert] model={Path.GetFileName(model)}");

var text = "今天天气真好，我们一起去公园散步吧";
var w2p = new int[text.Length];
Array.Fill(w2p, 1);

double Bench(string label, Func<RobertaFeatureExtractor> factory)
{
    try
    {
        using var ex = factory();
        var warm = ex.Extract(text, w2p);
        var times = new List<double>();
        for (int i = 0; i < iters; i++)
        {
            var sw = Stopwatch.StartNew();
            ex.Extract(text, w2p);
            times.Add(sw.Elapsed.TotalMilliseconds);
        }
        times.Sort();
        double med = times[times.Count / 2];
        Console.WriteLine($"[bert] {label,-30} {med,7:F1} ms  (min {times[0]:F1})");
        return med;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[bert] {label,-30} FAILED: {ex.Message.Split('\n')[0]}");
        return double.NaN;
    }
}

var engine = new InferenceEngineV1();
double cpu = Bench("CPU baseline", () => new RobertaFeatureExtractor(model, tok, engine.GetSessionOptions(config), useCache: false));

// ggml full (adopted only if it claims the whole graph)
double full = Bench("ggml full (claim all)", () =>
{
    var so = GgmlExecutionProvider.TryCreateSessionOptions("gpu", 0, partial: false);
    if (so == null) throw new InvalidOperationException("ggml unavailable");
    return new RobertaFeatureExtractor(model, tok, so, useCache: false);
});

// ggml partial: offload 1067/1089 nodes to GPU, keep Gelu on CPU
double partial = Bench("ggml partial (Gelu on CPU)", () =>
{
    var so = GgmlExecutionProvider.TryCreateSessionOptions("gpu", 0, partial: true);
    if (so == null) throw new InvalidOperationException("ggml unavailable");
    return new RobertaFeatureExtractor(model, tok, so, useCache: false);
});

if (!double.IsNaN(cpu) && !double.IsNaN(partial))
    Console.WriteLine($"[bert] partial speedup = {cpu / partial:F2}x");
if (!double.IsNaN(cpu) && !double.IsNaN(full))
    Console.WriteLine($"[bert] full    speedup = {cpu / full:F2}x");
