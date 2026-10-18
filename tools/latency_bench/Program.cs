using System.Diagnostics;
using SolTTS.Core.Config;
using SolTTS.Core.Core;

string resources = "resources";
int n = 5;
float factor = -1;
for (int i = 0; i < args.Length - 1; i++)
{
    if (args[i] == "--resources") resources = args[i + 1];
    if (args[i] == "--n") n = int.Parse(args[i + 1]);
    if (args[i] == "--vf") factor = float.Parse(args[i + 1]);
}

var config = TTSConfig.Load();
config.ResourcesDir = Path.GetFullPath(resources);
config.PoolCapacity = 1;
PerfStats.Enabled = true;
if (factor > 0) config.VocoderRetriggerFactor = factor;

using var sdk = new SolTtsSdk(config);
await sdk.InitializeAsync();
Console.WriteLine($"[lat] psent={config.ParallelSentenceDegree} vf={config.VocoderRetriggerFactor} chunk={config.StreamingChunkSize}");

const string single = "今天天气真好，我们一起去公园散步吧。";
const string multi = "今天天气真好。我们一起去公园散步吧。晚上吃点什么好呢？";
static string Med(List<double> v) { var s=v.OrderBy(x=>x).ToList(); return $"{s[s.Count/2]:F0}"; }

await sdk.PredictAsync("预热一下。", new TtsOptions(), config.DefaultAvatarId, null);
await foreach (var _ in sdk.PredictStreamAsync("预热一下。", new TtsOptions(), config.DefaultAvatarId, null)) { }
PerfStats.Reset();

async Task<(double ttfb,double total,int chunks,long samples)> Stream(string text)
{
    int chunks=0; long samples=0; double ttfb=0;
    var sw=Stopwatch.StartNew();
    await foreach (var c in sdk.PredictStreamAsync(text, new TtsOptions { VocoderRetriggerFactor = factor > 0 ? factor : config.VocoderRetriggerFactor }, config.DefaultAvatarId, null))
    {
        if(chunks==0) ttfb=sw.Elapsed.TotalMilliseconds;
        chunks++; samples+=c.Length;
    }
    sw.Stop();
    return (ttfb, sw.Elapsed.TotalMilliseconds, chunks, samples);
}

var t1=new List<double>(); var d1=new List<double>();
for(int i=0;i<n;i++){var r=await Stream(single);t1.Add(r.ttfb);d1.Add(r.total);}
Console.WriteLine($"[lat] stream single  TTFB {Med(t1)} ms | total {Med(d1)} ms");

var t3=new List<double>(); var d3=new List<double>(); long smp=0;
for(int i=0;i<n;i++){var r=await Stream(multi);t3.Add(r.ttfb);d3.Add(r.total);smp=r.samples;}
Console.WriteLine($"[lat] stream multi   TTFB {Med(t3)} ms | total {Med(d3)} ms ({smp/32000.0:F2}s)");

var ps=PerfStats.Snapshot();
Console.WriteLine($"[lat] t2s={ps.T2sMs:F0}ms vocoder={ps.VocMs:F0}ms/{ps.VocCalls}calls");
