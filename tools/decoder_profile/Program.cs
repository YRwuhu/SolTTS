using System.Diagnostics;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SolTTS.Core.Config;
using SolTTS.Core.Core;

// ORT kernel-level profile of one decoder step: where does the time go?
//
//   dotnet run --project tools/decoder_profile2 -c Release -- [--model P] [--kv S] [--iy L]
string model = "resources/models_v1/chenxing/t2s_stage_decoder_qkv.onnx";
int kv = 96, iy = 2, steps = 20, warmup = 5;
string kvLayout = "auto";
for (int i = 0; i < args.Length - 1; i++)
{
    switch (args[i])
    {
        case "--model": model = args[i + 1]; break;
        case "--kv": kv = int.Parse(args[i + 1]); break;
        case "--iy": iy = int.Parse(args[i + 1]); break;
        case "--steps": steps = int.Parse(args[i + 1]); break;
        case "--kv-layout": kvLayout = args[i + 1]; break;
    }
}

const int Layers = 24, Dim = 512;
var rng = new Random(1234);
float[] Noise(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * 0.2); return a; }

var inputs = new List<NamedOnnxValue>(50)
{
    NamedOnnxValue.CreateFromTensor("iy", new DenseTensor<long>(Enumerable.Range(0, iy).Select(i => (long)(i % 1025)).ToArray(), new[] { 1, iy })),
    NamedOnnxValue.CreateFromTensor("iy_emb", new DenseTensor<float>(Noise((iy - 1) * Dim), new[] { 1, iy - 1, Dim })),
};
for (int l = 0; l < Layers; l++)
{
    // Native-KV models take K=[1,16,32,S] / V=[1,16,S,32]; the shipped graphs
    // take [S,1,512]. Infer the layout from the file name unless told otherwise.
    bool native = kvLayout == "native" || (kvLayout == "auto" && model.Contains("kvlayout"));
    if (native)
    {
        inputs.Add(NamedOnnxValue.CreateFromTensor($"past_k_layer_{l}", new DenseTensor<float>(Noise(kv * Dim), new[] { 1, 16, 32, kv })));
        inputs.Add(NamedOnnxValue.CreateFromTensor($"past_v_layer_{l}", new DenseTensor<float>(Noise(kv * Dim), new[] { 1, 16, kv, 32 })));
    }
    else
    {
        inputs.Add(NamedOnnxValue.CreateFromTensor($"past_k_layer_{l}", new DenseTensor<float>(Noise(kv * Dim), new[] { kv, 1, Dim })));
        inputs.Add(NamedOnnxValue.CreateFromTensor($"past_v_layer_{l}", new DenseTensor<float>(Noise(kv * Dim), new[] { kv, 1, Dim })));
    }
}

var cfg = new TTSConfig();
var so = new InferenceEngineV1().GetSessionOptions(cfg);
string traceDir = Path.Combine(Path.GetTempPath(), "solt_ort_profile");
Directory.CreateDirectory(traceDir);
foreach (var f in Directory.GetFiles(traceDir)) File.Delete(f);
so.EnableProfiling = true;
so.ProfileOutputPathPrefix = traceDir;

using var session = new InferenceSession(model, so);
// Warmup runs are profiled too (profiling is a session option), so the events
// below cover warmup + steps runs; divide by that, not by steps.
for (int i = 0; i < warmup; i++) using (session.Run(inputs)) { }
var sw = Stopwatch.StartNew();
for (int i = 0; i < steps; i++) using (session.Run(inputs)) { }
sw.Stop();
int profiledRuns = warmup + steps;
Console.WriteLine($"[prof] {Path.GetFileName(model)} kv={kv} iy={iy}: {(sw.Elapsed.TotalMilliseconds / steps):F2} ms/step over {steps} steps");

string? trace = session.EndProfiling(); Console.WriteLine($"[prof] EndProfiling -> {trace}");
Console.WriteLine($"[prof] prefix={so.ProfileOutputPathPrefix} model={so.ProfileOutputPathPrefix}_*.json");
var file = trace ?? Directory.GetFiles(traceDir, "*.json").OrderByDescending(File.GetLastWriteTimeUtc).FirstOrDefault();
if (file == null) { Console.WriteLine("[prof] no trace written"); return; }

using var doc = JsonDocument.Parse(File.ReadAllText(file));
var byOp = new Dictionary<string, (double us, int n)>();
foreach (var e in doc.RootElement.EnumerateArray())
{
    if (!e.TryGetProperty("cat", out var cat) || cat.GetString() != "Node") continue;
    if (!e.TryGetProperty("args", out var a) || !a.TryGetProperty("op_name", out var op)) continue;
    double dur = e.GetProperty("dur").GetDouble();
    // Skip the session-level aggregate entries.
    string name = op.GetString() ?? "?";
    var cur = byOp.TryGetValue(name, out var v) ? v : (0, 0);
    byOp[name] = (cur.Item1 + dur, cur.Item2 + 1);
}
double total = byOp.Values.Sum(v => v.us);
Console.WriteLine($"[prof] {byOp.Values.Sum(v => v.n)} kernel events over {profiledRuns} runs, {total / 1000 / profiledRuns:F1} ms/step attributed");
foreach (var (op, v) in byOp.OrderByDescending(kv => kv.Value.us).Take(18))
    Console.WriteLine($"  {op,-28} {v.us / 1000 / profiledRuns,8:F2} ms/step  {100 * v.us / total,5:F1}%  x{v.n / profiledRuns}");
