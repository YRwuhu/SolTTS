using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// DirectML is ONNX Runtime's first-class GPU provider: no graph rebuild per Run,
// no host-only memory model, and it supports the INT8 ops the decoder uses.
// The ggml plugin was measured at 2-3x slower than CPU, but that verdict is about
// *that plugin*, not about "the decoder on a GPU". This probe separates the two.
//
//   dotnet run --project tools/dml_probe -c Release -- [--model P] [--steps N]
string model = "resources/models_v1/chenxing/t2s_stage_decoder_kvlayout.onnx";
int steps = 25;
for (int i = 0; i < args.Length - 1; i++)
{
    if (args[i] == "--model") model = args[i + 1];
    if (args[i] == "--steps") steps = int.Parse(args[i + 1]);
}
const int Layers = 24, Dim = 512, Heads = 16, HeadDim = 32, Vocab = 1025;
bool native = model.Contains("kvlayout") || model.Contains("kv_fp32");

Console.WriteLine($"[dml] ORT {typeof(InferenceSession).Assembly.GetName().Version}");
foreach (var p in OrtEnv.Instance().GetAvailableProvidersSafe()) Console.WriteLine($"[dml] provider: {p}");
Console.WriteLine($"[dml] model={Path.GetFileName(model)} nativeKV={native}");

var rng = new Random(7);
float[] Noise(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * 0.2); return a; }

List<NamedOnnxValue> Feed(int iy, int kv)
{
    var l = new List<NamedOnnxValue>(50)
    {
        NamedOnnxValue.CreateFromTensor("iy", new DenseTensor<long>(Enumerable.Range(0, iy).Select(i => (long)(i % Vocab)).ToArray(), new[] { 1, iy })),
        NamedOnnxValue.CreateFromTensor("iy_emb", new DenseTensor<float>(Noise((iy - 1) * Dim), new[] { 1, iy - 1, Dim })),
    };
    for (int i = 0; i < Layers; i++)
    {
        int n = kv * Dim;
        if (native)
        {
            l.Add(NamedOnnxValue.CreateFromTensor($"past_k_layer_{i}", new DenseTensor<float>(Noise(n), new[] { 1, Heads, HeadDim, kv })));
            l.Add(NamedOnnxValue.CreateFromTensor($"past_v_layer_{i}", new DenseTensor<float>(Noise(n), new[] { 1, Heads, kv, HeadDim })));
        }
        else
        {
            l.Add(NamedOnnxValue.CreateFromTensor($"past_k_layer_{i}", new DenseTensor<float>(Noise(n), new[] { kv, 1, Dim })));
            l.Add(NamedOnnxValue.CreateFromTensor($"past_v_layer_{i}", new DenseTensor<float>(Noise(n), new[] { kv, 1, Dim })));
        }
    }
    return l;
}

void Bench(string label, SessionOptions so)
{
    try
    {
        var swl = Stopwatch.StartNew();
        using var s = new InferenceSession(model, so);
        swl.Stop();
        Console.WriteLine($"\n=== {label} (load {swl.ElapsedMilliseconds} ms) ===");
        foreach (var (iy, kv) in new[] { (2, 1), (2, 32), (2, 96), (2, 160), (2, 320) })
        {
            var f = Feed(iy, kv);
            for (int i = 0; i < 8; i++) using (s.Run(f)) { }
            var t = new List<double>(steps);
            for (int i = 0; i < steps; i++) { var sw = Stopwatch.StartNew(); using var _ = s.Run(f); t.Add(sw.Elapsed.TotalMilliseconds); }
            t.Sort();
            Console.WriteLine($"  iy={iy,3} kv={kv,3}  p50 {t[t.Count / 2],7:F2} ms   p95 {t[(int)(t.Count * 0.95)],7:F2}");
        }
    }
    catch (Exception ex) { Console.WriteLine($"\n=== {label} FAILED: {ex.Message.Split('\n')[0]}"); }
}

Bench("CPU", new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL });

try
{
    var dml = new SessionOptions
    {
        GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        EnableMemoryPattern = false,
    };
    dml.AppendExecutionProvider_DML(0);
    Bench("DirectML GPU", dml);
}
catch (Exception ex) { Console.WriteLine($"[dml] DirectML unavailable: {ex.Message.Split('\n')[0]}"); }

static class OrtEnvExt
{
    public static string[] GetAvailableProvidersSafe(this OrtEnv env) => OrtEnv.Instance().GetAvailableProviders().ToArray();
}
