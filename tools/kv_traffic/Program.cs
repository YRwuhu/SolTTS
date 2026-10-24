using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// How much of a decode step is KV cache traffic? Run the same graph with a
// collapsed KV (length 1) and with realistic lengths; the difference is the
// per-step cost of reading past_* and writing present_*, which grows linearly
// with the sentence. That number sets the ceiling for any "resident KV" work.
//
//   dotnet run --project tools/kv_traffic -c Release -- [--model P] [--steps N]
string model = "resources/models_v1/chenxing/t2s_stage_decoder_kvlayout.onnx";
int steps = 40;
for (int i = 0; i < args.Length - 1; i++)
{
    if (args[i] == "--model") model = args[i + 1];
    if (args[i] == "--steps") steps = int.Parse(args[i + 1]);
}
const int Layers = 24, Dim = 512, Heads = 16, HeadDim = 32, Vocab = 1025;
var so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL, ExecutionMode = ExecutionMode.ORT_SEQUENTIAL };
using var s = new InferenceSession(model, so);
var rng = new Random(3);
float[] Noise(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * 0.2); return a; }

List<NamedOnnxValue> Feed(int kv)
{
    var l = new List<NamedOnnxValue>(50)
    {
        NamedOnnxValue.CreateFromTensor("iy", new DenseTensor<long>(new long[] { 7, 42 }, new[] { 1, 2 })),
        NamedOnnxValue.CreateFromTensor("iy_emb", new DenseTensor<float>(Noise(Dim), new[] { 1, 1, Dim })),
    };
    for (int i = 0; i < Layers; i++)
    {
        l.Add(NamedOnnxValue.CreateFromTensor($"past_k_layer_{i}", new DenseTensor<float>(Noise(kv * Dim), new[] { 1, Heads, HeadDim, kv })));
        l.Add(NamedOnnxValue.CreateFromTensor($"past_v_layer_{i}", new DenseTensor<float>(Noise(kv * Dim), new[] { 1, Heads, kv, HeadDim })));
    }
    return l;
}

double Time(int kv)
{
    var f = Feed(kv);
    for (int i = 0; i < 8; i++) using (s.Run(f)) { }
    var t = new List<double>(steps);
    for (int i = 0; i < steps; i++) { var sw = Stopwatch.StartNew(); using var _ = s.Run(f); t.Add(sw.Elapsed.TotalMilliseconds); }
    t.Sort();
    return t[t.Count / 2];
}

Console.WriteLine($"[kv] {Path.GetFileName(model)}");
double baseMs = Time(1);
// One step moves each layer's K and V twice: in as past_*, out as present_*.
double BytesPerToken(int kv) => (double)kv * Dim * 4 * Layers * 2 * 2;
Console.WriteLine($"  kv=  1  p50 {baseMs,6:F2} ms   (KV traffic ~{BytesPerToken(1) / 1048576:F2} MiB)");
foreach (int kv in new[] { 32, 96, 160, 320 })
{
    double ms = Time(kv);
    double mib = BytesPerToken(kv) / 1048576;
    double delta = ms - baseMs;
    string bw = delta <= 0.05 ? "below noise" : $"{(mib / delta):F1} MiB/ms ({mib / delta / 1024:F2} GB/s)";
    Console.WriteLine($"  kv={kv,3}  p50 {ms,6:F2} ms   delta {delta,5:F2} ms   KV ~{mib,6:F1} MiB   {bw}");
}
