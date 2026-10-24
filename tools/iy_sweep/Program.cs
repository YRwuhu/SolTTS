using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// How much does a decode step cost as `iy` grows?
//
// This is the make-or-break number for any "several tokens per forward" scheme:
// the transformer reads its weights once per Run regardless of how many query
// positions it is given, so if cost is flat in `iy`, one forward can verify
// several drafted tokens for the price of one.
//
//   dotnet run --project tools/iy_sweep -c Release -- [--kv S] [--steps N]
string model = "resources/models_v1/chenxing/t2s_stage_decoder_kvlayout.onnx";
int kv = 96, steps = 30;
bool native = model.Contains("kvlayout") || model.Contains("fullseq");
for (int i = 0; i < args.Length - 1; i++)
{
    if (args[i] == "--model") { model = args[i + 1]; native = model.Contains("kvlayout") || model.Contains("fullseq"); }
    if (args[i] == "--kv") kv = int.Parse(args[i + 1]);
    if (args[i] == "--steps") steps = int.Parse(args[i + 1]);
}

const int Layers = 24, Dim = 512, Heads = 16, HeadDim = 32, Vocab = 1025;
var rng = new Random(5);
float[] Noise(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * 0.2); return a; }

List<NamedOnnxValue> Feed(int iy)
{
    var l = new List<NamedOnnxValue>(50)
    {
        NamedOnnxValue.CreateFromTensor("iy", new DenseTensor<long>(Enumerable.Range(0, iy).Select(i => (long)(i % Vocab)).ToArray(), new[] { 1, iy })),
        NamedOnnxValue.CreateFromTensor("iy_emb", new DenseTensor<float>(Noise((iy - 1) * Dim), new[] { 1, iy - 1, Dim })),
    };
    for (int layer = 0; layer < Layers; layer++)
    {
        int n = kv * Dim;
        if (native)
        {
            l.Add(NamedOnnxValue.CreateFromTensor($"past_k_layer_{layer}", new DenseTensor<float>(Noise(n), new[] { 1, Heads, HeadDim, kv })));
            l.Add(NamedOnnxValue.CreateFromTensor($"past_v_layer_{layer}", new DenseTensor<float>(Noise(n), new[] { 1, Heads, kv, HeadDim })));
        }
        else
        {
            l.Add(NamedOnnxValue.CreateFromTensor($"past_k_layer_{layer}", new DenseTensor<float>(Noise(n), new[] { kv, 1, Dim })));
            l.Add(NamedOnnxValue.CreateFromTensor($"past_v_layer_{layer}", new DenseTensor<float>(Noise(n), new[] { kv, 1, Dim })));
        }
    }
    return l;
}

var so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL, ExecutionMode = ExecutionMode.ORT_SEQUENTIAL };
using var s = new InferenceSession(model, so);
Console.WriteLine($"[iy] {Path.GetFileName(model)} kv={kv} steps={steps}");
double baseline = 0;
// iy=1 is rejected by the graph: the sampler Squeezes it to a scalar and
// GatherElements requires rank 1. Two positions is the minimum.
foreach (int iy in new[] { 2, 4, 8, 16, 32 })
{
    var feed = Feed(iy);
    for (int i = 0; i < 8; i++) using (s.Run(feed)) { }
    var t = new List<double>(steps);
    for (int i = 0; i < steps; i++) { var sw = Stopwatch.StartNew(); using var _ = s.Run(feed); t.Add(sw.Elapsed.TotalMilliseconds); }
    t.Sort();
    double p50 = t[t.Count / 2];
    if (iy == 2) baseline = p50;
    Console.WriteLine($"  iy={iy,3}  p50 {p50,7:F2} ms   vs iy=2: {p50 / baseline,5:F2}x");
}
