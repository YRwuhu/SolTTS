using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// Interleaved A/B of two decoder graphs that take the *same* KV layout.
//
// The original use is flat ([S,1,512]) vs native ([1,16,32,S]) — different
// inputs, so the layout is declared per side. Both models are measured in the
// same process in alternating rounds, so machine noise hits both equally.
//
//   dotnet run --project tools/kv_layout_ab -c Release -- \
//       [--a modelA] [--b modelB] [--layout-a flat|native] [--layout-b flat|native]
string aPath = "resources/models_v1/chenxing/t2s_stage_decoder_qkv.onnx";
string bPath = "resources/models_v1/chenxing/t2s_stage_decoder_kvlayout.onnx";
string aLayout = "flat", bLayout = "native";
int steps = 30, rounds = 7;
for (int i = 0; i < args.Length - 1; i++)
{
    switch (args[i])
    {
        case "--steps": steps = int.Parse(args[i + 1]); break;
        case "--rounds": rounds = int.Parse(args[i + 1]); break;
        case "--a": aPath = args[i + 1]; break;
        case "--b": bPath = args[i + 1]; break;
        case "--layout-a": aLayout = args[i + 1]; break;
        case "--layout-b": bLayout = args[i + 1]; break;
        // Kept for the earlier flag names.
        case "--flat": aPath = args[i + 1]; aLayout = "flat"; break;
        case "--native": bPath = args[i + 1]; bLayout = "native"; break;
    }
}
foreach (var p in new[] { aPath, bPath })
    if (!File.Exists(p)) { Console.WriteLine($"[ab] missing {p}"); return; }

const int Layers = 24, Dim = 512, Heads = 16, HeadDim = 32, Vocab = 1025;
var so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL, ExecutionMode = ExecutionMode.ORT_SEQUENTIAL };
using var sa = new InferenceSession(aPath, so);
using var sb = new InferenceSession(bPath, so);

var rng = new Random(99);
float[] Noise(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * 0.2); return a; }

List<NamedOnnxValue> Feed(bool nativeLayout, int iyLen, int kvLen)
{
    var l = new List<NamedOnnxValue>(50)
    {
        NamedOnnxValue.CreateFromTensor("iy", new DenseTensor<long>(Enumerable.Range(0, iyLen).Select(i => (long)(i % Vocab)).ToArray(), new[] { 1, iyLen })),
        NamedOnnxValue.CreateFromTensor("iy_emb", new DenseTensor<float>(Noise((iyLen - 1) * Dim), new[] { 1, iyLen - 1, Dim })),
    };
    for (int layer = 0; layer < Layers; layer++)
    {
        int n = kvLen * Dim;
        if (nativeLayout)
        {
            l.Add(NamedOnnxValue.CreateFromTensor($"past_k_layer_{layer}", new DenseTensor<float>(Noise(n), new[] { 1, Heads, HeadDim, kvLen })));
            l.Add(NamedOnnxValue.CreateFromTensor($"past_v_layer_{layer}", new DenseTensor<float>(Noise(n), new[] { 1, Heads, kvLen, HeadDim })));
        }
        else
        {
            l.Add(NamedOnnxValue.CreateFromTensor($"past_k_layer_{layer}", new DenseTensor<float>(Noise(n), new[] { kvLen, 1, Dim })));
            l.Add(NamedOnnxValue.CreateFromTensor($"past_v_layer_{layer}", new DenseTensor<float>(Noise(n), new[] { kvLen, 1, Dim })));
        }
    }
    return l;
}

var cases = new[] { (2, 32), (16, 96), (48, 160), (96, 320) };
Console.WriteLine($"[ab] steps/round={steps} rounds={rounds}");
Console.WriteLine($"[ab] A={Path.GetFileName(aPath)} ({aLayout})");
Console.WriteLine($"[ab] B={Path.GetFileName(bPath)} ({bLayout})");

foreach (var (iy, kv) in cases)
{
    var fa = Feed(aLayout == "native", iy, kv);
    var fb = Feed(bLayout == "native", iy, kv);
    for (int i = 0; i < 6; i++) { using (sa.Run(fa)) { } using (sb.Run(fb)) { } }

    var at = new List<double>();
    var bt = new List<double>();
    for (int r = 0; r < rounds; r++)
    {
        var a = new List<double>(steps);
        for (int i = 0; i < steps; i++) { var sw = Stopwatch.StartNew(); using var _ = sa.Run(fa); a.Add(sw.Elapsed.TotalMilliseconds); }
        var b = new List<double>(steps);
        for (int i = 0; i < steps; i++) { var sw = Stopwatch.StartNew(); using var _ = sb.Run(fb); b.Add(sw.Elapsed.TotalMilliseconds); }
        a.Sort(); b.Sort();
        at.Add(a[a.Count / 2]);
        bt.Add(b[b.Count / 2]);
    }
    at.Sort(); bt.Sort();
    double am = at[at.Count / 2], bm = bt[bt.Count / 2];
    Console.WriteLine($"  iy={iy,3} kv={kv,3}  A {am,7:F2}  B {bm,7:F2}  A/B {am / bm,5:F3}x  " +
                      $"(A {at[0]:F1}-{at[^1]:F1}, B {bt[0]:F1}-{bt[^1]:F1})");
}
