using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// Is the per-step DynamicQuantizeLinear worth removing? Compares ORT's fused
// DynamicQuantizeMatMul against a static-scale QuantizeLinear + MatMulInteger at
// the decoder's real matvec shape.
//
//   dotnet run --project tools/quant_probe -c Release -- [--steps N]
int steps = 300;
for (int i = 0; i < args.Length - 1; i++) if (args[i] == "--steps") steps = int.Parse(args[i + 1]);
var so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
var rng = new Random(1);
var x = new float[512];
for (int i = 0; i < 512; i++) x[i] = (float)((rng.NextDouble() * 2 - 1) * 2.0);

(double ms, float[] y) Time(string label, string path)
{
    using var s = new InferenceSession(path, so);
    NamedOnnxValue[] feed() => new[] { NamedOnnxValue.CreateFromTensor("X", new DenseTensor<float>(x, new[] { 1, 512 })) };
    for (int i = 0; i < 30; i++) using (s.Run(feed())) { }
    var t = new List<double>(steps);
    float[] last = Array.Empty<float>();
    for (int i = 0; i < steps; i++)
    {
        var sw = Stopwatch.StartNew();
        using var r = s.Run(feed());
        t.Add(sw.Elapsed.TotalMilliseconds);
        last = r.First().AsTensor<float>().ToArray();
    }
    t.Sort();
    Console.WriteLine($"{label,-28} p50 {t[t.Count / 2],7:F3} ms  p95 {t[(int)(t.Count * 0.95)],7:F3} ms");
    return (t[t.Count / 2], last);
}

var (dynMs, dynY) = Time("DynamicQuantizeMatMul", "E:/AI/_work/synth/dyn_mm.onnx");
var (statMs, statY) = Time("static QuantizeLinear", "E:/AI/_work/synth/stat_mm.onnx");
double bad = 0;
for (int i = 0; i < dynY.Length; i++) bad = Math.Max(bad, Math.Abs(dynY[i] - statY[i]));
double scale = dynY.Select(Math.Abs).Max();
Console.WriteLine($"[quant] static/dynamic time ratio {dynMs / statMs:F2}x   max|diff| {bad:F4} ({(scale > 0 ? bad / scale * 100 : 0):F2}% of peak)");
