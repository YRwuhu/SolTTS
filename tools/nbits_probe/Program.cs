using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// Is MatMulNBits (weight-only INT4) faster than fp32 for the single-token
// matvecs the decoder spends its time in? Scaling across N and block size says
// whether the cost is per-call overhead or the kernel itself.
//
//   dotnet run --project tools/nbits_probe -c Release -- [--steps N]
int steps = 200;
for (int i = 0; i < args.Length - 1; i++) if (args[i] == "--steps") steps = int.Parse(args[i + 1]);
var so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
var rng = new Random(1);

double Time(string label, string path, int K, int N)
{
    try
    {
        using var s = new InferenceSession(path, so);
        var act = new float[K];
        for (int i = 0; i < K; i++) act[i] = (float)(rng.NextDouble() - 0.5);
        NamedOnnxValue[] feed() => new[] { NamedOnnxValue.CreateFromTensor("A", new DenseTensor<float>(act, new[] { 1, K })) };
        for (int i = 0; i < 30; i++) using (s.Run(feed())) { }
        var t = new List<double>(steps);
        for (int i = 0; i < steps; i++)
        {
            var sw = Stopwatch.StartNew();
            using var _ = s.Run(feed());
            t.Add(sw.Elapsed.TotalMilliseconds);
        }
        t.Sort();
        double p50 = t[t.Count / 2];
        Console.WriteLine($"{label,-34} p50 {p50,7:F3} ms    weight KiB {(long)K * N / 1024}");
        return p50;
    }
    catch (Exception ex) { Console.WriteLine($"{label,-34} FAILED: {ex.Message.Split('\n')[0]}"); return double.NaN; }
}

Console.WriteLine("[nbits] single-token matvec, fp32 vs int4");
foreach (var (K, N, bl) in new[] { (512, 512, 32), (512, 2048, 16), (512, 2048, 32), (512, 2048, 128), (2048, 512, 32) })
{
    double f = Time($"fp32 MatMul   K={K} N={N}", $"E:/AI/_work/synth/fp32_{K}_{N}.onnx", K, N);
    double q = Time($"int4 NBits    K={K} N={N} blk={bl}", $"E:/AI/_work/synth/nb_{K}_{N}_{bl}.onnx", K, N);
    if (!double.IsNaN(f) && !double.IsNaN(q)) Console.WriteLine($"   -> int4 is {f / q:F2}x of fp32 speed".Replace("of fp32 speed", f / q >= 1 ? "faster" : "slower"));
}
