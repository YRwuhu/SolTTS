using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// Stage-3 step 3: is packing the 48 KV tensors into one worth it?
//
// The decoder hands 48 separate tensors across the boundary every step. If a
// single [48,...] tensor is materially cheaper to move and the graph can slice it
// back internally with views, that is a cheap substitute for a device allocator.
// Here we time the pure hand-off cost of the two shapes, no compute.
//
//   dotnet run --project tools/kv_pack_probe -c Release -- [--kv S] [--steps N]
int kv = 320, steps = 40;
for (int i = 0; i < args.Length - 1; i++)
{
    if (args[i] == "--kv") kv = int.Parse(args[i + 1]);
    if (args[i] == "--steps") steps = int.Parse(args[i + 1]);
}
string dir = "E:/AI/_work/synth";
string packed = Path.Combine(dir, $"kvpack_one_{kv}.onnx");
string split = Path.Combine(dir, $"kvpack_48_{kv}.onnx");
if (!File.Exists(packed) || !File.Exists(split))
{
    Console.WriteLine($"[pack] build the synthetic graphs first (see docs/DECODER_BASELINE.md)");
    return;
}

var so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
int H = 16, D = 32;
double Time(string path, bool isPacked)
{
    using var s = new InferenceSession(path, so);
    List<NamedOnnxValue> feed;
    if (isPacked)
        feed = new() { NamedOnnxValue.CreateFromTensor("X", new DenseTensor<float>(new float[48 * kv * H * D], new[] { 48, kv, H, D })) };
    else
    {
        feed = new();
        for (int i = 0; i < 48; i++)
            feed.Add(NamedOnnxValue.CreateFromTensor($"X{i}", new DenseTensor<float>(new float[kv * H * D], new[] { kv, H, D })));
    }
    for (int i = 0; i < 5; i++) using (s.Run(feed)) { }
    var t = new List<double>(steps);
    for (int i = 0; i < steps; i++) { var sw = Stopwatch.StartNew(); using var _ = s.Run(feed); t.Add(sw.Elapsed.TotalMilliseconds); }
    t.Sort();
    return t[t.Count / 2];
}
double one = Time(packed, true);
double many = Time(split, false);
Console.WriteLine($"[pack] kv={kv}  1 tensor {one,7:F2} ms   48 tensors {many,7:F2} ms   ratio {many / one:F2}x");
