using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SolTTS.Core.Config;
using SolTTS.Core.Core;

// Decoder single-step benchmark.
//
//   dotnet run --project tools/decoder_bench -c Release -- [--model P] [--steps N]
//       [--warmup W] [--threads T] [--provider cpu|ggml|all] [--kv-layout flat|native]
//
// Shapes: iy=[1,L], iy_emb=[1,L-1,512], 24 layers of past_k/v. `--kv-layout flat`
// is the shipped [S,1,512]; `native` is what fuse_kv_layout.py produces,
// K=[1,16,32,S] and V=[1,16,S,32].

string model = "resources/models_v1/chenxing/t2s_stage_decoder_qkv.onnx";
int steps = 40, warmup = 8, threads = 0;
string provider = "all", kvLayout = "flat";
for (int i = 0; i < args.Length - 1; i++)
{
    switch (args[i])
    {
        case "--model": model = args[i + 1]; break;
        case "--steps": steps = int.Parse(args[i + 1]); break;
        case "--warmup": warmup = int.Parse(args[i + 1]); break;
        case "--threads": threads = int.Parse(args[i + 1]); break;
        case "--provider": provider = args[i + 1]; break;
        case "--kv-layout": kvLayout = args[i + 1]; break;
    }
}

const int Layers = 24, Dim = 512, Heads = 16, HeadDim = 32, Vocab = 1025;
bool native = kvLayout == "native";

Console.WriteLine($"[bench] {Path.GetFileName(model)} steps={steps} warmup={warmup} " +
                  $"threads={(threads == 0 ? "auto" : threads)} kv={kvLayout}");

var rng = new Random(1234);
float[] Noise(int n, float scale = 0.2f)
{
    var a = new float[n];
    for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
    return a;
}

List<NamedOnnxValue> Build(int iyLen, int kvLen)
{
    var list = new List<NamedOnnxValue>(50)
    {
        NamedOnnxValue.CreateFromTensor("iy", new DenseTensor<long>(
            Enumerable.Range(0, iyLen).Select(i => (long)(i % Vocab)).ToArray(), new[] { 1, iyLen })),
        NamedOnnxValue.CreateFromTensor("iy_emb", new DenseTensor<float>(Noise((iyLen - 1) * Dim), new[] { 1, iyLen - 1, Dim })),
    };
    for (int l = 0; l < Layers; l++)
    {
        int n = kvLen * Dim;
        if (native)
        {
            list.Add(NamedOnnxValue.CreateFromTensor($"past_k_layer_{l}", new DenseTensor<float>(Noise(n), new[] { 1, Heads, HeadDim, kvLen })));
            list.Add(NamedOnnxValue.CreateFromTensor($"past_v_layer_{l}", new DenseTensor<float>(Noise(n), new[] { 1, Heads, kvLen, HeadDim })));
        }
        else
        {
            list.Add(NamedOnnxValue.CreateFromTensor($"past_k_layer_{l}", new DenseTensor<float>(Noise(n), new[] { kvLen, 1, Dim })));
            list.Add(NamedOnnxValue.CreateFromTensor($"past_v_layer_{l}", new DenseTensor<float>(Noise(n), new[] { kvLen, 1, Dim })));
        }
    }
    return list;
}

void Bench(string label, SessionOptions so)
{
    try
    {
        var swLoad = Stopwatch.StartNew();
        using var session = new InferenceSession(model, so);
        swLoad.Stop();
        Console.WriteLine($"\n=== {label} (load {swLoad.ElapsedMilliseconds} ms) ===");
        // A real late decode step has a long iy as well as a long KV.
        foreach (var (iy, kv) in new[] { (2, 32), (2, 96), (2, 160), (48, 96), (96, 160), (160, 320) })
        {
            var inputs = Build(iy, kv);
            for (int i = 0; i < warmup; i++) using (session.Run(inputs)) { }

            var times = new List<double>(steps);
            for (int i = 0; i < steps; i++)
            {
                var sw = Stopwatch.StartNew();
                using var _ = session.Run(inputs);
                sw.Stop();
                times.Add(sw.Elapsed.TotalMilliseconds);
            }
            times.Sort();
            double p50 = times[times.Count / 2];
            double p95 = times[(int)Math.Min(times.Count - 1, Math.Ceiling(times.Count * 0.95) - 1)];
            Console.WriteLine($"  iy={iy,3} kv={kv,3}  p50 {p50,7:F2} ms   p95 {p95,7:F2} ms   min {times[0],7:F2}");
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"\n=== {label} FAILED: {ex.Message.Split('\n')[0]}");
    }
}

if (provider is "all" or "cpu")
{
    var cfg = new TTSConfig { IntraOpNumThreads = threads, InterOpNumThreads = 1 };
    Bench($"CPU threads={(threads == 0 ? "auto" : threads)}", new InferenceEngineV1().GetSessionOptions(cfg));
}
if (provider is "all" or "ggml")
{
    var so = GgmlExecutionProvider.TryCreateSessionOptions("gpu", 0, partial: false);
    if (so != null) Bench("ggml full claim", so);
}
