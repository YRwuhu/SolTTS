using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using SolTTS.Core.Core;

// Probe what the ggml plugin does with a graph: claim coverage and, optionally,
// a quick single-input timing. Used to check that a converter output is
// claimable before wiring it into the engine.
//
//   ggml-probe <model.onnx> [gol] [--partial] [--run]

string model = args.Length > 0 ? args[0] : "resources/models_v1/chenxing/t2s_stage_decoder.onnx";
int gol = args.Length > 1 && int.TryParse(args[1], out var g) ? g : 1;
bool partial = args.Contains("--partial");
bool run = args.Contains("--run");

Console.WriteLine($"[probe] model={Path.GetFileName(model)} GOL={gol} partial={partial}");
var so = GgmlExecutionProvider.TryCreateSessionOptions("gpu", 0, partial: partial);
if (so == null) { Console.WriteLine("[probe] ggml unavailable"); return; }
so.GraphOptimizationLevel = gol switch
{
    0 => GraphOptimizationLevel.ORT_DISABLE_ALL,
    1 => GraphOptimizationLevel.ORT_ENABLE_BASIC,
    _ => GraphOptimizationLevel.ORT_ENABLE_ALL
};

try
{
    var sw = Stopwatch.StartNew();
    using var session = new InferenceSession(model, so);
    Console.WriteLine($"[probe] session ready in {sw.ElapsedMilliseconds} ms, {session.InputMetadata.Count} inputs, {session.OutputMetadata.Count} outputs");
    if (run)
    {
        var rng = new Random(7);
        var inputs = new List<NamedOnnxValue>();
        foreach (var (name, meta) in session.InputMetadata)
        {
            var dims = meta.Dimensions.Select(d => d <= 0 ? 1 : d).ToArray();
            if (name == "past_k_layer_0" || name == "past_v_layer_0") dims = new[] { 8, 1, 512 };
            else if (name.StartsWith("past_")) dims = new[] { 8, 1, 512 };
            int n = dims.Aggregate(1, (a, b) => a * b);
            if (meta.ElementType == typeof(long))
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<long>(Enumerable.Range(0, n).Select(i => (long)(i % 1024)).ToArray(), dims)));
            else
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(Enumerable.Range(0, n).Select(i => (float)(rng.NextDouble() - 0.5) * 0.4f).ToArray(), dims)));
        }
        for (int i = 0; i < 3; i++) using (session.Run(inputs)) { }
        var times = new List<double>();
        for (int i = 0; i < 10; i++)
        {
            var sw2 = Stopwatch.StartNew();
            using var _ = session.Run(inputs);
            times.Add(sw2.Elapsed.TotalMilliseconds);
        }
        times.Sort();
        Console.WriteLine($"[probe] p50 {times[times.Count / 2]:F2} ms");
    }
}
catch (Exception ex) { Console.WriteLine("[probe] FAILED: " + ex.Message.Split('\n')[0]); }
