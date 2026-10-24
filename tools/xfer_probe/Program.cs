using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SolTTS.Core.Core;

// Isolates the ggml plugin's per-tensor transfer overhead at the KV boundary.
// Same byte count, split across 1 vs 48 outputs; both are pure copies.
string[] models = { "E:/AI/_work/synth/one_out.onnx", "E:/AI/_work/synth/many_out.onnx" };
var so = GgmlExecutionProvider.TryCreateSessionOptions("gpu", 0, partial: false);
if (so == null) { Console.WriteLine("ggml unavailable"); return; }
so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC;

foreach (var path in models)
{
    using var s = new InferenceSession(path, so);
    var feed = new[] { NamedOnnxValue.CreateFromTensor("x", new DenseTensor<float>(new float[48 * 160 * 512], new[] { 48, 160, 1, 512 })) };
    for (int i = 0; i < 5; i++) using (s.Run(feed)) { }
    var t = new List<double>();
    for (int i = 0; i < 20; i++) { var sw = Stopwatch.StartNew(); using var _ = s.Run(feed); t.Add(sw.Elapsed.TotalMilliseconds); }
    t.Sort();
    Console.WriteLine($"{Path.GetFileName(path),-16} outputs={s.OutputMetadata.Count,3}  15.7 MB each way  p50 {t[t.Count / 2],7:F2} ms");
}
