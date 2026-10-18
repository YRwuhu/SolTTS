using System.Diagnostics;
using SolTTS.Core.Config;
using SolTTS.Core.Core;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// 关键实验：把 KV 长度固定为极小值，排除 KV 拷贝干扰，看纯权重带宽的表现。
string capDir = @"E:\AI\_ab\cap";
string model = "resources/models_v1/chenxing/t2s_stage_decoder.onnx";

var meta = new Dictionary<string, (string type, int[] dims, string file)>();
foreach (var line in File.ReadAllLines(Path.Combine(capDir, "manifest.txt")))
{
    var p = line.Split('|');
    meta[p[0]] = (p[1], p[2].Split(',').Select(int.Parse).ToArray(), p[0] + (p[1] == "long" ? ".i64" : ".f32"));
}
List<NamedOnnxValue> Load(int iyLen, int kvLen)
{
    var list = new List<NamedOnnxValue>();
    foreach (var (name, (type, dims, file)) in meta)
    {
        int[] d = dims;
        if (name == "iy") d = new[] { 1, iyLen };
        else if (name == "iy_emb") d = new[] { 1, iyLen - 1, 512 };
        else if (name.StartsWith("past_")) d = new[] { kvLen, 1, 512 };
        int n = d.Aggregate(1, (a, b) => a * b);
        var bytes = File.ReadAllBytes(Path.Combine(capDir, file));
        if (type == "long") { var a = new long[n]; Buffer.BlockCopy(bytes,0,a,0,Math.Min(n,bytes.Length/8)*8); list.Add(NamedOnnxValue.CreateFromTensor(name,new DenseTensor<long>(a,d))); }
        else { var a = new float[n]; Buffer.BlockCopy(bytes,0,a,0,Math.Min(n,bytes.Length/4)*4); list.Add(NamedOnnxValue.CreateFromTensor(name,new DenseTensor<float>(a,d))); }
    }
    return list;
}

void Scan(string label, SessionOptions so)
{
    using var session = new InferenceSession(model, so);
    Console.WriteLine($"\n=== {label} ===");
    foreach (var (iy, kv) in new[] { (4, 8), (16, 24), (48, 60), (108, 138) })
    {
        var inputs = Load(iy, kv);
        for (int i = 0; i < 3; i++) using (session.Run(inputs)) { }
        var times = new List<double>();
        for (int r = 0; r < 3; r++)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < 10; i++) using (session.Run(inputs)) { }
            sw.Stop(); times.Add(sw.Elapsed.TotalMilliseconds / 10);
        }
        times.Sort();
        Console.WriteLine($"  iy={iy,3} kv={kv,3} -> {times[1]:F2} ms/step");
    }
}

var cfg = new TTSConfig();
Scan("CPU", new InferenceEngineV1().GetSessionOptions(cfg));
var gpuOpts = GgmlExecutionProvider.TryCreateSessionOptions("gpu", 0, partial: true);
if (gpuOpts != null) Scan("ggml partial", gpuOpts);
