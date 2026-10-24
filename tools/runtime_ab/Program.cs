using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// A/B the ONNX Runtime session options from the v2 plan's stage 1.
//
// A single pass over a 12 ms workload is dominated by scheduling and GC noise,
// so each variant is interleaved across several rounds and the reported number
// is the median of the per-round medians. Variants whose spread overlaps the
// baseline's are not improvements, whatever the point estimate says.
//
//   dotnet run --project tools/runtime_ab -c Release -- [--model P] [--steps N] [--rounds R]
string model = "resources/models_v1/chenxing/t2s_stage_decoder_qkv.onnx";
int steps = 30, rounds = 5;
for (int i = 0; i < args.Length - 1; i++)
{
    if (args[i] == "--model") model = args[i + 1];
    if (args[i] == "--steps") steps = int.Parse(args[i + 1]);
    if (args[i] == "--rounds") rounds = int.Parse(args[i + 1]);
}

const int Layers = 24, Dim = 512, Kv = 96, Iy = 2;
var rng = new Random(1234);
float[] Noise(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * 0.2); return a; }
var inputs = new List<NamedOnnxValue>(50)
{
    NamedOnnxValue.CreateFromTensor("iy", new DenseTensor<long>(new long[] { 3, 17 }, new[] { 1, Iy })),
    NamedOnnxValue.CreateFromTensor("iy_emb", new DenseTensor<float>(Noise((Iy - 1) * Dim), new[] { 1, Iy - 1, Dim })),
};
for (int l = 0; l < Layers; l++)
{
    inputs.Add(NamedOnnxValue.CreateFromTensor($"past_k_layer_{l}", new DenseTensor<float>(Noise(Kv * Dim), new[] { Kv, 1, Dim })));
    inputs.Add(NamedOnnxValue.CreateFromTensor($"past_v_layer_{l}", new DenseTensor<float>(Noise(Kv * Dim), new[] { Kv, 1, Dim })));
}

var variants = new (string Name, Action<SessionOptions> Configure)[]
{
    ("baseline", _ => { }),
    ("IntraOp=3", so => so.IntraOpNumThreads = 3),
    ("IntraOp=6", so => so.IntraOpNumThreads = 6),
    ("IntraOp=8", so => so.IntraOpNumThreads = 8),
    ("IntraOp=12", so => so.IntraOpNumThreads = 12),
    ("InterOp=1", so => so.InterOpNumThreads = 1),
    ("no memory pattern", so => so.EnableMemoryPattern = false),
    ("no cpu arena", so => so.EnableCpuMemArena = false),
    ("spinning off", so => so.AddSessionConfigEntry("session.intra_op.allow_spinning", "0")),
    ("spinning on", so => so.AddSessionConfigEntry("session.intra_op.allow_spinning", "1")),
    ("model bytes directly", so => so.AddSessionConfigEntry("session.use_ort_model_bytes_directly", "1")),
    ("thread affinity", so => so.AddSessionConfigEntry("session.intra_op.force_thread_affinity", "1")),
    ("no prepacking", so => so.AddSessionConfigEntry("session.disable_prepacking", "1")),
    ("disable dyn block quant", so => so.AddSessionConfigEntry("session.disable_dynamic_block_quantization", "1")),
};

var sessions = new Dictionary<string, InferenceSession>();
var options = new Dictionary<string, SessionOptions>();
foreach (var (name, configure) in variants)
{
    var so = new SessionOptions
    {
        GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
    };
    configure(so);
    options[name] = so;
    sessions[name] = new InferenceSession(model, so);
    for (int i = 0; i < 8; i++) using (sessions[name].Run(inputs)) { }
}

var perRound = variants.ToDictionary(v => v.Name, _ => new List<double>());
for (int r = 0; r < rounds; r++)
{
    foreach (var (name, _) in variants)
    {
        var t = new List<double>(steps);
        for (int i = 0; i < steps; i++)
        {
            var sw = Stopwatch.StartNew();
            using var _ = sessions[name].Run(inputs);
            t.Add(sw.Elapsed.TotalMilliseconds);
        }
        t.Sort();
        perRound[name].Add(t[t.Count / 2]);
    }
}

Console.WriteLine($"[ab] {Path.GetFileName(model)} kv={Kv} iy={Iy} steps/round={steps} rounds={rounds}");
Console.WriteLine($"{"variant",-22} {"median",8} {"min",8} {"max",8}   vs baseline");
double baseline = perRound["baseline"].OrderBy(x => x).ElementAt(rounds / 2);
foreach (var (name, _) in variants)
{
    var v = perRound[name].OrderBy(x => x).ToList();
    double med = v[rounds / 2];
    Console.WriteLine($"{name,-22} {med,8:F2} {v[0],8:F2} {v[^1],8:F2}   {baseline / med,5:F2}x");
}

foreach (var s in sessions.Values) s.Dispose();
foreach (var s in options.Values) s.Dispose();
