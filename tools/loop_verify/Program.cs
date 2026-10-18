using System.Diagnostics;
using SolTTS.Core.Config;
using SolTTS.Core.Core;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// Using the real captured initial state, run both:
//   A) Original model step by step for 30 steps (simulating current C# behavior)
//   B) Loop model in a single Run (trip=30)
// Compare whether the final y sequences match (sampling is random, differences allowed; focus on shape/chain correctness)
string capDir = @"E:\AI\_ab\capture";
string origPath = "resources/models_v1/chenxing/t2s_stage_decoder.onnx";
string loopPath = "_tmp/stage_loop_test.onnx";

var meta = new Dictionary<string,(string type,int[] dims,string file)>();
foreach (var line in File.ReadAllLines(Path.Combine(capDir,"manifest.txt")))
{
    var p=line.Split('|'); meta[p[0]]=(p[1],p[2].Split(',').Select(int.Parse).ToArray(),p[0]+(p[1]=="long"?".i64":".f32"));
}
NamedOnnxValue Load(string name)
{
    var (type,dims,file)=meta[name];
    var bytes=File.ReadAllBytes(Path.Combine(capDir,file));
    int n=dims.Aggregate(1,(a,b)=>a*b);
    if(type=="long"){var a=new long[n];Buffer.BlockCopy(bytes,0,a,0,n*8);return NamedOnnxValue.CreateFromTensor(name,new DenseTensor<long>(a,dims));}
    var f=new float[n];Buffer.BlockCopy(bytes,0,f,0,n*4);return NamedOnnxValue.CreateFromTensor(name,new DenseTensor<float>(f,dims));
}

var cfg=new TTSConfig();
var so=new InferenceEngineV1().GetSessionOptions(cfg);

int steps=30;

// ---- A: original model step by step ----
var swA=Stopwatch.StartNew();
string lastY; int aTokens=0;
{
    using var sess=new InferenceSession(origPath,so);
    var y=Load("iy"); var yemb=Load("iy_emb");
    var kv=new NamedOnnxValue[48];
    for(int l=0;l<24;l++){kv[l*2]=Load($"past_k_layer_{l}");kv[l*2+1]=Load($"past_v_layer_{l}");}
    int initLen=((Tensor<long>)y.Value).Dimensions[1];
    IDisposable? holder=null;
    Tensor<long> yT=(Tensor<long>)y.Value;
    Tensor<float> yEmbT=(Tensor<float>)yemb.Value;
    var kvT=new Tensor<float>[48];
    for(int l=0;l<24;l++){kvT[l*2]=(Tensor<float>)kv[l*2].Value;kvT[l*2+1]=(Tensor<float>)kv[l*2+1].Value;}
    for(int i=0;i<steps;i++)
    {
        var inp=new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("iy", yT),
            NamedOnnxValue.CreateFromTensor("iy_emb", yEmbT),
        };
        for(int l=0;l<24;l++){
            inp.Add(NamedOnnxValue.CreateFromTensor($"past_k_layer_{l}", kvT[l*2]));
            inp.Add(NamedOnnxValue.CreateFromTensor($"past_v_layer_{l}", kvT[l*2+1]));
        }
        var res=sess.Run(inp);
        if(i==0) Console.WriteLine($"[verify] A outputs: {string.Join(",", res.Select(r=>r.Name).Take(5))}");
        // All tensors from the previous result have been consumed by this Run and can be released
        holder?.Dispose();
        holder=res;
        yT=res.First(r=>r.Name=="y").AsTensor<long>();
        yEmbT=res.First(r=>r.Name=="y_emb").AsTensor<float>();
        for(int l=0;l<24;l++){kvT[l*2]=res.First(r=>r.Name==$"present_k_layer_{l}").AsTensor<float>();kvT[l*2+1]=res.First(r=>r.Name==$"present_v_layer_{l}").AsTensor<float>();}
        bool stop=res.First(r=>r.Name=="stop_condition_tensor").AsTensor<bool>().GetValue(0);
        if(stop) break;
        aTokens++;
    }
    holder?.Dispose();
    lastY=string.Join(",", yT.Dimensions.ToArray());
}
swA.Stop();
Console.WriteLine($"[verify] A (per-step x{steps}): {swA.ElapsedMilliseconds} ms, final y dims=[{lastY}]");

// ---- B: single Loop run ----
var swB=Stopwatch.StartNew();
string bDims;
{
    using var sess=new InferenceSession(loopPath,so);
    var inp=new List<NamedOnnxValue>
    {
        NamedOnnxValue.CreateFromTensor("loop_max_steps", new DenseTensor<long>(new[]{ (long)steps }, Array.Empty<int>())),
        NamedOnnxValue.CreateFromTensor("loop_cond_init", new DenseTensor<bool>(new[]{ true }, Array.Empty<int>())),
    };
    for(int l=0;l<24;l++){inp.Add(Load($"past_k_layer_{l}"));inp.Add(Load($"past_v_layer_{l}"));}
    inp.Add(Load("iy_emb"));
    inp.Add(Load("iy"));
    using var res=sess.Run(inp);
    var yOut=res.First(r=>r.Name=="y_out").AsTensor<long>();
    bDims=string.Join(",",yOut.Dimensions.ToArray());
    Console.WriteLine($"[verify] B outputs={res.Count} keys={string.Join(",",res.Select(r=>r.Name).Take(3))}");
}
swB.Stop();
Console.WriteLine($"[verify] B (loop x{steps}): {swB.ElapsedMilliseconds} ms, final y dims=[{bDims}]");
Console.WriteLine($"[verify] speedup = {(double)swA.ElapsedMilliseconds/Math.Max(1,swB.ElapsedMilliseconds):F2}x");
