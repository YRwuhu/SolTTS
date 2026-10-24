using System.Text.Json;
using SolTTS.Core.Config;
using SolTTS.Core.Core;

// Render the same text from two model directories and compare the audio.
// Both model sets must have the sampler seeded, otherwise RandomNormalLike
// makes every render different and the comparison is meaningless.
//
//   dotnet run --project tools/audio_cmp -c Release -- --a DIR_A --b DIR_B [--text "..."]
string dirA = "", dirB = "";
string text = "今天天气真好，我们一起去公园散步吧。";
for (int i = 0; i < args.Length - 1; i++)
{
    if (args[i] == "--a") dirA = args[i + 1];
    if (args[i] == "--b") dirB = args[i + 1];
    if (args[i] == "--text") text = args[i + 1];
}
if (dirA == "" || dirB == "") { Console.WriteLine("usage: --a DIR_A --b DIR_B [--text ...]"); return; }

var config = TTSConfig.Load();
config.ResourcesDir = Path.GetFullPath("resources");
config.PoolCapacity = 1;
config.ParallelSentenceDegree = 1;

async Task<float[]> Render(string modelDir)
{
    // Model paths derive from ResourcesDir, so point a temp resource root at the
    // directory under test and keep everything else from the real config.
    string root = Path.Combine(Path.GetTempPath(), "audio_cmp_" + Guid.NewGuid().ToString("N"));
    Directory.CreateDirectory(Path.Combine(root, "models_v1"));
    Directory.CreateSymbolicLink(Path.Combine(root, "models_v1", "chenxing"), Path.GetFullPath(modelDir));
    // Shared assets (G2P, BERT, avatars) come from the real tree.
    string realResources = Path.GetFullPath("resources");
    Directory.CreateSymbolicLink(Path.Combine(root, "shared"), Path.Combine(realResources, "shared"));
    Directory.CreateSymbolicLink(Path.Combine(root, "avatars"), Path.Combine(realResources, "avatars"));
    Directory.CreateSymbolicLink(Path.Combine(root, "emotion_vectors"), Path.Combine(realResources, "emotion_vectors"));

    var cfg = TTSConfig.Load();
    cfg.ResourcesDir = root;
    cfg.PoolCapacity = 1;
    cfg.ParallelSentenceDegree = 1;
    // The vocoder is stochastic too; take it out of the comparison so a
    // difference in the audio can only come from the decoders.
    cfg.NoiseScale = 0f;
    using var sdk = new SolTtsSdk(cfg);
    await sdk.InitializeAsync();
    var chunks = new List<float>();
    await foreach (var c in sdk.PredictStreamAsync(text, new TtsOptions { VocoderRetriggerFactor = 1.5f }, cfg.DefaultAvatarId, null))
        chunks.AddRange(c);
    try { Directory.Delete(root, recursive: true); } catch { }
    return chunks.ToArray();
}

Console.WriteLine($"[cmp] A={dirA}\n[cmp] B={dirB}\n[cmp] text={text}");
var a = await Render(dirA);
var b = await Render(dirB);
Console.WriteLine($"[cmp] A samples={a.Length} ({a.Length / 32000.0:F2}s)   B samples={b.Length} ({b.Length / 32000.0:F2}s)");
int n = Math.Min(a.Length, b.Length);
if (n == 0) { Console.WriteLine("[cmp] FAIL: no audio"); return; }
double dot = 0, na = 0, nb = 0, maxd = 0;
for (int i = 0; i < n; i++)
{
    dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
    maxd = Math.Max(maxd, Math.Abs(a[i] - b[i]));
}
double cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
Console.WriteLine($"[cmp] cosine={cos:F8}  max|Δ|={maxd:E3}  lengthΔ={Math.Abs(a.Length - b.Length)} samples");
Console.WriteLine(cos >= 0.9999 ? "[cmp] PASS" : "[cmp] FAIL");
