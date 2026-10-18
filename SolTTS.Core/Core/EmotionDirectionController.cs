using System.Text.Json;

namespace SolTTS.Core.Core
{
    /// <summary>
    /// Emotion direction vector (τ) controller.
    ///
    /// Principle: the mean difference of speaker embeddings in an emotional speech dataset forms an "emotion direction".
    /// At inference, x_new = x_base + α·||x_base||·τ̂ is injected into the speaker embedding (sv_emb, 20480-dim),
    /// and ge/ge_advanced are recomputed through prompt_encoder to steer the vocoder toward emotional speech
    /// without needing an emotional reference audio.
    ///
    /// τ files are extracted from the CSEMOTIONS dataset by tools/emotion/extract_tau.py
    /// using a within-speaker paired-difference formula to avoid mixing in speaker identity differences:
    ///     τ_e = mean_{s,i} ( x_e^{s,i} − x_n^{s} )
    /// </summary>
    public sealed class EmotionDirectionController
    {
        /// <summary>Neutral emotion name (no injection).</summary>
        public const string NeutralEmotion = "neutral";

        private readonly Dictionary<string, float[]> _tau = new(StringComparer.OrdinalIgnoreCase);

        /// <summary>Directory containing the τ files.</summary>
        public string DirectoryPath { get; }

        /// <summary>τ vector dimension (expected to match speaker_encoder output, 20480).</summary>
        public int Dim { get; private set; }

        /// <summary>Whether at least one τ was loaded successfully.</summary>
        public bool Loaded => _tau.Count > 0;

        /// <summary>The set of loaded emotion names.</summary>
        public IReadOnlyCollection<string> Emotions => _tau.Keys;

        public EmotionDirectionController(string? dir)
        {
            DirectoryPath = dir ?? string.Empty;
            Reload();
        }

        /// <summary>
        /// Load manifest.json and tau_*.f32 from the directory. Missing/corrupt files degrade silently (Loaded=false).
        /// All τ vectors are renormalized on load so τ̂ in the injection formula is a unit vector.
        /// </summary>
        public void Reload()
        {
            _tau.Clear();
            Dim = 0;
            try
            {
                if (string.IsNullOrEmpty(DirectoryPath) || !Directory.Exists(DirectoryPath)) return;

                var manifestPath = Path.Combine(DirectoryPath, "manifest.json");
                bool readAny = false;
                if (File.Exists(manifestPath))
                {
                    using var doc = JsonDocument.Parse(File.ReadAllText(manifestPath));
                    var root = doc.RootElement;
                    if (root.TryGetProperty("dim", out var dimEl) && dimEl.ValueKind == JsonValueKind.Number)
                        Dim = dimEl.GetInt32();

                    if (root.TryGetProperty("emotions", out var emos) && emos.ValueKind == JsonValueKind.Object)
                    {
                        foreach (var prop in emos.EnumerateObject())
                        {
                            var file = prop.Value.TryGetProperty("file", out var f) ? f.GetString() : null;
                            if (string.IsNullOrEmpty(file)) file = $"tau_{prop.Name}.f32";
                            if (TryLoadVector(Path.Combine(DirectoryPath, file), out var vec))
                            {
                                _tau[prop.Name] = vec;
                                readAny = true;
                            }
                        }
                    }
                }

                // Fall back to scanning the conventional filenames when the manifest is missing/incomplete
                if (!readAny)
                {
                    foreach (var file in Directory.EnumerateFiles(DirectoryPath, "tau_*.f32"))
                    {
                        var name = Path.GetFileNameWithoutExtension(file);
                        if (!name.StartsWith("tau_", StringComparison.OrdinalIgnoreCase)) continue;
                        name = name.Substring(4);
                        if (TryLoadVector(file, out var vec))
                        {
                            _tau[name] = vec;
                            readAny = true;
                            if (Dim == 0) Dim = vec.Length;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Emotion] Failed to load τ: {ex.Message}");
                _tau.Clear();
            }
        }

        private static bool TryLoadVector(string path, out float[] vec)
        {
            vec = Array.Empty<float>();
            try
            {
                if (!File.Exists(path)) return false;
                var bytes = File.ReadAllBytes(path);
                if (bytes.Length < 4 || bytes.Length % 4 != 0) return false;
                var arr = new float[bytes.Length / 4];
                Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);
                Normalize(arr);
                vec = arr;
                return true;
            }
            catch
            {
                return false;
            }
        }

        private static void Normalize(float[] v)
        {
            double sum = 0;
            for (int i = 0; i < v.Length; i++) sum += (double)v[i] * v[i];
            double norm = Math.Sqrt(sum);
            if (norm < 1e-12) return;
            float inv = (float)(1.0 / norm);
            for (int i = 0; i < v.Length; i++) v[i] *= inv;
        }

        /// <summary>Get the normalized τ for the given emotion (null if absent).</summary>
        public float[]? GetTau(string emotion)
            => _tau.TryGetValue(emotion, out var t) ? t : null;

        /// <summary>
        /// Injection formula: x_new = x_base + α·||x_base||·τ̂.
        /// τ̂ is a unit vector (guaranteed by GetTau).
        /// </summary>
        public static float[] Apply(float[] xBase, float[] tauNormalized, float alpha)
        {
            var result = new float[xBase.Length];
            double sum = 0;
            for (int i = 0; i < xBase.Length; i++) sum += (double)xBase[i] * xBase[i];
            float scale = alpha * (float)Math.Sqrt(sum);

            int n = Math.Min(xBase.Length, tauNormalized.Length);
            for (int i = 0; i < n; i++)
                result[i] = xBase[i] + scale * tauNormalized[i];
            if (n < xBase.Length)
                Array.Copy(xBase, n, result, n, xBase.Length - n);
            return result;
        }
    }
}
