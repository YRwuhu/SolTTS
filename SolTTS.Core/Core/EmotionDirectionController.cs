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
    /// τ files are produced by tools/emotion/extract_tau.py (either the CSEMOTIONS paired-difference recipe or
    /// the ATRI single-speaker calm-baseline recipe):
    ///     τ_e = mean_i ( x_e^{i} − mean_j x_calm^{j} )
    ///
    /// Loading is strict on purpose. A vector that fails validation is never silently used, and an unknown or
    /// unusable emotion is reported to the caller instead of falling back to the neutral path.
    /// </summary>
    public sealed class EmotionDirectionController
    {
        /// <summary>Legacy constant kept for source compatibility; the canonical baseline lives in <see cref="EmotionCatalog"/>.</summary>
        public const string NeutralEmotion = EmotionCatalog.Baseline;

        /// <summary>Expected speaker-embedding dimension produced by speaker_encoder.onnx.</summary>
        public const int ExpectedDim = 20480;

        /// <summary>Below this raw L2 norm a vector is treated as degenerate/empty.</summary>
        private const double MinRawNorm = 1e-6;

        private readonly Dictionary<string, float[]> _tau = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, EmotionInfo> _info = new(StringComparer.OrdinalIgnoreCase);
        private readonly List<string> _loadErrors = new();

        /// <summary>Directory containing the τ files.</summary>
        public string DirectoryPath { get; }

        /// <summary>τ vector dimension actually loaded (0 when nothing loaded).</summary>
        public int Dim { get; private set; }

        /// <summary>Whether at least one τ was loaded successfully.</summary>
        public bool Loaded => _tau.Count > 0;

        /// <summary>The set of loaded emotion ids.</summary>
        public IReadOnlyCollection<string> Emotions => _tau.Keys;

        /// <summary>Manifest source description, when present.</summary>
        public string? Source { get; private set; }

        /// <summary>Manifest method description, when present.</summary>
        public string? Method { get; private set; }

        /// <summary>Problems found while loading (missing files, bad dimension, etc.).</summary>
        public IReadOnlyList<string> LoadErrors => _loadErrors;

        public EmotionDirectionController(string? dir)
        {
            DirectoryPath = dir ?? string.Empty;
            Reload();
        }

        // ====================================================================
        // Loading
        // ====================================================================

        /// <summary>Re-read the manifest and vectors from disk.</summary>
        public void Reload()
        {
            _tau.Clear();
            _info.Clear();
            _loadErrors.Clear();
            Dim = 0;
            Source = null;
            Method = null;

            // Everything in the catalog starts unavailable; loading can only improve that.
            foreach (var def in EmotionCatalog.All)
            {
                _info[def.Id] = new EmotionInfo
                {
                    Id = def.Id,
                    ZhName = def.ZhName,
                    EnName = def.EnName,
                    IsBaseline = def.IsBaseline,
                    Readiness = def.IsBaseline ? EmotionReadiness.Baseline : EmotionReadiness.Unavailable,
                    Reason = def.IsBaseline ? null : "no direction vector loaded",
                };
            }

            if (!System.IO.Directory.Exists(DirectoryPath))
            {
                _loadErrors.Add($"emotion vector directory not found: {DirectoryPath}");
                return;
            }

            var manifestPath = Path.Combine(DirectoryPath, "manifest.json");
            bool manifestLoaded = false;
            if (File.Exists(manifestPath))
            {
                try
                {
                    using var doc = JsonDocument.Parse(File.ReadAllText(manifestPath));
                    var root = doc.RootElement;
                    Source = GetString(root, "source_dataset");
                    Method = GetString(root, "method");
                    if (root.TryGetProperty("dim", out var dimEl) && dimEl.TryGetInt32(out int mdim))
                    {
                        if (mdim != ExpectedDim)
                            _loadErrors.Add($"manifest dim {mdim} != expected {ExpectedDim}");
                    }
                    if (root.TryGetProperty("emotions", out var emotions) && emotions.ValueKind == JsonValueKind.Object)
                    {
                        foreach (var prop in emotions.EnumerateObject())
                        {
                            LoadOne(prop.Name, prop.Value);
                            manifestLoaded = true;
                        }
                    }
                    else
                    {
                        _loadErrors.Add("manifest has no 'emotions' object");
                    }
                }
                catch (Exception ex)
                {
                    _loadErrors.Add($"manifest parse failed: {ex.Message}");
                }
            }
            else
            {
                // No manifest: accept legacy tau_<id>.f32 files so older bundles still load.
                foreach (var def in EmotionCatalog.Directional)
                {
                    var legacy = Path.Combine(DirectoryPath, $"tau_{def.Id}.f32");
                    if (File.Exists(legacy))
                    {
                        if (TryReadVector(legacy, out _, out var err))
                        {
                            LoadOne(def.Id, default);
                            manifestLoaded = true;
                        }
                        else
                        {
                            _loadErrors.Add($"{def.Id}: {err}");
                        }
                    }
                }
                if (!manifestLoaded)
                    _loadErrors.Add($"manifest.json not found in {DirectoryPath}");
            }

            if (!manifestLoaded && _loadErrors.Count == 0)
                _loadErrors.Add("no emotion vectors found");
        }

        /// <summary>Load one manifest entry (or a legacy id with <paramref name="entry"/> unset).</summary>
        private void LoadOne(string rawId, JsonElement entry)
        {
            if (!EmotionCatalog.TryNormalize(rawId, out var id))
            {
                _loadErrors.Add($"manifest lists unknown emotion '{rawId}' (ignored)");
                return;
            }
            if (EmotionCatalog.IsBaseline(id))
            {
                _loadErrors.Add($"manifest lists a vector for baseline '{id}' (ignored: baseline is never injected)");
                return;
            }

            var fileName = $"tau_{id}.f32";
            if (entry.ValueKind == JsonValueKind.Object)
            {
                var declared = GetString(entry, "file");
                if (!string.IsNullOrWhiteSpace(declared))
                {
                    // Reject path traversal: only a bare filename inside the vector directory is allowed.
                    if (declared.IndexOfAny(new[] { '/', '\\' }) >= 0 || declared.Contains(".."))
                    {
                        _loadErrors.Add($"{id}: illegal file name '{declared}'");
                        MarkUnavailable(id, "illegal file name in manifest");
                        return;
                    }
                    fileName = declared;
                }
            }

            var path = Path.Combine(DirectoryPath, fileName);
            if (!TryReadVector(path, out var vec, out var error))
            {
                _loadErrors.Add($"{id}: {error}");
                MarkUnavailable(id, error);
                return;
            }

            int? samples = null;
            double? consistency = null;
            string? quality = null;
            string? qualityReason = null;
            if (entry.ValueKind == JsonValueKind.Object)
            {
                if (entry.TryGetProperty("n_samples", out var ns) && ns.TryGetInt32(out int n)) samples = n;
                else if (entry.TryGetProperty("samples", out var s2) && s2.TryGetInt32(out int n2)) samples = n2;
                if (entry.TryGetProperty("consistency", out var c) && c.TryGetDouble(out double cv)) consistency = cv;
                quality = GetString(entry, "quality");
                qualityReason = GetString(entry, "quality_reason");
            }

            _tau[id] = vec;
            var readiness = EmotionReadiness.Ready;
            string? reason = null;
            if (string.Equals(quality, "experimental", StringComparison.OrdinalIgnoreCase))
            {
                readiness = EmotionReadiness.Experimental;
                reason = string.IsNullOrWhiteSpace(qualityReason)
                    ? "low sample count or low direction consistency"
                    : qualityReason;
            }
            else if (string.Equals(quality, "unavailable", StringComparison.OrdinalIgnoreCase))
            {
                readiness = EmotionReadiness.Unavailable;
                reason = string.IsNullOrWhiteSpace(qualityReason) ? "marked unavailable by extraction" : qualityReason;
            }
            else if (samples.HasValue && samples.Value < 20)
            {
                readiness = EmotionReadiness.Experimental;
                reason = $"only {samples} samples";
            }
            else if (consistency.HasValue && consistency.Value < 0.35)
            {
                readiness = EmotionReadiness.Experimental;
                reason = $"low direction consistency ({consistency.Value:0.###})";
            }

            _info[id] = new EmotionInfo
            {
                Id = id,
                ZhName = EmotionCatalog.Find(id)!.ZhName,
                EnName = EmotionCatalog.Find(id)!.EnName,
                IsBaseline = false,
                Readiness = readiness,
                Reason = reason,
                SampleCount = samples,
                Consistency = consistency,
            };
        }

        private void MarkUnavailable(string id, string reason)
        {
            var def = EmotionCatalog.Find(id)!;
            _info[id] = new EmotionInfo
            {
                Id = id,
                ZhName = def.ZhName,
                EnName = def.EnName,
                IsBaseline = false,
                Readiness = EmotionReadiness.Unavailable,
                Reason = reason,
            };
        }

        /// <summary>Read + validate a normalized float32 vector. Rejects wrong dimension, NaN/Inf and zero vectors.</summary>
        private bool TryReadVector(string path, out float[] vec, out string error)
        {
            vec = Array.Empty<float>();
            error = string.Empty;
            try
            {
                if (!File.Exists(path))
                {
                    error = $"file not found: {Path.GetFileName(path)}";
                    return false;
                }
                var bytes = File.ReadAllBytes(path);
                if (bytes.Length == 0)
                {
                    error = "empty file";
                    return false;
                }
                if (bytes.Length % 4 != 0)
                {
                    error = $"size {bytes.Length} is not a multiple of 4";
                    return false;
                }
                int count = bytes.Length / 4;
                if (count != ExpectedDim)
                {
                    error = $"dimension {count} != {ExpectedDim}";
                    return false;
                }
                var arr = new float[count];
                Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);

                double sum = 0;
                for (int i = 0; i < arr.Length; i++)
                {
                    if (!float.IsFinite(arr[i]))
                    {
                        error = "vector contains NaN or Infinity";
                        return false;
                    }
                    sum += (double)arr[i] * arr[i];
                }
                double norm = Math.Sqrt(sum);
                if (norm < 1e-12)
                {
                    error = "vector is all zeros";
                    return false;
                }
                if (Math.Abs(norm - 1.0) > 1e-3)
                {
                    error = $"vector is not unit-normalized (norm={norm:0.######})";
                    return false;
                }
                if (Dim == 0) Dim = count;
                vec = arr;
                return true;
            }
            catch (Exception ex)
            {
                error = $"read failed: {ex.Message}";
                return false;
            }
        }

        private static string? GetString(JsonElement el, string name) =>
            el.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

        // ====================================================================
        // Lookup
        // ====================================================================

        /// <summary>
        /// Get the normalized τ for the given emotion (null if absent).
        /// Accepts ids, Chinese names and legacy aliases so external callers keep working.
        /// </summary>
        public float[]? GetTau(string emotion)
        {
            if (!EmotionCatalog.TryNormalize(emotion, out var id)) return null;
            return _tau.TryGetValue(id, out var t) ? t : null;
        }

        /// <summary>Resolve any accepted spelling and return its readiness, without throwing.</summary>
        public EmotionInfo? Describe(string? emotion)
        {
            if (!EmotionCatalog.TryNormalize(emotion, out var id)) return null;
            return _info.TryGetValue(id, out var info) ? info : null;
        }

        /// <summary>Snapshot of all 19 emotions with their current readiness, in display order.</summary>
        public IReadOnlyList<EmotionInfo> Catalog()
            => EmotionCatalog.All.Select(d => _info.TryGetValue(d.Id, out var i) ? i : new EmotionInfo
            {
                Id = d.Id,
                ZhName = d.ZhName,
                EnName = d.EnName,
                IsBaseline = d.IsBaseline,
                Readiness = d.IsBaseline ? EmotionReadiness.Baseline : EmotionReadiness.Unavailable,
                Reason = d.IsBaseline ? null : "not loaded",
            }).ToArray();

        /// <summary>
        /// Resolve a request emotion to (canonical id or null for baseline, tau, reason).
        /// Throws for unknown labels; returns a reason instead of throwing for known-but-unusable ones so the
        /// caller can produce a clear 400 response.
        /// </summary>
        public (string? Id, float[]? Tau, string? Error) Resolve(string? emotion)
        {
            if (emotion == null || emotion.Trim().Length == 0)
                return (null, null, null);
            if (!EmotionCatalog.TryNormalize(emotion, out var id))
                return (null, null, $"unknown emotion '{emotion}'");
            if (EmotionCatalog.IsBaseline(id))
                return (null, null, null);
            if (!_tau.TryGetValue(id, out var tau))
            {
                var info = _info.TryGetValue(id, out var i) ? i : null;
                return (null, null,
                    $"emotion '{id}' ({info?.ZhName}) has no usable direction vector" +
                    (string.IsNullOrEmpty(info?.Reason) ? "" : $": {info!.Reason}"));
            }
            return (id, tau, null);
        }

        // ====================================================================
        // Injection
        // ====================================================================

        /// <summary>
        /// Injection formula: x_new = x_base + α·||x_base||·τ̂.
        /// τ̂ is a unit vector (guaranteed at load time).
        /// </summary>
        public static float[] Apply(float[] xBase, float[] tauNormalized, float alpha)
        {
            if (!float.IsFinite(alpha))
                throw new ArgumentOutOfRangeException(nameof(alpha), "emotion alpha must be a finite number");
            if (alpha < 0)
                throw new ArgumentOutOfRangeException(nameof(alpha), "emotion alpha must be >= 0");
            if (tauNormalized.Length != xBase.Length)
                throw new ArgumentException(
                    $"tau dimension {tauNormalized.Length} does not match speaker embedding {xBase.Length}");

            var result = new float[xBase.Length];
            double sum = 0;
            for (int i = 0; i < xBase.Length; i++) sum += (double)xBase[i] * xBase[i];
            float scale = alpha * (float)Math.Sqrt(sum);

            for (int i = 0; i < xBase.Length; i++)
                result[i] = xBase[i] + scale * tauNormalized[i];
            return result;
        }
    }
}