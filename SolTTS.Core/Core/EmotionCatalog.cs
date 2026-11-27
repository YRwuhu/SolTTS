using System;
using System.Collections.Generic;
using System.Linq;

namespace SolTTS.Core.Core
{
    /// <summary>
    /// Canonical 19-emotion taxonomy shared by the τ loader, HTTP API, WebUI and CLI.
    ///
    /// The order matches the user-facing list. Chinese names are accepted as input aliases, so callers
    /// may send "高兴" or "happy". Legacy CSEMOTIONS ids are kept as compatibility aliases only; they are
    /// never shown as separate choices.
    /// </summary>
    public static class EmotionCatalog
    {
        /// <summary>The neutral baseline. It never has a τ file: requesting it means "no injection".</summary>
        public const string Baseline = "calm";

        private static readonly EmotionDefinition[] Defs =
        {
            new("happy", "高兴", "Happy"),
            new("calm", "平静", "Calm", isBaseline: true),
            new("disgust", "厌恶", "Disgust"),
            new("shy", "害羞", "Shy"),
            new("fearful", "害怕", "Fearful"),
            new("angry", "生气", "Angry"),
            new("serious", "认真", "Serious"),
            new("nervous", "紧张", "Nervous"),
            new("flustered", "慌张", "Flustered"),
            new("confused", "疑惑", "Confused"),
            new("excited", "兴奋", "Excited"),
            new("helpless", "无奈", "Helpless"),
            new("worried", "担心", "Worried"),
            new("surprised", "惊讶", "Surprised"),
            new("crying", "哭泣", "Crying"),
            new("lovestruck", "心动", "Lovestruck"),
            new("embarrassed", "难为情", "Embarrassed"),
            new("confident", "自信", "Confident"),
            new("playful", "调皮", "Playful"),
        };

        private static readonly Dictionary<string, string> Aliases = BuildAliases();

        /// <summary>All 19 definitions in display order.</summary>
        public static IReadOnlyList<EmotionDefinition> All => Defs;

        /// <summary>Emotions other than the baseline, i.e. the ones that need a τ file.</summary>
        public static IReadOnlyList<EmotionDefinition> Directional =>
            Defs.Where(d => !d.IsBaseline).ToArray();

        public static bool IsKnown(string idOrAlias) => TryNormalize(idOrAlias, out _);

        public static bool IsBaseline(string id) =>
            string.Equals(id, Baseline, StringComparison.OrdinalIgnoreCase);

        public static EmotionDefinition? Find(string id)
        {
            foreach (var d in Defs)
                if (string.Equals(d.Id, id, StringComparison.OrdinalIgnoreCase)) return d;
            return null;
        }

        /// <summary>
        /// Map an id, Chinese name, English display name or legacy alias to the canonical id.
        /// Returns false for unknown values; callers must report an error rather than fall back.
        /// </summary>
        public static bool TryNormalize(string? value, out string id)
        {
            id = string.Empty;
            if (value == null) return false;
            var key = value.Trim();
            if (key.Length == 0) return false;
            if (Aliases.TryGetValue(key, out var mapped))
            {
                id = mapped;
                return true;
            }
            var low = key.ToLowerInvariant();
            if (Aliases.TryGetValue(low, out mapped))
            {
                id = mapped;
                return true;
            }
            return false;
        }

        public static string NormalizeOrThrow(string value)
        {
            if (!TryNormalize(value, out var id))
                throw new ArgumentException(
                    $"Unknown emotion '{value}'. Supported: " + string.Join(", ",
                        Defs.Select(d => $"{d.Id}({d.ZhName})")));
            return id;
        }

        /// <summary>
        /// Parse an emotion expression that may be either a single label ("happy", "高兴")
        /// or a weighted blend ("happy+shy", "happy:0.6,shy:0.4", "高兴:0.7,害羞:0.3").
        /// Returns false with an error message if any term is unknown or has an invalid weight.
        /// </summary>
        public static bool TryParseBlend(
            string? value,
            out List<(string Id, float Weight)> components,
            out string canonicalExpr,
            out string? error)
        {
            components = new List<(string Id, float Weight)>();
            canonicalExpr = string.Empty;
            error = null;

            if (string.IsNullOrWhiteSpace(value))
                return true;

            var trimmed = value.Trim();
            // Fast path: single label (no blend separators)
            if (trimmed.IndexOfAny(new[] { '+', ',', ':' }) < 0)
            {
                if (!TryNormalize(trimmed, out var singleId))
                {
                    error = $"unknown emotion '{value}'";
                    return false;
                }
                if (!IsBaseline(singleId))
                {
                    components.Add((singleId, 1f));
                    canonicalExpr = singleId;
                }
                return true;
            }

            var tokens = trimmed.Split(new[] { '+', ',' }, StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length == 0)
            {
                error = $"unknown emotion '{value}'";
                return false;
            }

            var merged = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);
            var order = new List<string>();
            foreach (var tok in tokens)
            {
                string labelPart = tok;
                float weight = 1f;
                int colon = tok.IndexOf(':');
                if (colon >= 0)
                {
                    labelPart = tok[..colon].Trim();
                    var wStr = tok[(colon + 1)..].Trim();
                    if (!float.TryParse(wStr, System.Globalization.NumberStyles.Float,
                            System.Globalization.CultureInfo.InvariantCulture, out weight)
                        || !float.IsFinite(weight) || weight <= 0f)
                    {
                        error = $"invalid weight '{wStr}' in emotion blend '{value}' (must be > 0)";
                        return false;
                    }
                }

                if (!TryNormalize(labelPart, out var id))
                {
                    error = $"unknown emotion '{labelPart}' in '{value}'";
                    return false;
                }

                if (IsBaseline(id))
                    continue; // Baseline contributes zero direction vector

                if (merged.ContainsKey(id))
                    merged[id] += weight;
                else
                {
                    merged[id] = weight;
                    order.Add(id);
                }
            }

            if (order.Count == 0)
            {
                // All terms were baseline (e.g. "calm+neutral")
                return true;
            }

            foreach (var id in order)
                components.Add((id, merged[id]));

            canonicalExpr = components.Count == 1
                ? components[0].Id
                : string.Join("+", components.Select(c =>
                    Math.Abs(c.Weight - 1f) < 1e-4f
                        ? c.Id
                        : $"{c.Id}:{c.Weight.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture)}"));
            return true;
        }

        /// <summary>
        /// Recommended speaking-rate multiplier for an emotion, compensating for the fact that
        /// V1 T2S does not receive speaker embedding directly.
        /// </summary>
        public static float GetRecommendedSpeedFactor(string canonicalId) => canonicalId.ToLowerInvariant() switch
        {
            "excited" or "flustered" => 1.06f,
            "happy" or "playful" or "angry" => 1.03f,
            "crying" or "helpless" or "worried" or "fearful" => 0.95f,
            "shy" or "embarrassed" or "nervous" => 0.97f,
            _ => 1.00f,
        };

        private static Dictionary<string, string> BuildAliases()
        {
            var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var d in Defs)
            {
                map[d.Id] = d.Id;
                map[d.ZhName] = d.Id;
                map[d.EnName] = d.Id;
            }

            // Legacy / external spellings kept for backward compatibility.
            Add("neutral", "calm");
            Add("none", "calm");
            Add("off", "calm");
            Add("sad", "crying");
            Add("sadness", "crying");
            Add("fear", "fearful");
            Add("surprise", "surprised");
            Add("playfulness", "playful");
            Add("joy", "happy");
            Add("disgusted", "disgust");
            Add("nervousness", "nervous");
            Add("confusion", "confused");
            Add("excitement", "excited");
            Add("helplessness", "helpless");
            Add("worry", "worried");
            Add("shyness", "shy");
            Add("embarrassment", "embarrassed");
            Add("love", "lovestruck");
            Add("calmness", "calm");
            return map;

            void Add(string alias, string target) => map[alias] = target;
        }
    }

    /// <summary>Static definition of a single emotion label.</summary>
    public sealed class EmotionDefinition
    {
        public string Id { get; }
        public string ZhName { get; }
        public string EnName { get; }
        public bool IsBaseline { get; }

        public EmotionDefinition(string id, string zhName, string enName, bool isBaseline = false)
        {
            Id = id;
            ZhName = zhName;
            EnName = enName;
            IsBaseline = isBaseline;
        }
    }

    /// <summary>How usable an emotion is at inference time.</summary>
    public enum EmotionReadiness
    {
        /// <summary>Neutral baseline; always usable, no vector needed.</summary>
        Baseline = 0,
        /// <summary>τ loaded, validated and passed the extraction quality gate.</summary>
        Ready = 1,
        /// <summary>τ loaded but the extraction reported low sample count / confidence; usable but flagged.</summary>
        Experimental = 2,
        /// <summary>No usable τ (missing file, invalid vector or no training samples).</summary>
        Unavailable = 3,
    }

    /// <summary>Public, serializable view of one emotion for the HTTP API and UI.</summary>
    public sealed class EmotionInfo
    {
        public string Id { get; set; } = string.Empty;
        public string ZhName { get; set; } = string.Empty;
        public string EnName { get; set; } = string.Empty;
        public bool IsBaseline { get; set; }
        public EmotionReadiness Readiness { get; set; }
        public string? Reason { get; set; }
        public int? SampleCount { get; set; }
        public double? Consistency { get; set; }
    }
}