using SolTTS.Core.Config;
using SolTTS.Core.Core;

namespace SolTTS.Web.Services
{
    /// <summary>
    /// Shared request validation for the TTS endpoints.
    ///
    /// Emotion validation happens at the HTTP boundary so that an unknown label or an unusable direction
    /// yields a deterministic 400 before any audio is generated, instead of depending on the engine's
    /// internal state or silently producing neutral speech.
    /// </summary>
    public static class TtsRequestValidator
    {
        /// <summary>Maximum accepted α. Mirrors TtsEngineV1.MaxEmotionAlpha.</summary>
        public const float MaxEmotionAlpha = 1.5f;

        public readonly record struct EmotionValidation(string? EmotionId);

        /// <summary>
        /// Validate an emotion request against the shared catalog and the loaded vectors.
        /// Returns the canonical id (null for baseline/omitted) or throws ArgumentException.
        /// </summary>
        /// <param name="config">Current configuration (for the enable flag).</param>
        /// <param name="catalog">Readiness snapshot from the engine, or null when the engine has no emotion support.</param>
        /// <param name="emotionId">Raw id/Chinese name/legacy alias.</param>
        /// <param name="emotionAlpha">Optional intensity.</param>
        public static EmotionValidation ValidateEmotion(
            TTSConfig config,
            IReadOnlyList<EmotionInfo>? catalog,
            string? emotionId,
            float? emotionAlpha)
        {
            ValidateAlpha(emotionAlpha);

            if (string.IsNullOrWhiteSpace(emotionId))
                return new EmotionValidation(null);

            if (!EmotionCatalog.TryParseBlend(emotionId, out var components, out var canonicalExpr, out var parseErr))
            {
                throw new ArgumentException(
                    $"{parseErr ?? $"Unknown emotion '{emotionId}'"}. Supported ids: " +
                    string.Join(", ", EmotionCatalog.All.Select(e => $"{e.Id}({e.ZhName})")));
            }

            if (components.Count == 0)
                return new EmotionValidation(null);

            if (catalog == null)
                throw new ArgumentException(
                    "Emotion control is not available for the current engine. " +
                    "The V2 engine has no emotion support; set UseEngineV2=false to use V1 emotion control.");

            if (!config.UseEmotionControl)
                throw new ArgumentException(
                    "Emotion control is disabled (UseEmotionControl=false); enable it or omit emotionId.");

            foreach (var (compId, _) in components)
            {
                var info = catalog.FirstOrDefault(e =>
                    string.Equals(e.Id, compId, StringComparison.OrdinalIgnoreCase));

                if (info == null || info.Readiness == EmotionReadiness.Unavailable)
                {
                    var reason = info?.Reason;
                    throw new ArgumentException(
                        $"Emotion '{compId}' ({info?.ZhName}) is not usable: " +
                        (string.IsNullOrEmpty(reason) ? "no direction vector is loaded" : reason));
                }
            }

            return new EmotionValidation(canonicalExpr);
        }

        /// <summary>Reject non-finite, negative or out-of-range α.</summary>
        public static void ValidateAlpha(float? emotionAlpha)
        {
            if (!emotionAlpha.HasValue) return;
            var a = emotionAlpha.Value;
            if (!float.IsFinite(a))
                throw new ArgumentException("emotionAlpha must be a finite number");
            if (a < 0f)
                throw new ArgumentException("emotionAlpha must be >= 0");
            if (a > MaxEmotionAlpha)
                throw new ArgumentException($"emotionAlpha must be <= {MaxEmotionAlpha}");
        }
    }
}