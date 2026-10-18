namespace SolTTS.Web.Models
{
    public class TtsRequest
    {
        /// <summary>
        /// Text to synthesize.
        /// </summary>
        public string Text { get; set; } = "";

        /// <summary>
        /// Voice ID. Uses the default voice if not specified.
        /// </summary>
        public string? AvatarId { get; set; }

        /// <summary>
        /// Reference audio ID. Uses the voice's default reference audio if not specified.
        /// </summary>
        public string? ReferenceId { get; set; }

        /// <summary>
        /// Emotion ID (optional, e.g. "happy"/"angry"/"sad"/"surprise"/"fearful"/"playfulness").
        /// Requires global UseEmotionControl=true and an existing τ file. Unspecified or "neutral" means no injection.
        /// </summary>
        public string? EmotionId { get; set; }

        /// <summary>
        /// Emotion intensity α (0~1.5). Uses the global EmotionAlpha if not specified (default 0.3).
        /// </summary>
        public float? EmotionAlpha { get; set; }

        /// <summary>
        /// Speech rate (0.5 - 2.0).
        /// </summary>
        public float? Speed { get; set; }

        /// <summary>
        /// Noise scale.
        /// </summary>
        public float? NoiseScale { get; set; }

        /// <summary>
        /// Sampling temperature.
        /// </summary>
        public float? Temperature { get; set; }

        /// <summary>
        /// Top-K sampling.
        /// </summary>
        public int? TopK { get; set; }

        /// <summary>
        /// Streaming chunk size.
        /// </summary>
        public int? StreamingChunkSize { get; set; }

        /// <summary>
        /// G2P priority mode: 0 - dictionary first, 1 - dictionary only, 2 - model first.
        /// </summary>
        public int? G2PPriorityMode { get; set; }

        /// <summary>
        /// List of allowed languages (e.g. ["zh", "en"]).
        /// </summary>
        [System.Text.Json.Serialization.JsonPropertyName("languages")]
        public List<string>? Languages { get; set; }
    }
}
