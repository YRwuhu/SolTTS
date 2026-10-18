using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace SolTTS.Core.Core
{
    /// <summary>
    /// TTS inference engine interface defining the standard inference flow.
    /// </summary>
    public interface ITtsEngine : IDisposable
    {
        /// <summary>
        /// Engine output sample rate (Hz)
        /// </summary>
        int SamplingRate { get; }

        /// <summary>
        /// Asynchronously load the models and resources required by the engine.
        /// </summary>
        /// <param name="config">Global TTS configuration object.</param>
        Task LoadAsync(Config.TTSConfig config);

        /// <summary>
        /// Run asynchronous inference and return the complete audio data with metadata.
        /// </summary>
        /// <param name="text">Text to synthesize.</param>
        /// <param name="options">Inference options (speed, noise scale, etc.).</param>
        /// <returns>A TtsResult containing the audio sample array and metadata.</returns>
        Task<TtsResult> PredictAsync(string text, TtsOptions options);

        /// <summary>
        /// Run streaming inference, continuously returning audio chunks.
        /// </summary>
        /// <param name="text">Text to synthesize.</param>
        /// <param name="options">Inference options.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>An asynchronous stream of audio chunks.</returns>
        IAsyncEnumerable<float[]> PredictStreamAsync(string text, TtsOptions options, CancellationToken cancellationToken = default);
    }

    /// <summary>
    /// TTS inference result.
    /// </summary>
    public class TtsResult
    {
        public float[] Audio { get; set; } = Array.Empty<float>();
        public int TokenCount { get; set; }
    }

    /// <summary>
    /// Configurable options for inference.
    /// </summary>
    public class TtsOptions
    {
        /// <summary>
        /// Speech rate (0.5 - 2.0).
        /// </summary>
        public float Speed { get; set; } = 1.0f;

        /// <summary>
        /// Noise scale affecting timbre expressiveness.
        /// </summary>
        public float NoiseScale { get; set; } = 0.35f;

        /// <summary>
        /// Sampling temperature affecting generation diversity.
        /// </summary>
        public float Temperature { get; set; } = 1.0f;

        /// <summary>
        /// Top-K sampling parameter.
        /// </summary>
        public int TopK { get; set; } = 15;

        /// <summary>
        /// Streaming chunk size. For the V1 engine this is the T2S sampling interval; for V2 it is the number of tokens generated per inference.
        /// </summary>
        public int StreamingChunkSize { get; set; } = 24;

        /// <summary>
        /// Growth factor for the streaming vocoder retrigger threshold. The vocoder recomputes all accumulated tokens,
        /// so a higher factor reduces total recompute (less pressure on long text) at the cost of larger chunk latency.
        /// </summary>
        public float VocoderRetriggerFactor { get; set; } = 4.0f;

        /// <summary>
        /// Requested voice ID (optional). When set, the engine switches to that voice before inference.
        /// </summary>
        public string? AvatarId { get; set; }

        /// <summary>
        /// Requested reference audio ID (optional). When set, the engine uses that reference audio.
        /// </summary>
        public string? ReferenceId { get; set; }

        /// <summary>
        /// Emotion ID (optional, e.g. "happy"/"angry"/"sad"). When set, the corresponding emotion direction vector τ
        /// is injected into the speaker embedding before using the reference audio. Requires the global UseEmotionControl=true
        /// and the τ files to exist; otherwise it is silently ignored.
        /// </summary>
        public string? EmotionId { get; set; }

        /// <summary>
        /// Emotion strength α (optional). Overrides the global EmotionAlpha.
        /// Injection formula: x_new = x_base + α·||x_base||·τ̂.
        /// </summary>
        public float? EmotionAlpha { get; set; }

        /// <summary>
        /// G2P priority mode (optional): 0-dictionary first, 1-dictionary only, 2-model first.
        /// Uses the global configuration when not set.
        /// </summary>
        public int? G2PPriorityMode { get; set; }

        /// <summary>
        /// Maximum number of retries. The engine regenerates when the generated token count is insufficient.
        /// Uses the global configuration when not set.
        /// </summary>
        public int? MaxRetries { get; set; }

        /// <summary>
        /// Chinese minimum expected token multiplier. Higher values increase the retry probability.
        /// Uses the global configuration when not set.
        /// </summary>
        public float? MinTokenMultiplierChinese { get; set; }

        /// <summary>
        /// Japanese minimum expected token multiplier.
        /// Uses the global configuration when not set.
        /// </summary>
        public float? MinTokenMultiplierJapanese { get; set; }

        /// <summary>
        /// English minimum expected token multiplier.
        /// Uses the global configuration when not set.
        /// </summary>
        public float? MinTokenMultiplierEnglish { get; set; }

        /// <summary>
        /// Punctuation/other minimum expected token multiplier.
        /// Uses the global configuration when not set.
        /// </summary>
        public float? MinTokenMultiplierOther { get; set; }

        /// <summary>
        /// Debug mode switch.
        /// Uses the global configuration when not set.
        /// </summary>
        public bool? DebugMode { get; set; }

        /// <summary>
        /// Allowed language list (e.g. ["zh", "en"]).
        /// When set, overrides the language restrictions in the global configuration.
        /// </summary>
        public List<string>? Languages { get; set; }

        /// <summary>
        /// Default options.
        /// </summary>
        public static TtsOptions Default => new TtsOptions();
    }
}
