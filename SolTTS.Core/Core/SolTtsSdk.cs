using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using SolTTS.Core.Config;

namespace SolTTS.Core.Core
{
    /// <summary>
    /// SolTTS.Core SDK core class providing the high-level API.
    /// </summary>
    public class SolTtsSdk : IDisposable
    {
        private TtsEnginePool _pool;
        private TTSConfig _config;
        private readonly object _lock = new();

        /// <summary>
        /// Get the sample rate.
        /// </summary>
        public int SamplingRate => _pool.SamplingRate;

        /// <summary>
        /// Get the currently active configuration.
        /// </summary>
        public TTSConfig Config => _config;

        /// <summary>
        /// Get all available voices.
        /// </summary>
        public IReadOnlyList<Avatar> Avatars => _config.Avatars;

        /// <summary>
        /// Initialize the SDK.
        /// </summary>
        /// <param name="config">Configuration object. When null, attempts to load the default configuration.</param>
        public SolTtsSdk(TTSConfig? config = null)
        {
            _config = config ?? TTSConfig.Load();
            _pool = new TtsEnginePool(_config);
        }

        /// <summary>
        /// Asynchronously initialize the engine.
        /// </summary>
        public async Task InitializeAsync()
        {
            await _pool.InitializeAsync();
        }

        /// <summary>
        /// Reload the configuration and reinitialize the engine.
        /// </summary>
        public async Task ReloadConfigAsync()
        {
            var newConfig = TTSConfig.Reload();
            await ApplyConfigAsync(newConfig);
            Console.WriteLine("[SolTTS] Configuration reloaded successfully.");
        }

        /// <summary>
        /// Apply the current (or specified) configuration to the engine directly.
        /// Does not reload the configuration file from disk.
        /// </summary>
        public async Task ApplyConfigAsync(TTSConfig? config = null)
        {
            var targetConfig = config ?? _config;
            var newPool = new TtsEnginePool(targetConfig);
            await newPool.InitializeAsync();

            lock (_lock)
            {
                var oldPool = _pool;
                _pool = newPool;
                _config = targetConfig;
                oldPool.Dispose();
            }
        }

        /// <summary>
        /// Get the voice with the specified ID.
        /// </summary>
        public Avatar? GetAvatar(string? avatarId)
        {
            if (string.IsNullOrEmpty(avatarId))
                avatarId = _config.DefaultAvatarId;

            return _config.Avatars.Find(a => a.Id == avatarId);
        }

        /// <summary>
        /// Synthesize the full audio.
        /// </summary>
        /// <param name="text">Text to synthesize</param>
        /// <param name="options">Inference options. Defaults from the configuration when null.</param>
        /// <param name="avatarId">Voice ID (optional).</param>
        /// <param name="referenceId">Reference audio ID (optional).</param>
        /// <returns>A TtsResult containing the audio data and metadata</returns>
        public async Task<TtsResult> PredictAsync(string text, TtsOptions? options = null, string? avatarId = null, string? referenceId = null)
        {
            var opt = options ?? GetDefaultOptions();
            opt.AvatarId = avatarId ?? _config.DefaultAvatarId;
            opt.ReferenceId = referenceId;
            opt.Languages ??= _config.G2P.Languages != null ? new List<string>(_config.G2P.Languages) : null;

            using var lease = await _pool.LeaseAsync();
            return await lease.Engine.PredictAsync(text, opt);
        }

        /// <summary>
        /// Synthesize audio in streaming mode.
        /// </summary>
        /// <param name="text">Text to synthesize</param>
        /// <param name="options">Inference options. Defaults from the configuration when null.</param>
        /// <param name="avatarId">Voice ID (optional).</param>
        /// <param name="referenceId">Reference audio ID (optional).</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>Asynchronous stream of audio chunks</returns>
        public async IAsyncEnumerable<float[]> PredictStreamAsync(string text, TtsOptions? options = null, string? avatarId = null, string? referenceId = null, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var opt = options ?? GetDefaultOptions();
            opt.AvatarId = avatarId ?? _config.DefaultAvatarId;
            opt.ReferenceId = referenceId;
            opt.Languages ??= _config.G2P.Languages != null ? new List<string>(_config.G2P.Languages) : null;

            using var lease = await _pool.LeaseAsync(cancellationToken);
            await foreach (var chunk in lease.Engine.PredictStreamAsync(text, opt, cancellationToken))
            {
                yield return chunk;
            }
        }

        private TtsOptions GetDefaultOptions()
        {
            return new TtsOptions
            {
                Speed = _config.Speed,
                NoiseScale = _config.NoiseScale,
                Temperature = _config.Temperature,
                TopK = _config.TopK,
                StreamingChunkSize = _config.StreamingChunkSize,
                VocoderRetriggerFactor = _config.VocoderRetriggerFactor,
                Languages = _config.G2P.Languages != null ? new List<string>(_config.G2P.Languages) : null
            };
        }

        public void Dispose()
        {
            _pool.Dispose();
        }
    }
}
