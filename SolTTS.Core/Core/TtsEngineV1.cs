using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using SolTTS.Core.Config;
using SolTTS.Core.Frontend.BERT;
using SolTTS.Core.Frontend.G2P;
using SolTTS.Core.Frontend.G2P.Common;
using SolTTS.Core.Frontend.G2P.Chinese;
using SolTTS.Core.Frontend.G2P.English;
using SolTTS.Core.Frontend.G2P.Japanese;
using SolTTS.Core.Frontend.TextNorm;
using SolTTS.Core.Utils;
using System.Buffers;
using System.Runtime.InteropServices;

namespace SolTTS.Core.Core
{
    /// <summary>
    /// V1 inference engine implementation (based on the Genie-TTS architecture).
    /// </summary>
    public class TtsEngineV1 : ITtsEngine
    {
        private readonly InferenceEngineV1 _engine = new();
        private RobertaFeatureExtractor? _bert;
        private ChineseG2P? _chineseG2p;
        private EnglishG2P? _englishG2p;
        private JapaneseG2P? _japaneseG2p;
        private MixedLanguageG2P? _mixedG2p;

        private float[] _refAudio16k = Array.Empty<float>();
        private float[] _refAudio32k = Array.Empty<float>();
        private float[] _sslContent = Array.Empty<float>();
        private float[] _svEmb = Array.Empty<float>();
        private float[] _ge = Array.Empty<float>();
        private float[] _geAdvanced = Array.Empty<float>();

        private long[] _refPhoneIds = Array.Empty<long>();
        private float[,] _refBert = new float[0, 1024]; // Preallocate 1024 columns to avoid ONNX dimension errors

        private TTSConfig? _config;

        // Emotion control (emotion guidance without reference audio)
        private EmotionDirectionController? _emotion;

        // Voice switching state tracking
        private string? _currentAvatarId;
        private string? _currentReferenceId;
        // Currently injected emotion state (used to decide whether ge/ge_advanced must be recomputed)
        private string? _currentEmotionId;
        private float _currentEmotionAlpha;
        private readonly object _referenceLock = new();

        public int SamplingRate => 32000;

        public async Task LoadAsync(TTSConfig config)
        {
            _config = config;

            // Validate the V1 model directory
            if (!Directory.Exists(config.V1TtsDir))
            {
                throw new DirectoryNotFoundException(
                    $"V1 TTS model directory not found: '{config.V1TtsDir}'. " +
                    $"Ensure resources/models_v1/tts exists, or set UseEngineV2 to true to use the V2 engine.");
            }
            // Load resources concurrently
            await Task.Run(() =>
            {
                var opt = _engine.GetSessionOptions(config);
                Parallel.Invoke(
                    () => _engine.LoadModels(config.V1TtsDir, config, config.HubertPath, config.SpeakerEncoderPath),
                    () =>
                    {
                        if (File.Exists(config.EffectiveBertModelPath) && File.Exists(config.TokenizerJsonPath))
                        {
                            _bert = config.UseBertGgml
                                ? RobertaFeatureExtractor.CreateWithGpuFallback(
                                    config.EffectiveBertModelPath, config.TokenizerJsonPath, opt, config.UseBertCache,
                                    () => GgmlExecutionProvider.TryCreateSessionOptions(device: "gpu", gpuMinWeightMb: 0, partial: false))
                                : new RobertaFeatureExtractor(config.EffectiveBertModelPath, config.TokenizerJsonPath, opt, config.UseBertCache);
                        }
                    },
                    () => _chineseG2p = new ChineseG2P(config.ChineseG2PDict, config.PinyinDict, config.CustomDictFullPath ?? "", config.PolyphonicJson, config.JiebaDictDir),
                    () => _englishG2p = new EnglishG2P(config.CmuDict, config.NeuralG2PModel, config.CustomDictFullPath,
                        config.EnglishPosTaggerDir, config.EnglishWordSegmentDir, config.EnglishSpecialDict, config.G2P.PriorityMode),
                    () =>
                    {
                        // Japanese G2P (optional - loaded when the dictionary directory exists)
                        if (Directory.Exists(config.JapaneseDictDir))
                        {
                            _japaneseG2p = new JapaneseG2P(config.JapaneseDictDir, config.CustomDictFullPath);
                        }
                        else
                        {
                            Console.WriteLine($"[TtsEngineV1] Warning: Japanese dictionary not found at: {config.JapaneseDictDir}");
                        }
                    }
                );
            });

            if (_chineseG2p == null || _englishG2p == null)
                throw new Exception("G2P core initialization failed");

            _englishG2p.DebugMode = config.DebugMode;
            if (_japaneseG2p != null) _japaneseG2p.DebugMode = config.DebugMode;
            if (_bert != null)
            {
                _bert.DebugMode = config.DebugMode;
            }

            _mixedG2p = new MixedLanguageG2P(_chineseG2p, _englishG2p, _japaneseG2p, _config.G2P.Languages);
            _mixedG2p.DebugMode = config.DebugMode;

            // Emotion direction vectors (optional; Loaded=false when files are missing, feature silently disabled)
            if (config.UseEmotionControl)
            {
                _emotion = new EmotionDirectionController(config.EffectiveEmotionVectorPath);
                if (_emotion.Loaded)
                    Console.WriteLine($"[Emotion] Loaded {_emotion.Emotions.Count} emotion direction vectors (dim={_emotion.Dim}): {string.Join(", ", _emotion.Emotions)}");
                else
                    Console.WriteLine($"[Emotion] Warning: emotion direction vectors not found, emotion control disabled (path={config.EffectiveEmotionVectorPath})");
            }

            // Preprocess reference audio (may involve model inference; keep inside Task.Run or run after)
            await Task.Run(PrepareReference);
        }

        /// <summary>
        /// Share core resources from another engine instance (used for memory-reuse pooling mode).
        /// </summary>
        public void ShareResourcesFrom(TtsEngineV1 other)
        {
            _engine.ShareModelsFrom(other._engine);
            _bert = other._bert;
            _chineseG2p = other._chineseG2p;
            _englishG2p = other._englishG2p;
            _japaneseG2p = other._japaneseG2p;
            _mixedG2p = other._mixedG2p;
            _emotion = other._emotion;
            _config = other._config;
        }


        private void PrepareReference() => PrepareReference(null, null, null, null);

        /// <summary>
        /// "Heavy" preprocessing results derived from the reference audio/voice. These depend only on
        /// the reference audio itself, not on emotion; reuse them across emotion/α changes to avoid re-inference.
        /// </summary>
        private sealed class ReferenceCache
        {
            public string? AvatarId;
            public string? ReferenceId;
            public string RefPath = string.Empty;
            public float[] RefAudio16k = Array.Empty<float>();
            public float[] RefAudio32k = Array.Empty<float>();
            public float[] SslContent = Array.Empty<float>();
            public float[] SvEmbBase = Array.Empty<float>();   // Original speaker embedding without injection
            public long[] RefPhoneIds = Array.Empty<long>();
            public float[,] RefBert = new float[0, 1024];
            public bool Valid;
        }

        private ReferenceCache? _refCache;

        /// <summary>
        /// Update only the emotion-related conditions (inject sv_emb → recompute ge/ge_advanced).
        /// Reuses the reference-audio cache, avoiding repeated HuBERT / speaker_encoder / G2P / BERT work.
        /// </summary>
        private void ApplyEmotionOnly(string? emotionId, float? emotionAlpha)
        {
            if (_config == null || _refCache == null || !_refCache.Valid) return;

            float effectiveAlpha = 0f;
            var baseEmb = _refCache.SvEmbBase;

            if (_emotion != null && _emotion.Loaded && !string.IsNullOrEmpty(emotionId)
                && !string.Equals(emotionId, EmotionDirectionController.NeutralEmotion, StringComparison.OrdinalIgnoreCase))
            {
                var tau = _emotion.GetTau(emotionId);
                if (tau != null && tau.Length == baseEmb.Length)
                {
                    effectiveAlpha = emotionAlpha ?? _config.EmotionAlpha;
                    if (effectiveAlpha > 0f)
                    {
                        _svEmb = EmotionDirectionController.Apply(baseEmb, tau, effectiveAlpha);
                        Console.WriteLine($"[Emotion] Injected {emotionId}: α={effectiveAlpha:0.###} (lightweight recompute)");
                    }
                }
            }

            if (effectiveAlpha <= 0f)
            {
                // No emotion: reuse the original embedding to avoid extra allocations
                _svEmb = baseEmb;
            }

            var (ge, geAdvanced) = _engine.GetPromptEmbedding(_refAudio32k, _svEmb);
            _ge = ge;
            _geAdvanced = geAdvanced;
        }

        private void PrepareReference(string? avatarId, string? referenceId, string? emotionId = null, float? emotionAlpha = null)
        {
            if (_config == null) return;

            // Resolve target voice
            avatarId ??= _config.DefaultAvatarId;
            var avatar = _config.Avatars.Find(a => a.Id == avatarId);

            string refPath;
            string refText;
            string? refLanguage;
            string resolvedRefId = "default";

            if (avatar != null)
            {
                var reference = avatar.GetReference(referenceId);
                if (reference != null)
                {
                    refPath = reference.GetFullAudioPath(_config.AvatarsDir, avatarId);
                    refText = reference.Text;
                    refLanguage = reference.Language;
                    resolvedRefId = reference.Id ?? "default";
                }
                else
                {
                    Console.WriteLine($"Warning: reference audio '{referenceId}' not found for voice '{avatarId}'.");
                    return;
                }
            }
            else
            {
                Console.WriteLine("Warning: no reference audio configured. Add an entry under Avatars.");
                return;
            }

            // Auto-fill the language tag
            if (string.IsNullOrEmpty(refLanguage))
            {
                bool hasJaHints = refText.Any(c => (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF) || c == 0x30FC || c == 0x300C || c == 0x300D || c == 0x30FB || c == 0x3005);
                if (hasJaHints) refLanguage = "ja";
                else
                {
                    var mode = LanguageDetector.DetectMode(refText);
                    if (mode == LanguageDetector.LanguageMode.Japanese) refLanguage = "ja";
                }
            }

            if (!File.Exists(refPath))
            {
                throw new FileNotFoundException($"Reference audio file not found: {refPath}. Check the Avatars configuration in config.json.");
            }

            _refAudio16k = AudioHelper.ReadWav(refPath, 16000);
            _refAudio32k = AudioHelper.ReadWav(refPath, 32000);

            _sslContent = _engine.GetHubertContent(_refAudio16k);
            var svEmbBase = _engine.GetSpeakerEmbedding(_refAudio16k);

            // Emotion guidance (optional): inject an emotion direction vector into sv_emb space,
            // x_new = x_base + α·||x_base||·τ̂
            // When α=0 or the feature is disabled/unloaded, the original embedding is kept (output bit-identical).
            float effectiveAlpha = 0f;
            _svEmb = svEmbBase;
            if (_emotion != null && _emotion.Loaded && !string.IsNullOrEmpty(emotionId)
                && !string.Equals(emotionId, EmotionDirectionController.NeutralEmotion, StringComparison.OrdinalIgnoreCase))
            {
                var tau = _emotion.GetTau(emotionId);
                if (tau != null && tau.Length == svEmbBase.Length)
                {
                    effectiveAlpha = emotionAlpha ?? _config?.EmotionAlpha ?? 0.3f;
                    if (effectiveAlpha > 0f)
                    {
                        _svEmb = EmotionDirectionController.Apply(svEmbBase, tau, effectiveAlpha);
                        Console.WriteLine($"[Emotion] Injected {emotionId}: α={effectiveAlpha:0.###}, ||x_base||={Math.Sqrt(svEmbBase.Sum(v => (double)v * v)):0.##}");
                    }
                }
                else if (tau != null)
                {
                    Console.WriteLine($"[Emotion] Ignoring {emotionId}: τ dimension {tau.Length} does not match sv_emb dimension {svEmbBase.Length}");
                }
                else if (_config?.DebugMode == true)
                {
                    Console.WriteLine($"[Emotion] τ not found for emotion '{emotionId}', ignoring");
                }
            }

            if (string.IsNullOrEmpty(emotionId) || effectiveAlpha <= 0f)
            {
                emotionId = null;
                effectiveAlpha = 0f;
            }

            var (ge, geAdvanced) = _engine.GetPromptEmbedding(_refAudio32k, _svEmb);
            _ge = ge;
            _geAdvanced = geAdvanced;

            // Cache heavy reference-audio results (reused when emotion/α changes)
            var cache = new ReferenceCache
            {
                AvatarId = avatarId,
                ReferenceId = referenceId,
                RefPath = refPath,
                RefAudio16k = _refAudio16k,
                RefAudio32k = _refAudio32k,
                SslContent = _sslContent,
                SvEmbBase = svEmbBase,
                Valid = false,   // Set to true after G2P/BERT complete
            };

            // Run G2P using the reference audio's configured language
            // If that language is not in the global allow-list, create a temporary G2P that includes it
            MixedLanguageG2P refG2p = _mixedG2p!;
            if (!string.IsNullOrEmpty(refLanguage))
            {
                var refLangs = new List<string>(_config.G2P.Languages ?? new List<string> { "zh", "en" });
                var lower = refLanguage.ToLowerInvariant();
                if (!refLangs.Contains(lower) && (lower == "ja" || lower == "jp"))
                {
                    refLangs.Add("ja");
                    refG2p = new MixedLanguageG2P(_chineseG2p!, _englishG2p!, _japaneseG2p, refLangs);
                    refG2p.DebugMode = _config.DebugMode;
                }
                else if (!refLangs.Contains(lower) && lower == "en")
                {
                    refLangs.Add("en");
                    refG2p = new MixedLanguageG2P(_chineseG2p!, _englishG2p!, _japaneseG2p, refLangs);
                    refG2p.DebugMode = _config.DebugMode;
                }
            }
            else
            {
                // Reference audio has no language set → auto-detect: temporarily allow Japanese if the text contains kana
                bool hasKana = refText.Any(c => (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF));
                if (hasKana)
                {
                    refLanguage = "ja"; // Inferred and assigned for the later BERT check
                    var refLangs = new List<string>(_config.G2P.Languages ?? new List<string> { "zh", "en" });
                    if (!refLangs.Contains("ja")) refLangs.Add("ja");
                    refG2p = new MixedLanguageG2P(_chineseG2p!, _englishG2p!, _japaneseG2p, refLangs);
                    refG2p.DebugMode = _config.DebugMode;
                }
            }
            var res = refG2p.Process(refText, refLanguage);
            _refPhoneIds = res.PhoneIds;

            if (_bert != null)
            {
                // Use the unified BERT extraction logic so reference audio and synthesized text behave identically
                float[] pooledBert = ExtractBertOptimal(res);
                try
                {
                    // Convert the auxiliary 1D array into a persistent 2D array
                    _refBert = new float[res.PhoneIds.Length, 1024];
                    Buffer.BlockCopy(pooledBert, 0, _refBert, 0, res.PhoneIds.Length * 1024 * sizeof(float));
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(pooledBert);
                }
            }
            else
            {
                _refBert = new float[_refPhoneIds.Length, 1024];
            }

            if (_sslContent == null || _sslContent.Length == 0)
                throw new InvalidOperationException("Failed to extract SSL features from the reference audio. Ensure the hubert model files are complete and the audio is readable.");

            if (_refPhoneIds == null || _refPhoneIds.Length == 0)
                throw new InvalidOperationException("Failed to process reference audio text (G2P failure). Check the RefText configuration.");

            // All reference-audio processing succeeded → cache is complete for reuse across emotion/α switches
            cache.RefPhoneIds = _refPhoneIds;
            cache.RefBert = _refBert;
            cache.Valid = true;
            _refCache = cache;

            // Sync state: avoid repeating the full preprocessing on the first request after LoadAsync
            _currentAvatarId = avatarId;
            _currentReferenceId = resolvedRefId;
            _currentEmotionId = emotionId;
            _currentEmotionAlpha = effectiveAlpha;
        }

        public async Task<TtsResult> PredictAsync(string text, TtsOptions options)
        {
            // Detect voice switch
            EnsureReferenceLoaded(options.AvatarId, options.ReferenceId, options.EmotionId, options.EmotionAlpha);

            // 1. Detect the overall language tendency to give sentence processing a global context
            var overallMode = LanguageDetector.DetectMode(text);
            string? globalLangHint = (overallMode == LanguageDetector.LanguageMode.Japanese) ? "ja" : null;

            // 2. Split into sentences and process one by one
            string normalized = LanguageDetector.NormalizePunctuation(text);
            var sentences = LanguageDetector.SplitSentences(normalized);
            if (sentences.Count == 0) return new TtsResult { Audio = Array.Empty<float>() };

            // 2b. Cross-sentence pipeline: frontend (BERT/G2P) prefetch + vocoder overlaps the next sentence's T2S.
            //     - T2S on CPU (sessions _t2sEncoder/_firstStageDecoder/_stageDecoder)
            //     - Vocoder on GPU (session _vocoder, ggml/Vulkan)
            //     They use different Sessions, so they can safely run in parallel; vocoder calls are serialized
            //     through a task chain (single _vocoder session).
            var contexts = sentences.Select(s => new SentenceContext { Text = s }).ToList();
            foreach (var ctx in contexts)
            {
                ctx.PreprocessTask = Task.Run(() =>
                {
                    var (res, bertFeat) = ProcessFrontendOptimal(ctx.Text, options, globalLangHint);
                    ctx.G2p = res;
                    ctx.BertFeat = bertFeat;
                });
            }

            var allAudio = new List<float>();
            int totalTokens = 0;

            // Vocoder task chain: keeps vocoder calls serialized in sentence order while allowing overlap with the next T2S
            var vocoderChain = Task.CompletedTask;
            var pendingAudio = new List<Task<(float[] audio, int tokens)>>();

            // 2c. Cross-sentence parallel decoding: each sentence's T2S is independent and can run concurrently to
            //     shorten end-to-end time; the vocoder stays serialized via vocoderChain (one Session cannot be shared
            //     concurrently), but each sentence's audio is queued as soon as its T2S finishes, preserving order.
            int degree = Math.Clamp(_config?.ParallelSentenceDegree ?? 1, 1, Math.Max(1, contexts.Count));
            using var decodeGate = new SemaphoreSlim(degree, degree);
            var t2sTasks = new List<Task<(long[] tokens, G2PResult res)>>();

            try
            {
            foreach (var ctx in contexts)
            {
                var captured = ctx;
                t2sTasks.Add(Task.Run(async () =>
                {
                    if (captured.PreprocessTask != null) await captured.PreprocessTask;
                    if (captured.G2p == null || captured.BertFeat == null) return (Array.Empty<long>(), default);

                    G2PResult res = captured.G2p.Value;
                    float[] bertFeat = captured.BertFeat!;
                    try
                    {
                        var bert2D = new float[res.PhoneIds.Length, 1024];
                        Buffer.BlockCopy(bertFeat, 0, bert2D, 0, res.PhoneIds.Length * 1024 * sizeof(float));
                        await decodeGate.WaitAsync();
                        try
                        {
                            var pred = _engine.RunT2S(res.PhoneIds, bert2D, _refPhoneIds, _refBert, _sslContent, res.LanguageTags, options);
                            return (pred, res);
                        }
                        finally
                        {
                            decodeGate.Release();
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(bertFeat);
                        captured.BertFeat = null;
                    }
                }));
            }

            // Wait for T2S in sentence order and queue the vocoder
            for (int sentenceIndex = 0; sentenceIndex < t2sTasks.Count; sentenceIndex++)
            {
                var (predSemantic, res) = await t2sTasks[sentenceIndex];
                if (predSemantic.Length == 0) continue;

                var phoneIds = res.PhoneIds;
                var refAudio = _refAudio32k;
                var ge = _ge;
                var geAdvanced = _geAdvanced;
                var speed = options.Speed;
                var sentenceOrdinal = sentenceIndex;
                var prev = vocoderChain;
                var next = Task.Run(async () =>
                {
                    await prev; // Serialize vocoder calls (one Session cannot run concurrently)
                    var audio = _engine.RunVocoder(phoneIds, predSemantic, refAudio, out int audioLen, ge, geAdvanced, speed);
                    try
                    {
                        var audioData = new float[audioLen];
                        Array.Copy(audio, 0, audioData, 0, audioLen);
                        if (PerfStats.Enabled) PerfStats.AddSentence(sentenceOrdinal, predSemantic.Length, audioLen);
                        return (audioData, predSemantic.Length);
                    }
                    finally
                    {
                        _engine.ReturnAudioBuffer(audio);
                    }
                });
                vocoderChain = next;
                pendingAudio.Add(next);
            }
            }
            finally
            {
                // On the exception path, still await all T2S tasks so none is left in WaitAsync after decodeGate is released.
                foreach (var t in t2sTasks)
                {
                    if (!t.IsCompleted)
                    {
                        try { await t; } catch { /* The first exception is already surfaced by the main flow */ }
                    }
                }
            }

            // Collect audio in sentence order
            foreach (var t in pendingAudio)
            {
                var (audioData, tokens) = await t;
                allAudio.AddRange(audioData);
                totalTokens += tokens;
            }

            return new TtsResult
            {
                Audio = allAudio.ToArray(),
                TokenCount = totalTokens
            };
        }

        /// <summary>
        /// Check whether the currently loaded voice matches the request; reload if not.
        /// </summary>
        private void EnsureReferenceLoaded(string? avatarId, string? referenceId, string? emotionId = null, float? emotionAlpha = null)
        {
            if (_config == null) return;

            // Use the defaults when not specified
            avatarId ??= _config.DefaultAvatarId;

            // Find the Avatar
            var avatar = _config.Avatars.Find(a => a.Id == avatarId);
            if (avatar == null)
            {
                Console.WriteLine($"Warning: voice '{avatarId}' not found, keeping the current voice.");
                return;
            }

            // Find the Reference
            var reference = avatar.GetReference(referenceId);
            string effectiveRefId = reference?.Id ?? "default";
            string refText = reference?.Text ?? "";
            string? refLanguage = reference?.Language;

            // If no language is specified, infer it from text features (fixes Japanese reference text in pure kanji being detected as Chinese)
            if (string.IsNullOrEmpty(refLanguage))
            {
                bool hasJaHints = refText.Any(c => (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF) || c == 0x30FC || c == 0x300C || c == 0x300D || c == 0x3005);
                if (hasJaHints) refLanguage = "ja";
                else
                {
                    var mode = LanguageDetector.DetectMode(refText);
                    if (mode == LanguageDetector.LanguageMode.Japanese) refLanguage = "ja";
                }
            }

            // Normalize emotion: neutral / disabled / missing τ / α<=0 means no emotion
            string? effectiveEmotion = null;
            float effectiveAlpha = 0f;
            if (_emotion != null && _emotion.Loaded && !string.IsNullOrEmpty(emotionId)
                && !string.Equals(emotionId, EmotionDirectionController.NeutralEmotion, StringComparison.OrdinalIgnoreCase))
            {
                var tau = _emotion.GetTau(emotionId);
                if (tau != null)
                {
                    float alpha = emotionAlpha ?? _config.EmotionAlpha;
                    if (alpha > 0f)
                    {
                        // Dimension matching is validated in PrepareReference (on first call _svEmb is not yet loaded)
                        effectiveEmotion = emotionId;
                        effectiveAlpha = alpha;
                    }
                }
            }

            bool sameEmotion = string.Equals(effectiveEmotion, _currentEmotionId, StringComparison.OrdinalIgnoreCase)
                && effectiveAlpha == _currentEmotionAlpha;
            bool sameAvatarRef = avatarId == _currentAvatarId && effectiveRefId == _currentReferenceId;

            // Check whether a reload is needed
            if (sameAvatarRef && sameEmotion)
                return;

            lock (_referenceLock)
            {
                // Double-check
                if (sameAvatarRef && sameEmotion)
                    return;

                // Fast path: when only emotion/α changed, the heavy reference-audio work (HuBERT / speaker_encoder /
                // G2P / BERT) need not be recomputed; only re-inject sv_emb and recompute ge/ge_advanced.
                if (sameAvatarRef && _refCache != null && _refCache.Valid)
                {
                    Console.WriteLine($"🔄 Switching emotion: {avatarId}/{effectiveRefId} → {effectiveEmotion ?? "neutral"}" +
                                      (effectiveEmotion != null ? $" (α={effectiveAlpha:0.###})" : ""));
                    ApplyEmotionOnly(effectiveEmotion, effectiveAlpha > 0f ? effectiveAlpha : null);
                    _currentEmotionId = effectiveEmotion;
                    _currentEmotionAlpha = effectiveAlpha;
                    return;
                }

                Console.WriteLine($"🔄 Switching {(!sameEmotion ? "emotion/voice" : "voice")}: {avatarId}/{effectiveRefId}" +
                                  (effectiveEmotion != null ? $" [emotion={effectiveEmotion}, α={effectiveAlpha:0.###}]" : ""));
                PrepareReference(avatarId, referenceId, effectiveEmotion, effectiveAlpha > 0f ? effectiveAlpha : null);
                _currentAvatarId = avatarId;
                _currentReferenceId = effectiveRefId;
                _currentEmotionId = effectiveEmotion;
                _currentEmotionAlpha = effectiveAlpha;
            }
        }

        internal class SentenceContext
        {
            public string Text { get; set; } = string.Empty;
            public G2PResult? G2p { get; set; }
            public float[]? BertFeat { get; set; } // Rented from ArrayPool
            public Task? PreprocessTask { get; set; }
        }

        public async IAsyncEnumerable<float[]> PredictStreamAsync(string text, TtsOptions options, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            // Detect voice switch
            EnsureReferenceLoaded(options.AvatarId, options.ReferenceId, options.EmotionId, options.EmotionAlpha);

            string normalized = LanguageDetector.NormalizePunctuation(text);

            // 1. Sentence splitting
            var sentencesText = LanguageDetector.SplitSentences(normalized);
            var contexts = sentencesText.Select(s => new SentenceContext { Text = s }).ToList();

            // 2. Start parallel frontend prefetch (BERT + G2P)
            var overallMode = LanguageDetector.DetectMode(normalized);
            string? globalLangHint = (overallMode == LanguageDetector.LanguageMode.Japanese) ? "ja" : null;

            foreach (var ctx in contexts)
            {
                ctx.PreprocessTask = Task.Run(() =>
                {
                    var (res, bertFeat) = ProcessFrontendOptimal(ctx.Text, options, globalLangHint);
                    ctx.G2p = res;
                    ctx.BertFeat = bertFeat;
                }, cancellationToken);
            }

            // 3. Drive the pipeline sentence by sentence: T2S -> Vocoder (exponential-interval strategy)
            //
            // Cross-sentence overlap: vocoder tasks are serialized per sentence (the single _vocoder session
            // cannot run concurrently), but the next sentence's T2S no longer waits for the previous vocoder
            // to finish. The previous sentence's GPU vocoder thereby overlaps the next sentence's CPU T2S,
            // noticeably shortening end-to-end time; output order is still guaranteed by vocoderChain.
            var mainChannel = System.Threading.Channels.Channel.CreateUnbounded<float[]>();
            var vocoderChain = Task.CompletedTask;

            _ = Task.Run(async () =>
            {
                try
                {
                    foreach (var ctx in contexts)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        if (ctx.PreprocessTask != null) await ctx.PreprocessTask;
                        if (ctx.G2p == null || ctx.BertFeat == null) continue;

                        G2PResult res = ctx.G2p.Value;
                        float[] bertFeat = ctx.BertFeat!;

                        // Per-sentence retry/watermark state
                        int lastAudioLength = 0;
                        var tokenQueue = new System.Collections.Concurrent.ConcurrentQueue<(long[] tokens, bool isFinal)>();
                        var t2sDone = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);

                        // Exponential-interval strategy: initial threshold is chunkSize, then grows by a factor
                        int nextVocoderThreshold = options.StreamingChunkSize;

                        var bert2D = new float[res.PhoneIds.Length, 1024];
                        Buffer.BlockCopy(bertFeat, 0, bert2D, 0, res.PhoneIds.Length * 1024 * sizeof(float));

                        // Vocoder task: queued after the previous sentence, consumes this sentence's tokens chunk by chunk
                        var prevVocoder = vocoderChain;
                        var thisVocoder = Task.Run(async () =>
                        {
                            try
                            {
                                await prevVocoder; // Serialize vocoders across sentences to preserve output order
                                while (!t2sDone.Task.IsCompleted || !tokenQueue.IsEmpty)
                                {
                                    if (tokenQueue.TryDequeue(out var item))
                                    {
                                        var (tokens, isFinal) = item;
                                        float[] fullAudio = _engine.RunVocoder(res.PhoneIds, tokens, _refAudio32k, out int currentTotalLen, _ge, _geAdvanced, options.Speed);
                                        try
                                        {
                                            // Precise skip logic:
                                            // If the current total length currentTotalLen has not yet exceeded the previously sent
                                            // watermark lastAudioLength, this is duplicate audio from a retry and is skipped.
                                            int newSamplesToPush = currentTotalLen - lastAudioLength;
                                            if (newSamplesToPush > 0)
                                            {
                                                // Append a silent tail at sentence end to avoid truncation (200ms margin)
                                                int tailPadding = isFinal ? 6400 : 0;  // 200ms @ 32kHz
                                                float[] chunk = new float[newSamplesToPush + tailPadding];

                                                // Take the last newSamplesToPush samples of fullAudio,
                                                // effectively skipping earlier audio and keeping only samples past the watermark.
                                                Array.Copy(fullAudio, currentTotalLen - newSamplesToPush, chunk, 0, newSamplesToPush);

                                                // Update the watermark
                                                lastAudioLength = currentTotalLen;
                                                await mainChannel.Writer.WriteAsync(chunk, cancellationToken);
                                            }
                                        }
                                        finally
                                        {
                                            _engine.ReturnAudioBuffer(fullAudio);
                                        }
                                    }
                                    else
                                    {
                                        await Task.Delay(5, cancellationToken);
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                // Surface the error to the consumer first and fault the task chain so
                                // subsequent sentences' vocoders fail immediately at await prevVocoder.
                                mainChannel.Writer.TryComplete(ex);
                                throw;
                            }
                        }, cancellationToken);
                        vocoderChain = thisVocoder;

                        // T2S inference: exponential-interval strategy (1.5x growth)
                        // Note: this call is synchronous; while it runs, the previous sentence's vocoder (if any) runs in parallel.
                        try
                        {
                            _engine.RunT2SStreamingTokens(
                                res.PhoneIds, bert2D,
                                _refPhoneIds, _refBert, _sslContent,
                                chunkSize: options.StreamingChunkSize,
                                onTokenChunk: (tokens, isFinal) =>
                                {
                                    // Exponential interval: invoke the vocoder only when the token count reaches the threshold or on isFinal
                                    // Use a 1.5x growth factor to smooth out stalls
                                    if (isFinal || tokens.Length >= nextVocoderThreshold)
                                    {
                                        tokenQueue.Enqueue((tokens, isFinal));
                                        // Threshold grows by the configured factor (default 1.5x); a larger factor means less recompute but higher chunk latency
                                        float factor = options.VocoderRetriggerFactor > 1.05f ? options.VocoderRetriggerFactor : 1.5f;
                                        nextVocoderThreshold = Math.Max((int)(nextVocoderThreshold * factor), tokens.Length + options.StreamingChunkSize);
                                    }
                                },
                                 onRetry: () =>
                                 {
                                     // Core fix: do not reset the lastAudioLength watermark on retry.
                                     // Audio already sent to the user cannot be recalled; duplicates from the new attempt are skipped before pushing.
                                     while (tokenQueue.TryDequeue(out _)) { }
                                     nextVocoderThreshold = options.StreamingChunkSize;
                                     if (_config?.DebugMode == true) Console.WriteLine("[TtsEngineV1] T2S retry triggered, watermark preserved.");
                                 },
                                languageTags: res.LanguageTags,
                                options: options); // Pass options through
                        }
                        finally
                        {
                            // Always signal the vocoder task to exit the loop, on success or failure
                            t2sDone.TrySetResult(true);
                            if (ctx.BertFeat != null)
                            {
                                ArrayPool<float>.Shared.Return(ctx.BertFeat);
                                ctx.BertFeat = null;
                            }
                        }
                    }

                    // All sentences submitted; wait for the last vocoder to finish writing audio
                    await vocoderChain;
                    mainChannel.Writer.Complete();
                }
                catch (Exception ex)
                {
                    mainChannel.Writer.TryComplete(ex);
                }
            }, cancellationToken);

            await foreach (var chunk in mainChannel.Reader.ReadAllAsync(cancellationToken))
            {
                yield return chunk;
            }
        }

        private (G2PResult res, float[] bertFeat) ProcessFrontendOptimal(string text, TtsOptions options, string? explicitLanguage = null)
        {
            var phaseSw = PerfStats.Enabled ? Stopwatch.StartNew() : null;
            try
            {
            // Only normalize punctuation, which helps subsequent language detection
            string normalized = LanguageDetector.NormalizePunctuation(text);

            // Always use MixedLanguageG2P, which handles language segmentation and routing automatically.
            // If Options specifies an allowed-language list, create a temporary MixedLanguageG2P for this request.
            MixedLanguageG2P g2p = _mixedG2p!;
            if (options.Languages != null && options.Languages.Count > 0)
            {
                g2p = new MixedLanguageG2P(_chineseG2p!, _englishG2p!, _japaneseG2p, options.Languages);
                g2p.DebugMode = _config?.DebugMode ?? false;
            }

            var res = g2p.Process(normalized, explicitLanguage, options.G2PPriorityMode);

            var bertFeat = ExtractBertOptimal(res);
            return (res, bertFeat);
            }
            finally
            {
                if (phaseSw != null) PerfStats.AddFrontend(phaseSw);
            }
        }

        private float[] ExtractBertOptimal(G2PResult res)
        {
            // Delegate to the shared mixed-language BERT extraction logic
            // It uses the G2P segmentation info (PhoneLanguage) to decide which spans need BERT.
            // This ensures Japanese kanji are not misclassified as Chinese for BERT extraction (as long as G2P marks them Japanese).
            return ExtractMixedBertOptimal(res);
        }


        private float[] ExtractMixedBertOptimal(G2PResult res)
        {
            float[] final = ArrayPool<float>.Shared.Rent(res.PhoneIds.Length * 1024);
            Array.Clear(final, 0, final.Length);

            if (_bert == null || res.Segments == null) return final;

            foreach (var seg in res.Segments)
            {
                bool isSupported = seg.Language == PhoneLanguage.Chinese;
                if (isSupported && !string.IsNullOrWhiteSpace(seg.Text) && seg.Word2Ph.Length > 0)
                {
                    try
                    {
                        // Alignment fix: the BERT tokenizer ignores spaces, but ChineseG2P emits SP phonemes for them (word2ph=1).
                        // Remove the word2ph entries for spaces and strip spaces from the text so dimensions match.
                        string bertText = seg.Text;
                        int[] bertWord2Ph = seg.Word2Ph;

                        // Character-by-character matching: word2ph maps 1:1 to the normalized text characters.
                        // word2ph entries for whitespace characters must be removed.
                        var cleanChars = new List<char>();
                        var cleanW2P = new List<int>();
                        int charIdx = 0;
                        for (int w = 0; w < bertWord2Ph.Length && charIdx < bertText.Length; w++)
                        {
                            char c = bertText[charIdx];
                            if (char.IsWhiteSpace(c))
                            {
                                // Skip the space and its word2ph entry
                                charIdx++;
                                continue;
                            }
                            cleanChars.Add(c);
                            cleanW2P.Add(bertWord2Ph[w]);
                            charIdx++;
                        }

                        bertText = new string(cleanChars.ToArray());
                        bertWord2Ph = cleanW2P.ToArray();

                        if (bertWord2Ph.Length > 0 && !string.IsNullOrEmpty(bertText))
                        {
                            var feat = _bert.Extract(bertText, bertWord2Ph);
                            int rows = Math.Min(feat.GetLength(0), seg.PhoneCount);
                            // Defensive check
                            rows = Math.Min(rows, res.PhoneIds.Length - seg.StartPhoneIndex);

                            if (rows > 0)
                            {
                                int offset = seg.StartPhoneIndex * 1024;
                                Buffer.BlockCopy(feat, 0, final, offset * sizeof(float), rows * 1024 * sizeof(float));
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        if (_config?.DebugMode == true)
                            Console.WriteLine($"[TtsEngineV1] Mixed BERT extraction failed for segment: {ex.Message}");
                    }
                }
            }
            return final;
        }

        public void Dispose()
        {
            _engine.Dispose();
            _bert?.Dispose();
        }
    }
}
