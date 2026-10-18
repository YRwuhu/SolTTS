using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
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

namespace SolTTS.Core.Core
{
    /// <summary>
    /// V2 inference engine implementation (based on the GPT-SoVITS Minimal Inference architecture).
    /// </summary>
    public class TtsEngineV2 : ITtsEngine
    {
        private readonly InferenceEngineV2 _engine = new();
        private RobertaFeatureExtractor? _bert;
        private ChineseG2P? _chineseG2p;
        private EnglishG2P? _englishG2p;
        private JapaneseG2P? _japaneseG2p;
        private MixedLanguageG2P? _mixedG2p;

        private long[] _refPhoneIds = Array.Empty<long>();
        private float[,] _refBert = new float[0, 0];
        private long[] _promptCodes = Array.Empty<long>();
        private float[,] _referSpec = new float[0, 0];
        private float[]? _svEmb;

        private TTSConfig? _config;

        public int SamplingRate => _engine.SamplingRate;

        public async Task LoadAsync(TTSConfig config)
        {
            _config = config;

            // Validate the V2 model directory
            if (!Directory.Exists(config.ModelsV2Dir))
            {
                throw new DirectoryNotFoundException(
                    $"V2 model directory not found: '{config.ModelsV2Dir}'. " +
                    $"Ensure resources/models_v2 exists, or set UseEngineV2 to false to use the V1 engine.");
            }

            await Task.Run(() =>
            {
                // Load dynamic symbols
                var modelConfigPath = Path.Combine(config.ModelsV2Dir, "config.json");
                if (File.Exists(modelConfigPath))
                {
                    var modelConfig = V2ModelConfig.Load(modelConfigPath);
                    if (modelConfig.SymbolToId != null)
                    {
                        Symbols.LoadDynamicSymbols(modelConfig.SymbolToId);
                    }
                }

                // Load resources in parallel
                Parallel.Invoke(
                    () => _engine.LoadModels(config.ModelsV2Dir, config),
                    () =>
                    {
                        if (File.Exists(config.EffectiveBertModelPath) && File.Exists(config.TokenizerJsonPath))
                        {
                            var options = _engine.CreateSessionOptions(config);
                            _bert = new RobertaFeatureExtractor(config.EffectiveBertModelPath, config.TokenizerJsonPath, options, config.UseBertCache);
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

            // Preprocess reference audio
            await Task.Run(PrepareReference);
        }

        /// <summary>
        /// Share core resources from another engine instance (used for memory-reuse pooling mode).
        /// </summary>
        public void ShareResourcesFrom(TtsEngineV2 other)
        {
            _engine.ShareModelsFrom(other._engine);
            _bert = other._bert;
            _chineseG2p = other._chineseG2p;
            _englishG2p = other._englishG2p;
            _japaneseG2p = other._japaneseG2p;
            _mixedG2p = other._mixedG2p;
            _config = other._config;
        }


        private void PrepareReference()
        {
            if (_config == null) return;

            // Prefer the Avatar system
            string refPath;
            string refText;
            string? refLanguage = null;

            var avatar = _config.GetDefaultAvatar();
            if (avatar != null)
            {
                var reference = avatar.GetDefaultReference();
                if (reference != null)
                {
                    refPath = reference.GetFullAudioPath(_config.AvatarsDir, avatar.Id);
                    refText = reference.Text;
                    refLanguage = reference.Language;
                }
                else
                {
                    Console.WriteLine("Warning: no reference audio configured. Add an entry under Avatars.");
                    return;
                }
            }
            else
            {
                Console.WriteLine("Warning: no reference audio configured. Add an entry under Avatars.");
                return;
            }

            if (!File.Exists(refPath))
            {
                Console.WriteLine($"Warning: reference audio file not found: {refPath}");
                return;
            }

            float[] audio16k = AudioHelper.ReadWav(refPath, 16000);
            float[] audio32k = AudioHelper.ReadWav(refPath, _engine.SamplingRate);

            // Extract features
            var sslContent = _engine.ExtractSSL(audio16k);
            _promptCodes = _engine.EncodeVQ(sslContent);
            _referSpec = ComputeSpectrogram(audio32k, _engine.SamplingRate);

            // Pro versions handle speaker embeddings
            if (_engine.ModelVersion.Contains("Pro", StringComparison.OrdinalIgnoreCase))
            {
                string npyPath = Path.Combine(_config.ModelsV2Dir, "sv_emb.npy");
                if (File.Exists(npyPath))
                {
                    _svEmb = LoadNpyFloat32(npyPath);
                }

                if (_svEmb == null)
                {
                    _svEmb = _engine.ExtractSpeakerEmbedding(audio16k);
                }
            }

            // Build a suitable G2P for the reference audio (extend temporarily if its language is not globally allowed)
            MixedLanguageG2P refG2p = _mixedG2p!;
            if (!string.IsNullOrEmpty(refLanguage))
            {
                var refLangs = new List<string>(_config.G2P.Languages ?? new List<string> { "zh", "en" });
                var lower = refLanguage.ToLowerInvariant();
                if (!refLangs.Contains(lower))
                {
                    refLangs.Add(lower == "jp" ? "ja" : lower);
                    refG2p = new MixedLanguageG2P(_chineseG2p!, _englishG2p!, _japaneseG2p, refLangs);
                    refG2p.DebugMode = _config.DebugMode;
                }
            }
            else
            {
                // Reference audio has no language set → auto-detect: temporarily allow Japanese if the text has kana or Japanese hints
                bool hasJaHints = refText.Any(c => (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF) || c == 0x30FC || c == 0x300C || c == 0x300D || c == 0x3005);
                if (hasJaHints)
                {
                    refLanguage = "ja";
                    var refLangs = new List<string>(_config.G2P.Languages ?? new List<string> { "zh", "en" });
                    if (!refLangs.Contains("ja")) refLangs.Add("ja");
                    refG2p = new MixedLanguageG2P(_chineseG2p!, _englishG2p!, _japaneseG2p, refLangs);
                    refG2p.DebugMode = _config.DebugMode;
                }
            }

            var res = refG2p.Process(refText, refLanguage);
            _refPhoneIds = Symbols.HasDynamicSymbols ? Symbols.GetIdsV2(res.Phones) : Symbols.GetIds(res.Phones);
            _refBert = ExtractMixedBertFeature(_bert, res);
        }

        public async Task<TtsResult> PredictAsync(string text, TtsOptions options)
        {
            // Detect the overall language tendency
            var overallMode = LanguageDetector.DetectMode(text);
            string? globalLangHint = (overallMode == LanguageDetector.LanguageMode.Japanese) ? "ja" : null;

            return await Task.Run(() =>
            {
                var (res, phoneIds, bertFeat) = ProcessFrontend(text, options, globalLangHint);

                var semanticTokens = _engine.InferTokens(
                    _refPhoneIds, ConvertTo2D(_refBert),
                    phoneIds, ConvertTo2D(bertFeat),
                    _promptCodes,
                    res.LanguageTags,
                    options,
                    topK: options.TopK,
                    temperature: options.Temperature
                );

                var audio = _engine.RunSoVITS(
                    semanticTokens,
                    phoneIds,
                    _referSpec,
                    _svEmb,
                    options.NoiseScale,
                    options.Speed
                );

                return new TtsResult
                {
                    Audio = audio,
                    TokenCount = semanticTokens.Length
                };
            });
        }

        public async IAsyncEnumerable<float[]> PredictStreamAsync(string text, TtsOptions options, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            // Detect the overall language tendency
            var overallMode = LanguageDetector.DetectMode(text);
            string? globalLangHint = (overallMode == LanguageDetector.LanguageMode.Japanese) ? "ja" : null;

            var (res, phoneIds, bertFeat) = ProcessFrontend(text, options, globalLangHint);

            await foreach (var chunk in _engine.InferStream(
                _refPhoneIds, ConvertTo2D(_refBert),
                phoneIds, ConvertTo2D(bertFeat),
                _promptCodes, _referSpec, _svEmb,
                res.LanguageTags, options,
                topK: options.TopK,
                temperature: options.Temperature,
                noiseScale: options.NoiseScale,
                speed: options.Speed,
                chunkSize: options.StreamingChunkSize).WithCancellation(cancellationToken))
            {
                if (chunk.Length > 0)
                {
                    yield return chunk;
                }
            }
        }

        private (G2PResult res, long[] phoneIds, float[,] bert) ProcessFrontend(string text, TtsOptions options, string? explicitLanguage = null)
        {
            // Normalize punctuation only; this helps subsequent language detection
            string normalized = LanguageDetector.NormalizePunctuation(text);

            // Always use MixedLanguageG2P for processing
            MixedLanguageG2P g2p = _mixedG2p!;
            if (options.Languages != null && options.Languages.Count > 0)
            {
                g2p = new MixedLanguageG2P(_chineseG2p!, _englishG2p!, _japaneseG2p, options.Languages);
                g2p.DebugMode = _config?.DebugMode ?? false;
            }

            var res = g2p.Process(normalized, explicitLanguage, options.G2PPriorityMode);

            var mode = LanguageDetector.DetectMode(res.NormalizedText);
            float[,] bertFeat;
            if (mode == LanguageDetector.LanguageMode.Chinese || mode == LanguageDetector.LanguageMode.Mixed)
                bertFeat = ExtractMixedBertFeature(_bert, res);
            else
                bertFeat = new float[res.PhoneIds.Length, 1024];

            long[] phoneIds = Symbols.HasDynamicSymbols ? Symbols.GetIdsV2(res.Phones) : Symbols.GetIds(res.Phones);
            return (res, phoneIds, bertFeat);
        }

        #region Helpers (Migrated from Program_V2)

        private static float[,] ComputeSpectrogram(float[] audio, int sampleRate)
        {
            int fftSize = 2048;
            int hopLength = 640;
            int winLength = 2048;
            int numBins = fftSize / 2 + 1;
            int padSize = (fftSize - hopLength) / 2;
            var paddedAudio = new float[audio.Length + padSize * 2];

            for (int i = 0; i < padSize; i++)
            {
                paddedAudio[i] = audio[Math.Min(padSize - i, audio.Length - 1)];
                paddedAudio[paddedAudio.Length - 1 - i] = audio[Math.Max(0, audio.Length - padSize + i - 1)];
            }
            Array.Copy(audio, 0, paddedAudio, padSize, audio.Length);

            int numFrames = (paddedAudio.Length - fftSize) / hopLength + 1;
            if (numFrames <= 0) numFrames = 1;

            var spec = new float[numBins, numFrames];
            var window = new float[winLength];
            for (int i = 0; i < winLength; i++) window[i] = 0.5f * (1 - MathF.Cos(2 * MathF.PI * i / winLength));

            for (int frame = 0; frame < numFrames; frame++)
            {
                int start = frame * hopLength;
                var real = new double[fftSize];
                for (int i = 0; i < fftSize && start + i < paddedAudio.Length; i++)
                    real[i] = paddedAudio[start + i] * (i < winLength ? window[i] : 0);

                for (int k = 0; k < numBins; k++)
                {
                    double sumReal = 0, sumImag = 0;
                    for (int n = 0; n < fftSize; n++)
                    {
                        double angle = -2.0 * Math.PI * k * n / fftSize;
                        sumReal += real[n] * Math.Cos(angle);
                        sumImag += real[n] * Math.Sin(angle);
                    }
                    spec[k, frame] = (float)Math.Sqrt(sumReal * sumReal + sumImag * sumImag + 1e-8);
                }
            }
            return spec;
        }


        private static float[,] ExtractMixedBertFeature(RobertaFeatureExtractor? bert, G2PResult res)
        {
            var final = new float[res.PhoneIds.Length, 1024];
            if (bert == null || res.Segments == null) return final;
            foreach (var seg in res.Segments)
            {
                if (seg.Language == PhoneLanguage.Chinese && !string.IsNullOrWhiteSpace(seg.Text) && seg.Word2Ph.Length > 0)
                {
                    try
                    {
                        var feat = bert.Extract(seg.Text, seg.Word2Ph);
                        int rows = Math.Min(feat.GetLength(0), seg.PhoneCount);
                        rows = Math.Min(rows, res.PhoneIds.Length - seg.StartPhoneIndex);
                        if (rows > 0) Buffer.BlockCopy(feat, 0, final, seg.StartPhoneIndex * 1024 * sizeof(float), rows * 1024 * sizeof(float));
                    }
                    catch { }
                }
            }
            return final;
        }

        private static float[,] ConvertTo2D(float[,] bert)
        {
            int rows = bert.GetLength(0);
            int cols = bert.GetLength(1);
            var transposed = new float[cols, rows];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    transposed[j, i] = bert[i, j];
            return transposed;
        }

        private static float[]? LoadNpyFloat32(string path)
        {
            try
            {
                using var fs = File.OpenRead(path);
                using var br = new BinaryReader(fs);
                br.ReadBytes(6); // Magic
                byte major = br.ReadByte(); br.ReadByte(); // Version
                int headerLen = major == 1 ? br.ReadUInt16() : (int)br.ReadUInt32();
                var header = System.Text.Encoding.ASCII.GetString(br.ReadBytes(headerLen));
                int totalElements = 1;
                int shapeStart = header.IndexOf("'shape':");
                if (shapeStart >= 0)
                {
                    int pStart = header.IndexOf('(', shapeStart);
                    int pEnd = header.IndexOf(')', shapeStart);
                    if (pStart >= 0 && pEnd > pStart)
                    {
                        var dims = header.Substring(pStart + 1, pEnd - pStart - 1).Split(',', StringSplitOptions.RemoveEmptyEntries);
                        foreach (var dim in dims) if (int.TryParse(dim.Trim(), out int d)) totalElements *= d;
                    }
                }
                var data = new float[totalElements];
                for (int i = 0; i < totalElements; i++) data[i] = br.ReadSingle();
                return data;
            }
            catch { return null; }
        }

        #endregion

        public void Dispose()
        {
            _engine.Dispose();
            _bert?.Dispose();
        }
    }
}
