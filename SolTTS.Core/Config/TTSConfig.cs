using System.Text.Json.Serialization;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;
using SolTTS.Core.Core;

namespace SolTTS.Core.Config
{
    public class TTSConfig
    {
        // ============================================================
        // Resource directory settings (core path - only this needs configuring)
        // ============================================================

        private string _resourcesDir = "resources";
        public string ResourcesDir
        {
            get => Path.IsPathRooted(_resourcesDir)
                ? _resourcesDir
                : Path.Combine(AppDomain.CurrentDomain.BaseDirectory, _resourcesDir);
            set => _resourcesDir = value;
        }

        // ============================================================
        // Derived paths (computed automatically; no manual configuration needed)
        // ============================================================

        /// <summary>
        /// Shared resources directory (dictionaries, etc.)
        /// </summary>
        [JsonIgnore]
        public string SharedDir => Path.Combine(ResourcesDir, "shared");

        /// <summary>
        /// Dictionaries directory
        /// </summary>
        [JsonIgnore]
        public string DictionariesDir => Path.Combine(SharedDir, "dictionaries");

        /// <summary>
        /// V1 engine shared resources directory (BERT, HuBERT, etc.)
        /// </summary>
        [JsonIgnore]
        public string V1ExtraDir => Path.Combine(SharedDir, "v1_extra");

        /// <summary>
        /// V1 engine voice-specific model directory
        /// </summary>
        [JsonIgnore]
        public string ModelsV1BaseDir => Path.Combine(ResourcesDir, "models_v1");

        /// <summary>
        /// V2 engine base model directory
        /// </summary>
        [JsonIgnore]
        public string ModelsV2BaseDir => Path.Combine(ResourcesDir, "models_v2");

        /// <summary>
        /// Get the V1 model directory for the specified avatar
        /// </summary>
        public string GetModelsV1Dir(string avatarId) => Path.Combine(ModelsV1BaseDir, avatarId);

        /// <summary>
        /// Get the V2 model directory for the specified avatar
        /// </summary>
        public string GetModelsV2Dir(string avatarId) => Path.Combine(ModelsV2BaseDir, avatarId);

        /// <summary>
        /// Get the V1 model directory for the default avatar (legacy compatibility)
        /// </summary>
        [JsonIgnore]
        public string ModelsV1Dir => GetModelsV1Dir(DefaultAvatarId);

        /// <summary>
        /// Get the V2 model directory for the default avatar (legacy compatibility)
        /// </summary>
        [JsonIgnore]
        public string ModelsV2Dir => GetModelsV2Dir(DefaultAvatarId);

        // ============================================================
        // V1 engine model paths
        // ============================================================

        [JsonIgnore]
        public string HubertPath => Path.Combine(V1ExtraDir, "hubert", "chinese-hubert-base_full.onnx");

        [JsonIgnore]
        public string SpeakerEncoderPath => Path.Combine(V1ExtraDir, "speaker_encoder.onnx");

        // V1 voice-specific VITS model directory (core models are placed directly in the voice directory)
        [JsonIgnore]
        public string V1TtsDir => ModelsV1Dir;

        [JsonIgnore]
        public string BertModelPath => Path.Combine(V1ExtraDir, "bert", "roberta.onnx");

        /// <summary>
        /// BERT model path actually in use. Prefers the no-Loop variant when UseBertNoLoopModel=true
        /// and it exists; otherwise falls back to the standard model (ensuring it always loads).
        /// </summary>
        [JsonIgnore]
        public string EffectiveBertModelPath
        {
            get
            {
                var standard = BertModelPath;
                if (!UseBertNoLoopModel) return standard;
                var noLoop = Path.Combine(V1ExtraDir, "bert", "roberta_noloop.onnx");
                return File.Exists(noLoop) ? noLoop : standard;
            }
        }

        [JsonIgnore]
        public string TokenizerJsonPath => Path.Combine(V1ExtraDir, "bert", "tokenizer", "tokenizer.json");

        // ============================================================
        // Get V1 paths by avatar
        // ============================================================

        // public string GetHubertPath(string avatarId) => HubertPath;
        // public string GetSpeakerEncoderPath(string avatarId) => SpeakerEncoderPath;
        // public string GetV1TtsDir(string avatarId) => GetModelsV1Dir(avatarId);
        // public string GetBertModelPath(string avatarId) => BertModelPath;
        // public string GetTokenizerJsonPath(string avatarId) => TokenizerJsonPath;

        // ============================================================
        // Shared resource paths (G2P dictionaries and models)
        // ============================================================

        [JsonIgnore]
        public string G2PDir => Path.Combine(SharedDir, "g2p");

        [JsonIgnore]
        public string G2PDictsDir => Path.Combine(G2PDir, "dicts");

        [JsonIgnore]
        public string G2PModelsDir => Path.Combine(G2PDir, "models");

        [JsonIgnore]
        public string CmuDict => Path.Combine(G2PDictsDir, "english", "cmudict.dict");

        [JsonIgnore]
        public string PinyinDict => Path.Combine(G2PDictsDir, "chinese", "mandarin_pinyin.dict");

        [JsonIgnore]
        public string ChineseG2PDict => Path.Combine(G2PDictsDir, "chinese", "opencpop-strict.txt");

        [JsonIgnore]
        public string PolyphonicJson => Path.Combine(G2PDictsDir, "chinese", "polyphonic.json");

        [JsonIgnore]
        public string NeuralG2PModel => Path.Combine(G2PModelsDir, "checkpoint20.npz");

        [JsonIgnore]
        public string EnglishSpecialDict => Path.Combine(G2PDictsDir, "english", "en_special_words.txt");

        /// <summary>
        /// English POS tagging model directory
        /// </summary>
        [JsonIgnore]
        public string EnglishPosTaggerDir => Path.Combine(G2PModelsDir, "english", "taggers", "averaged_perceptron_tagger_eng");

        /// <summary>
        /// English word segmentation data directory
        /// </summary>
        [JsonIgnore]
        public string EnglishWordSegmentDir => Path.Combine(G2PModelsDir, "english", "wordsegment");

        /// <summary>
        /// Japanese OpenJTalk dictionary directory
        /// </summary>
        [JsonIgnore]
        public string JapaneseDictDir => Path.Combine(G2PDictsDir, "japanese", "open_jtalk_dic_utf_8-1.11");

        [JsonIgnore]
        public string JiebaDictDir => Path.Combine(G2PDictsDir, "chinese", "jieba");

        // ============================================================
        // Voice (Avatar) configuration
        // ============================================================

        /// <summary>
        /// Default voice ID. Used when the request does not specify one.
        /// </summary>
        public string DefaultAvatarId { get; set; } = "default";

        /// <summary>
        /// Voice directory.
        /// </summary>
        [JsonIgnore]
        public string AvatarsDir => Path.Combine(ResourcesDir, "avatars");

        /// <summary>
        /// Voice list. Can be defined manually in the config file or auto-scanned from AvatarsDir.
        /// </summary>
        public List<Avatar> Avatars { get; set; } = new();

        /// <summary>
        /// Get the default avatar
        /// </summary>
        public Avatar? GetDefaultAvatar()
        {
            return Avatars.FirstOrDefault(a => a.Id == DefaultAvatarId) ?? Avatars.FirstOrDefault();
        }

        /// <summary>
        /// Get the full path and text of the default reference audio
        /// </summary>
        // public (string audioPath, string text)? GetDefaultReferenceAudio()
        // {
        //     var avatar = GetDefaultAvatar();
        //     if (avatar == null) return null;

        //     var reference = avatar.GetDefaultReference();
        //     if (reference == null) return null;

        //     var fullPath = reference.GetFullAudioPath(AvatarsDir, avatar.Id);
        //     return (fullPath, reference.Text);
        // }

        // ============================================================
        // Engine selection
        // ============================================================

        /// <summary>
        /// Use the V2 inference engine (based on GPT-SoVITS minimal inference)
        /// </summary>
        public bool UseEngineV2 { get; set; } = false;

        // ============================================================
        // Hardware acceleration (GPU is provided by the ggml plugin EP: Windows/Linux=Vulkan, macOS=Metal)
        // ============================================================

        /// <summary>
        /// Hybrid mode: only the vocoder (vits) uses the ggml plugin EP (Windows/Linux=Vulkan, macOS=Metal),
        /// all other models stay on CPU. Measured: vits is about 5x faster on Vulkan than CPU (AMD RX 5500 XT);
        /// about 2x faster for the full streaming pipeline and about 25% faster for non-streaming. Requires ORT 1.29+ and the onnxruntime_ggml runtime;
        /// falls back to CPU automatically when the library is missing or fails to load. Default false.
        /// </summary>
        public bool UseVocoderGgml { get; set; } = false;

        /// <summary>
        /// Experimental: use the ggml plugin EP for BERT (roberta) (Vulkan/Metal). Default false.
        ///
        /// Measured result (onnxruntime-ggml v0.2.0 + ORT 1.30, RX 5500 XT): the plugin reports
        /// "claiming nothing" for roberta — 22 Gelu nodes in the graph are unsupported, the whole graph is still
        /// executed by the CPU EP, yet the session still pays plugin init/compile-probe overhead, making
        /// end-to-end about 1.7x slower (294→489 ms). Therefore this switch stays off by default and is only
        /// used when re-evaluating after a plugin upgrade.
        /// </summary>
        public bool UseBertGgml { get; set; } = false;

        /// <summary>
        /// Experimental: use the Loop-fused decoder model (t2s_stage_decoder_loop.onnx).
        ///
        /// This variant merges about 95 per-token InferenceSession.Run calls into a single in-graph Loop.
        /// Measured result (RX 5500 XT / Ryzen 5 5500, 30 steps): 1.02x and 0.98x, i.e. no gain beyond noise.
        /// Reason: the 15.7 ms per step is the real weight-bandwidth cost of the 24-layer Transformer (about 72MB/step),
        /// not Run boundary overhead; the Loop only removes the latter.
        /// This switch remains as a baseline for validating future in-graph optimizations (fixed-shape KV, pruning redundant ops).
        /// Automatically falls back to the normal per-token path when the model file is missing. Default false.
        /// </summary>
        public bool UseStageDecoderLoopModel { get; set; } = false;

        /// <summary>
        /// Whether to enable BERT feature caching. When enabled, repeated calls with the same text (including alignment)
        /// reuse the result, saving one BERT forward pass of about 280–390 ms. The cache lives with the engine instance. Default true.
        /// </summary>
        public bool UseBertCache { get; set; } = true;

        /// <summary>
        /// Whether to use the no-Loop roberta model (roberta_noloop.onnx).
        /// This variant replaces the Loop node that causes about 278 ms of CPU control flow with equivalent vectorized
        /// operators (a CumSum+Gather implementation of repeat_interleave), benefiting both CPU and GPU backends.
        /// Falls back to the original model when the file is absent. Default false (verify identical output with
        /// the comparison tool before enabling).
        /// </summary>
        public bool UseBertNoLoopModel { get; set; } = false;

        // ============================================================
        // Emotion control (emotion guidance without reference audio)
        // ============================================================

        /// <summary>
        /// Whether to enable emotion control. When enabled, the requested EmotionId's emotion direction vector is
        /// injected when loading the reference audio. Silently disabled when the τ file is missing (normal synthesis is unaffected). Default false.
        /// </summary>
        public bool UseEmotionControl { get; set; } = false;

        /// <summary>
        /// Emotion direction vector (τ) directory, relative to ResourcesDir. Default "emotion_vectors".
        /// The directory should contain manifest.json and tau_&lt;emotion&gt;.f32 (generated by tools/emotion/extract_tau.py).
        /// </summary>
        public string EmotionVectorPath { get; set; } = "emotion_vectors";

        /// <summary>
        /// Default emotion strength α (0~1.5). x_new = x_base + α·||x_base||·τ̂.
        /// Empirical range 0.1~0.5: too small has no visible effect, too large degrades voice similarity. Default 0.3.
        /// </summary>
        public float EmotionAlpha { get; set; } = 0.3f;

        /// <summary>Full path to the emotion direction vector directory.</summary>
        [JsonIgnore]
        public string EmotionVectorFullPath =>
            Path.IsPathRooted(EmotionVectorPath)
                ? EmotionVectorPath
                : Path.Combine(ResourcesDir, EmotionVectorPath);

        /// <summary>
        /// τ directory actually in use: prefers ResourcesDir/EmotionVectorPath; if absent, falls back to
        /// a directory of the same name under the application base directory (so τ can ship with the app
        /// without depending on the user resource bundle).
        /// </summary>
        [JsonIgnore]
        public string EffectiveEmotionVectorPath
        {
            get
            {
                var primary = EmotionVectorFullPath;
                if (Directory.Exists(primary)) return primary;
                var fallback = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, EmotionVectorPath);
                return Directory.Exists(fallback) ? fallback : primary;
            }
        }


        /// <summary>
        /// Whether to enable CoreML acceleration (macOS only).
        /// Off by default: for models with loop control flow such as GPT-SoVITS, CoreML subgraph partitioning is
        /// actually slower than pure CPU, and the first load requires tens of seconds of ANE compilation.
        /// Enable only if you have verified your model benefits.
        /// </summary>
        public bool UseCoreML { get; set; } = false;

        /// <summary>
        /// CoreML acceleration flag combination (macOS only).
        /// 0   = default (CPU + ANE + GPU all used)
        /// 1   = COREML_FLAG_USE_CPU_ONLY (CPU only, not recommended)
        /// 2   = COREML_FLAG_ENABLE_ON_SUBGRAPH (subgraph fallback)
        /// 4   = COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE (prefer ANE)
        /// 8   = COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES
        /// 16  = COREML_FLAG_CREATE_MLPROGRAM (use MLProgram format; more modern but slightly less compatible)
        /// Recommended: 2 (subgraph fallback) for better compatibility
        /// </summary>
        public uint CoreMLFlags { get; set; } = 2;

        // ============================================================
        // Performance configuration
        // ============================================================

        /// <summary>
        /// Number of intra-op parallel threads in ONNX Runtime. 0 means use the default.
        /// </summary>
        public int IntraOpNumThreads { get; set; } = 0;

        /// <summary>
        /// Number of inter-op parallel threads in ONNX Runtime. 0 means use the default.
        /// </summary>
        public int InterOpNumThreads { get; set; } = 0;

        /// <summary>
        /// Graph optimization level (0=disabled, 1=basic, 2=aggressive).
        /// </summary>
        public int GraphOptimizationLevel { get; set; } = 1;

        /// <summary>
        /// Whether to allow ORT operator threads to spin-wait. Default false (more power-efficient on Windows,
        /// but higher scheduling latency for small operators). Enabling is recommended for per-token decoding workloads to reduce wake-up latency.
        /// </summary>
        public bool IntraOpAllowSpinning { get; set; } = false;

        /// <summary>
        /// Whether to use ORT_PARALLEL execution mode (inter-op parallelism). Default false (sequential execution).
        /// For per-token Transformer decoding, this allows independent branches such as Q/K/V in the same layer to run in parallel.
        /// </summary>
        public bool ParallelExecution { get; set; } = false;

        /// <summary>
        /// Upper bound on the number of sentences decoded in parallel (1 = disabled). In non-streaming synthesis,
        /// each sentence's T2S decoding is independent, so they can run in parallel to shorten total end-to-end time.
        /// Measured: a 3-sentence text on 6C/12T drops median latency from ~3.5s to ~2.4s;
        /// single-sentence requests are unaffected (degree converges to the sentence count). Default 3.
        /// The streaming path is unaffected (must preserve per-sentence output order).
        /// </summary>
        public int ParallelSentenceDegree { get; set; } = 3;

        /// <summary>
        /// Inference pool capacity (concurrency). Default 1.
        /// </summary>
        public int PoolCapacity { get; set; } = 1;

        /// <summary>
        /// Whether to reuse memory (shared model Session). Default true.
        /// </summary>
        public bool ReuseMemory { get; set; } = true;

        // ============================================================
        // Synthesis parameters
        // ============================================================

        /// <summary>
        /// Speech rate (0.5 - 2.0)
        /// </summary>
        public float Speed { get; set; } = 1.0f;

        /// <summary>
        /// Noise scale (affects voice variation)
        /// </summary>
        public float NoiseScale { get; set; } = 0.35f;

        /// <summary>
        /// Top-K sampling (recommended 15-50)
        /// </summary>
        public int TopK { get; set; } = 15;

        /// <summary>
        /// Sampling temperature (higher is more random)
        /// </summary>
        public float Temperature { get; set; } = 1.0f;

        // ============================================================
        // Streaming configuration
        // ============================================================

        public int StreamingChunkSize { get; set; } = 22;
        public int StreamingPreBufferChunks { get; set; } = 2;
        public bool StreamingMode { get; set; } = true;

        /// <summary>
        /// Growth factor for the streaming vocoder retrigger threshold (default 4.0).
        /// Each vocoder call recomputes over "all tokens accumulated so far" (VITS includes enc_p attention and
        /// flow global dependencies, so it cannot decode only the new window); total compute per sentence is about
        /// Σmin(factor^k·chunk, L)/L + 1 times the final length.
        /// Measured 1.5→4.0: vocoder calls 56→42, total vocoder time -29%, streaming 3-sentence total time about -10%,
        /// with no impact on first-packet latency (TTFB). The cost is larger audio chunk intervals (longer chunks).
        /// Valid range [1.2, 4.0].
        /// </summary>
        public float VocoderRetriggerFactor { get; set; } = 4.0f;

        // ============================================================
        // Audio backend
        // ============================================================

        public bool WasapiExclusiveMode { get; set; } = true;
        public int LockFreeBufferSize { get; set; } = 65536 * 32;

        /// <summary>
        /// Use native/sol_audio (Rust) for real-time audio primitives: SPSC ring buffer,
        /// first-order high-pass filter, crossfade, float→PCM16. Falls back to the managed implementation when the DLL is
        /// missing or the version does not match. This path only affects playback-side DSP, not inference or audio content. Default false.
        /// </summary>
        public bool UseNativeAudio { get; set; } = false;

        // ============================================================
        // Debug and inference configuration
        // ============================================================

        /// <summary>
        /// Whether to enable debug mode. When enabled, detailed inference logs are output.
        /// </summary>
        public bool DebugMode { get; set; } = false;

        public InferenceConfig Inference { get; set; } = new InferenceConfig();

        public class InferenceConfig
        {
            /// <summary>
            /// Maximum number of retries. When too few tokens are generated, the engine attempts regeneration.
            /// </summary>
            public int MaxRetries { get; set; } = 3;

            /// <summary>
            /// Minimum expected token multiplier for Chinese (default 2.1). Higher values increase retry probability.
            /// </summary>
            public float MinTokenMultiplierChinese { get; set; } = 2.1f;

            /// <summary>
            /// Minimum expected token multiplier for Japanese (default 2.0).
            /// </summary>
            public float MinTokenMultiplierJapanese { get; set; } = 2.0f;

            /// <summary>
            /// Minimum expected token multiplier for English (default 1.3).
            /// </summary>
            public float MinTokenMultiplierEnglish { get; set; } = 1.3f;

            /// <summary>
            /// Minimum expected token multiplier for punctuation/other (default 0.5).
            /// </summary>
            public float MinTokenMultiplierOther { get; set; } = 0.5f;
        }

        // ============================================================
        // G2P configuration
        // ============================================================

        public G2PConfig G2P { get; set; } = new G2PConfig();

        public class G2PConfig
        {
            /// <summary>
            /// User-defined dictionary path (relative to SharedDir)
            /// </summary>
            public string? CustomDictPath { get; set; } = "custom_dict.txt";

            /// <summary>
            /// Priority mode: 0 = dictionary first, 1 = dictionary only, 2 = model first
            /// </summary>
            public int PriorityMode { get; set; } = 0;

            /// <summary>
            /// Allowed language list (at most 2). Valid values: "zh", "en", "ja"
            /// </summary>
            public List<string> Languages { get; set; } = new() { "zh", "en" };
        }

        // ============================================================
        // Full custom dictionary path
        // ============================================================

        [JsonIgnore]
        public string? CustomDictFullPath =>
            string.IsNullOrEmpty(G2P.CustomDictPath) ? null : Path.Combine(SharedDir, G2P.CustomDictPath);

        // ============================================================
        // Configuration load and save
        // ============================================================

        private static string? _loadedPath;

        [JsonIgnore]
        public static string? LoadedPath => _loadedPath;

        [JsonIgnore]
        [YamlIgnore]
        public static string DefaultConfigName => "config.yaml";

        [JsonIgnore]
        [YamlIgnore]
        public static string LegacyConfigName => "config.json";

        public static string DefaultConfigPath => Path.Combine(AppDomain.CurrentDomain.BaseDirectory, DefaultConfigName);

        public static TTSConfig Load()
        {
            // Try CWD first
            if (File.Exists(DefaultConfigName)) return Load(DefaultConfigName);
            if (File.Exists(LegacyConfigName)) return Load(LegacyConfigName);

            // Then try AppDir
            var appDirDefault = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, DefaultConfigName);
            if (File.Exists(appDirDefault)) return Load(appDirDefault);

            var appDirLegacy = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, LegacyConfigName);
            if (File.Exists(appDirLegacy)) return Load(appDirLegacy);

            // Fallback to default path (will create if LoadOrCreate is used)
            return LoadOrCreate(appDirDefault);
        }

        public static TTSConfig Load(string path)
        {
            _loadedPath = Path.GetFullPath(path);
            if (!File.Exists(_loadedPath)) return new TTSConfig();
            try
            {
                var content = File.ReadAllText(_loadedPath);
                var extension = Path.GetExtension(_loadedPath).ToLower();

                if (extension == ".yaml" || extension == ".yml")
                {
                    var deserializer = new DeserializerBuilder()
                        .WithNamingConvention(PascalCaseNamingConvention.Instance)
                        .IgnoreUnmatchedProperties()
                        .Build();
                    return deserializer.Deserialize<TTSConfig>(content) ?? new TTSConfig();
                }
                else
                {
                    // Fallback to JSON
                    var options = new System.Text.Json.JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true,
                        ReadCommentHandling = System.Text.Json.JsonCommentHandling.Skip
                    };
                    return System.Text.Json.JsonSerializer.Deserialize<TTSConfig>(content, options) ?? new TTSConfig();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading config from {path}: {ex.Message}, using defaults.");
                return new TTSConfig();
            }
        }

        public static TTSConfig LoadOrCreate(string path)
        {
            _loadedPath = path;
            if (!File.Exists(path))
            {
                Console.WriteLine($"Config file not found at '{path}'. Creating default config...");
                var defaultConfig = new TTSConfig();
                defaultConfig.Save(path);
                Console.WriteLine($"Default config created at '{path}'.");
                return defaultConfig;
            }
            return Load(path);
        }

        public static TTSConfig Reload()
        {
            if (string.IsNullOrEmpty(_loadedPath))
                return Load();
            return Load(_loadedPath);
        }

        public void Save(string path)
        {
            var extension = Path.GetExtension(path).ToLower();
            if (extension == ".yaml" || extension == ".yml")
            {
                var serializer = new SerializerBuilder()
                    .WithNamingConvention(PascalCaseNamingConvention.Instance)
                    .Build();
                var yaml = serializer.Serialize(this);
                File.WriteAllText(path, yaml);
            }
            else
            {
                var options = new System.Text.Json.JsonSerializerOptions { WriteIndented = true };
                var json = System.Text.Json.JsonSerializer.Serialize(this, options);
                File.WriteAllText(path, json);
            }
        }

        // ============================================================
        // Path diagnostics output
        // ============================================================

        public void PrintPaths()
        {
            Console.WriteLine("=== SolTTS Path Configuration ===");
            Console.WriteLine($"ResourcesDir: {ResourcesDir}");
            Console.WriteLine($"SharedDir: {SharedDir}");
            Console.WriteLine($"DictionariesDir: {DictionariesDir}");
            Console.WriteLine($"ModelsV1Dir: {ModelsV1Dir}");
            Console.WriteLine($"ModelsV2Dir: {ModelsV2Dir}");
            Console.WriteLine($"AvatarsDir: {AvatarsDir}");
            Console.WriteLine($"UseEngineV2: {UseEngineV2}");
        }
    }
}
