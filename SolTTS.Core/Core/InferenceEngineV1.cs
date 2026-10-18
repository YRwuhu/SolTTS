using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Buffers;
using System.Runtime.InteropServices;
using SolTTS.Core.Frontend.G2P.Common;

namespace SolTTS.Core.Core
{
    public class InferenceEngineV1 : IDisposable
    {
        public bool DebugMode { get; set; } = false;

        /// <summary>
        /// Whether shared mode is enabled. In shared mode Dispose does not release sessions.
        /// </summary>
        public bool IsShared { get; set; } = false;

        private void DebugLog(string message)
        {
            if (DebugMode) Console.WriteLine(message);
        }

        private SolTTS.Core.Config.TTSConfig? _config;

        private InferenceSession? _t2sEncoder;
        private InferenceSession? _firstStageDecoder;
        private InferenceSession? _stageDecoder;
        private InferenceSession? _stageDecoderLoop;
        private InferenceSession? _vocoder;

        // Decode loop name/index cache: avoids per-token LINQ name lookups and string interpolation.
        private const int KvLayerCount = 24;
        private static readonly string[] PastKNames = BuildLayerNames("past_k_layer_");
        private static readonly string[] PastVNames = BuildLayerNames("past_v_layer_");
        private static readonly string[] PresentKNames = BuildLayerNames("present_k_layer_");
        private static readonly string[] PresentVNames = BuildLayerNames("present_v_layer_");

        private static string[] BuildLayerNames(string prefix)
        {
            var names = new string[KvLayerCount];
            for (int i = 0; i < KvLayerCount; i++) names[i] = prefix + i;
            return names;
        }

        // V2ProPlus components
        private InferenceSession? _promptEncoder;
        private InferenceSession? _hubert;
        private InferenceSession? _svModel; // Speaker Verification

        public SessionOptions GetSessionOptions(SolTTS.Core.Config.TTSConfig config)
        {
            return GetSessionOptions(config, useCoreML: config.UseCoreML);
        }

        /// <summary>
        /// Create SessionOptions with explicit control over CoreML (used for failure fallback).
        /// </summary>
        public SessionOptions GetSessionOptions(SolTTS.Core.Config.TTSConfig config, bool useCoreML)
        {
            var sessionOptions = new SessionOptions();

            bool isMac = RuntimeInformation.IsOSPlatform(OSPlatform.OSX);

            // 1. Thread configuration
            //    Match Win/Linux: leave ORT defaults untouched (users can override in config).
            //    Measured on Apple Silicon: limiting to P-Core count actually reduces GEMM throughput.
            sessionOptions.IntraOpNumThreads = config.IntraOpNumThreads;
            sessionOptions.InterOpNumThreads = config.InterOpNumThreads;

            // 2. Execution mode
            sessionOptions.ExecutionMode = config.ParallelExecution
                ? ExecutionMode.ORT_PARALLEL
                : ExecutionMode.ORT_SEQUENTIAL;

            // 3. Graph optimization level
            sessionOptions.GraphOptimizationLevel = config.GraphOptimizationLevel switch
            {
                0 => GraphOptimizationLevel.ORT_DISABLE_ALL,
                1 => GraphOptimizationLevel.ORT_ENABLE_BASIC,
                2 => GraphOptimizationLevel.ORT_ENABLE_ALL,
                _ => GraphOptimizationLevel.ORT_ENABLE_ALL
            };

            // 4. Memory optimization
            sessionOptions.EnableCpuMemArena = true;
            sessionOptions.EnableMemoryPattern = true;

            // 5. CPU spinning
            sessionOptions.AddSessionConfigEntry(
                "session.intra_op.allow_spinning",
                (isMac || config.IntraOpAllowSpinning) ? "1" : "0");

            // 6. CoreML (macOS only, when enabled)
            //    GPU acceleration (Windows/Linux=Vulkan, macOS=Metal) is provided by the ggml plugin EP;
            //    see GgmlExecutionProvider and LoadVocoderWithFallback.
            if (isMac && useCoreML)
            {
                try
                {
                    sessionOptions.EnableMemoryPattern = false;

                    var coremlOptions = new Dictionary<string, string>
                    {
                        ["EnableOnSubgraphs"] = (config.CoreMLFlags & 0x2) != 0 ? "1" : "0",
                        ["MLComputeUnits"] = (config.CoreMLFlags & 0x4) != 0 ? "ANEOnly" : "ALL",
                        ["ModelFormat"] = (config.CoreMLFlags & 0x10) != 0 ? "MLProgram" : "NeuralNetwork",
                        ["RequireStaticInputShapes"] = (config.CoreMLFlags & 0x8) != 0 ? "1" : "0",
                    };
                    sessionOptions.AppendExecutionProvider("CoreML", coremlOptions);
                    DebugLog($"CoreML acceleration enabled (macOS) flags=0x{config.CoreMLFlags:X}");
                }
                catch (Exception ex)
                {
                    DebugLog($"⚠️ Failed to enable CoreML acceleration, fell back to CPU: {ex.Message}");
                }
            }

            return sessionOptions;
        }

        /// <summary>
        /// Load a model, automatically falling back to CPU if CoreML compilation fails.
        /// </summary>
        private InferenceSession LoadModelWithFallback(string modelPath, SolTTS.Core.Config.TTSConfig config)
        {
            bool isMac = RuntimeInformation.IsOSPlatform(OSPlatform.OSX);
            // On macOS with CoreML enabled, try CoreML first and fall back to CPU on failure
            if (isMac && config.UseCoreML)
            {
                try
                {
                    return new InferenceSession(modelPath, GetSessionOptions(config, useCoreML: true));
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"⚠️ [CoreML] Compilation failed for model '{Path.GetFileName(modelPath)}', falling back to CPU: {ex.Message}");
                    // CoreML compilation failed; fall back to pure CPU
                    return new InferenceSession(modelPath, GetSessionOptions(config, useCoreML: false));
                }
            }
            return new InferenceSession(modelPath, GetSessionOptions(config));
        }

        /// <summary>
        /// Load the vocoder. In hybrid mode (UseVocoderGgml) only the vocoder uses the GPU;
        /// load failures fall back to CPU automatically with no availability impact.
        /// </summary>
        private InferenceSession LoadVocoderWithFallback(string modelPath, SolTTS.Core.Config.TTSConfig config)
        {
            // ggml plugin EP (Windows/Linux=Vulkan, macOS=Metal) — measured ~5x kernel speedup, try first.
            if (config.UseVocoderGgml)
            {
                try
                {
                    var so = GgmlExecutionProvider.TryCreateSessionOptions(device: "gpu", gpuMinWeightMb: 0);
                    if (so != null)
                    {
                        using (so)
                        {
                            var session = new InferenceSession(modelPath, so);
                            DebugLog("[CPU+GPU hybrid] Vocoder is using ggml (Vulkan/Metal); all other models stay on CPU");
                            return session;
                        }
                    }
                    Console.WriteLine("⚠️ [Hybrid mode] ggml EP unavailable (missing library or Vulkan), falling back to CPU.");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"⚠️ [Hybrid mode] Vocoder ggml load failed, falling back to CPU: {ex.Message}");
                }
            }
            return LoadModelWithFallback(modelPath, config);
        }

        /// <summary>
        /// Get the macOS performance-core (P-Core) count for setting the optimal IntraOpNumThreads.
        /// For Apple Silicon: M1=4P, M2=4P, M3=4P, M2 Pro=6/8P, M3 Pro=5/6P, etc.
        /// For Intel Mac: returns the physical core count.
        /// </summary>
        private static int GetMacPerformanceCoreCount()
        {
            try
            {
                // sysctl hw.perflevel0.physicalcpu returns the P-core count (Apple Silicon only)
                int p = ReadSysctlInt("hw.perflevel0.physicalcpu");
                if (p > 0) return p;
                // Fall back to the physical core count
                int phys = ReadSysctlInt("hw.physicalcpu");
                if (phys > 0) return phys;
            }
            catch { }
            // Final fallback: logical cores / 2
            int lp = Environment.ProcessorCount;
            return Math.Max(1, lp / 2);
        }

        private static int ReadSysctlInt(string key)
        {
            try
            {
                var psi = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "sysctl",
                    Arguments = $"-n {key}",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };
                using var p = System.Diagnostics.Process.Start(psi);
                if (p == null) return 0;
                string output = p.StandardOutput.ReadToEnd().Trim();
                p.WaitForExit(500);
                return int.TryParse(output, out int v) ? v : 0;
            }
            catch { return 0; }
        }

        public void LoadModels(string modelDir, SolTTS.Core.Config.TTSConfig config, string? hubertPath = null, string? svPath = null)
        {
            _config = config;
            this.DebugMode = config.DebugMode;
            var opt = GetSessionOptions(config);

            Parallel.Invoke(
                () => _t2sEncoder = LoadModelWithFallback(Path.Combine(modelDir, "t2s_encoder.onnx"), config),
                () => _firstStageDecoder = LoadModelWithFallback(Path.Combine(modelDir, "t2s_first_stage_decoder.onnx"), config),
                () => _stageDecoder = LoadStageDecoder(modelDir, config),
                () => _stageDecoderLoop = LoadStageDecoderLoop(modelDir, config),
                () => _vocoder = LoadVocoderWithFallback(Path.Combine(modelDir, "vits.onnx"), config),
                () =>
                {
                    string promptEncoderPath = Path.Combine(modelDir, "prompt_encoder.onnx");
                    if (File.Exists(promptEncoderPath))
                    {
                        _promptEncoder = LoadModelWithFallback(promptEncoderPath, config);
                    }
                },
                () =>
                {
                    if (!string.IsNullOrEmpty(hubertPath) && File.Exists(hubertPath))
                    {
                        _hubert = LoadModelWithFallback(hubertPath, config);
                    }
                },
                () =>
                {
                    if (!string.IsNullOrEmpty(svPath) && File.Exists(svPath))
                    {
                        _svModel = LoadModelWithFallback(svPath, config);
                    }
                }
            );
        }

        /// <summary>
        /// 加载 stage decoder，优先使用 QKV Split 融合变体。
        ///
        /// 原图每层把 Q/K/V 用三次 Slice 切开；融合版用一次 Split，少 48 个节点。
        /// 两者逐位等价（已验证 51/51 输出 diff=0），融合版只是少了两次算子调度。
        /// 融合模型不存在时自动回退原图。
        /// </summary>
        private InferenceSession LoadStageDecoder(string modelDir, SolTTS.Core.Config.TTSConfig config)
        {
            string fused = Path.Combine(modelDir, "t2s_stage_decoder_qkv.onnx");
            if (config.UseFusedQkvDecoder && File.Exists(fused))
            {
                try
                {
                    var s = LoadModelWithFallback(fused, config);
                    DebugLog("[StageDecoder] 使用 QKV Split 融合模型");
                    return s;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"⚠️ [StageDecoder] 融合模型加载失败，回退原图: {ex.Message}");
                }
            }
            return LoadModelWithFallback(Path.Combine(modelDir, "t2s_stage_decoder.onnx"), config);
        }

        /// <summary>
        /// Load the optional Loop merged-decoding model. Loaded only when the switch is on and the file exists;
        /// any failure returns null and the caller falls back to the per-token path.
        /// </summary>
        private InferenceSession? LoadStageDecoderLoop(string modelDir, SolTTS.Core.Config.TTSConfig config)
        {
            if (!config.UseStageDecoderLoopModel) return null;

            string path = Path.Combine(modelDir, "t2s_stage_decoder_loop.onnx");
            if (!File.Exists(path))
            {
                if (config.DebugMode)
                    Console.WriteLine($"[StageDecoder] Loop model not found, falling back to the per-token path: {path}");
                return null;
            }

            try
            {
                var session = LoadModelWithFallback(path, config);
                DebugLog("[StageDecoder] Loop merged-decoding model loaded");
                return session;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️ [StageDecoder] Loop model load failed, falling back to per-token: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Share model sessions from another instance
        /// </summary>
        public void ShareModelsFrom(InferenceEngineV1 other)
        {
            _t2sEncoder = other._t2sEncoder;
            _firstStageDecoder = other._firstStageDecoder;
            _stageDecoder = other._stageDecoder;
            _stageDecoderLoop = other._stageDecoderLoop;
            _vocoder = other._vocoder;
            _promptEncoder = other._promptEncoder;
            _hubert = other._hubert;
            _svModel = other._svModel;
            _config = other._config;
            IsShared = true;
        }

        public float[] GetHubertContent(float[] audio16k)
        {
            if (_hubert == null) throw new InvalidOperationException("Hubert model not loaded.");

            // Detect input name: the base version uses "source", the full version uses "input_values"
            string inputName = _hubert.InputMetadata.Keys.Contains("input_values") ? "input_values" : "source";
            var inputTensor = new DenseTensor<float>(audio16k, new[] { 1, audio16k.Length });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };

            using var results = _hubert.Run(inputs);
            var tensor = results.First().AsTensor<float>(); // [1, T, 768] or [1, 768, T]

            int dim1 = tensor.Dimensions[1];
            int dim2 = tensor.Dimensions[2];

            // Important: T2S expects [1, 768, T].
            // If the output is already [1, 768, T], return it as-is.
            if (dim1 == 768)
            {
                var arr = tensor.ToArray();
                Console.WriteLine($"[Hubert] Output shape: [1, {dim1}, {dim2}] (direct)");
                if (arr.Length > 5)
                    Console.WriteLine($"[Hubert] First 5: {string.Join(", ", arr.Take(5))}");
                return arr;
            }

            // If the output is [1, T, 768], transpose manually to [1, 768, T]
            // to avoid interleaved feature vectors causing noise
            int T = dim1;
            int C = dim2; // 768
            float[] transposed = new float[T * C];

            for (int t = 0; t < T; t++)
            {
                for (int c = 0; c < C; c++)
                {
                    transposed[c * T + t] = tensor[0, t, c];
                }
            }
            float sum = 0;
            float sumSq = 0;
            foreach (var f in transposed)
            {
                sum += f;
                sumSq += f * f;
            }
            float mean = sum / transposed.Length;
            float std = (float)Math.Sqrt(sumSq / transposed.Length - mean * mean);

            Console.WriteLine($"[Hubert] Output mean={mean}, std={std}");
            Console.WriteLine($"[Hubert] Output shape: [1, {C}, {T}]");
            if (transposed.Length > 5)
                Console.WriteLine($"[Hubert] First 5: {string.Join(", ", transposed.Take(5))}");

            return transposed;
        }

        public float[] GetSpeakerEmbedding(float[] audio16k)
        {
            if (_svModel == null) throw new InvalidOperationException("SV model not loaded.");

            // Input names according to the metadata: "waveform"
            var inputTensor = new DenseTensor<float>(audio16k, new[] { 1, audio16k.Length });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("waveform", inputTensor) };

            using var results = _svModel.Run(inputs);
            return results.First().AsTensor<float>().ToArray();
        }

        public (float[] ge, float[] ge_advanced) GetPromptEmbedding(float[] audio32k, float[] svEmb)
        {
            if (_promptEncoder == null) throw new InvalidOperationException("Prompt Encoder not loaded.");

            // Input names according to the metadata: "ref_audio" and "sv_emb"
            var audioTensor = new DenseTensor<float>(audio32k, new[] { 1, audio32k.Length });
            var svTensor = new DenseTensor<float>(svEmb, new[] { 1, svEmb.Length });

            Console.WriteLine($"[PromptEncoder] svEmb length: {svEmb.Length}");
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("ref_audio", audioTensor),
                NamedOnnxValue.CreateFromTensor("sv_emb", svTensor)
            };

            using var results = _promptEncoder.Run(inputs);
            // Output [0]: ge, [1]: ge_advanced
            var ge = results[0].AsTensor<float>().ToArray();
            var geAdvanced = results[1].AsTensor<float>().ToArray();
            Console.WriteLine($"[PromptEncoder] ge length: {ge.Length}, geAdvanced length: {geAdvanced.Length}");
            return (ge, geAdvanced);
        }

        public long[] RunT2S(long[] textSeq, float[,] textBert, long[] refSeq, float[,] refBert, float[] sslContent, PhoneLanguage[]? languageTags = null, TtsOptions? options = null)
        {
            options = ApplyConfig(options);
            var phaseSw = PerfStats.Enabled ? System.Diagnostics.Stopwatch.StartNew() : null;
            int stageSteps = 0;
            if (_t2sEncoder == null || _firstStageDecoder == null || _stageDecoder == null)
                throw new InvalidOperationException("T2S models not loaded.");

            if (sslContent == null || sslContent.Length == 0)
                throw new ArgumentException("sslContent must not be null and must have non-zero length. This usually means reference audio loading failed or feature extraction was abnormal; check the audio paths in config.json.");

            if (sslContent.Length % 768 != 0)
                throw new ArgumentException($"sslContent length {sslContent.Length} is invalid; it must be a multiple of 768.");

            float[] refBertFlat = Flatten(refBert);
            float[] textBertFlat = Flatten(textBert);
            try
            {
                // 1. Encoder inference
                var encoderInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("ref_seq", new DenseTensor<long>(refSeq, new[] { 1, refSeq.Length })),
                    NamedOnnxValue.CreateFromTensor("text_seq", new DenseTensor<long>(textSeq, new[] { 1, textSeq.Length })),
                    NamedOnnxValue.CreateFromTensor("ref_bert", new DenseTensor<float>(refBertFlat.AsMemory(0, refBert.Length), new[] { (int)refBert.GetLength(0), (int)refBert.GetLength(1) })),
                    NamedOnnxValue.CreateFromTensor("text_bert", new DenseTensor<float>(textBertFlat.AsMemory(0, textBert.Length), new[] { (int)textBert.GetLength(0), (int)textBert.GetLength(1) })),
                    NamedOnnxValue.CreateFromTensor("ssl_content", new DenseTensor<float>(sslContent, new[] { 1, 768, sslContent.Length / 768 }))
                };

                // Detailed input logging (debug mode only)
                if (DebugMode)
                {
                    Console.WriteLine($"[T2S Input] ref_seq shape: [1, {refSeq.Length}]");
                    Console.WriteLine($"[T2S Input] text_seq shape: [1, {textSeq.Length}]");
                    Console.WriteLine($"[T2S Input] ref_bert shape: [{refBert.GetLength(0)}, {refBert.GetLength(1)}], first 3: {refBertFlat[0]}, {refBertFlat[1]}, {refBertFlat[2]}");
                    Console.WriteLine($"[T2S Input] text_bert shape: [{textBert.GetLength(0)}, {textBert.GetLength(1)}], first 3: {textBertFlat[0]}, {textBertFlat[1]}, {textBertFlat[2]}");
                    Console.WriteLine($"[T2S Input] ssl_content shape: [1, 768, {sslContent.Length / 768}], mean: {sslContent.Average()}, first 3: {string.Join(", ", sslContent.Take(3))}");
                }

                using var encoderResults = _t2sEncoder.Run(encoderInputs);

                if (encoderResults.Count < 2) throw new Exception("Expected at least 2 outputs from T2S Encoder");

                var val0 = encoderResults[0].Value;
                var val1 = encoderResults[1].Value;

                Tensor<float> x;
                if (val0 is Tensor<float> tf) x = tf.Clone();
                else throw new Exception($"Output 'x' is not Tensor<float>. Actual: {val0?.GetType().FullName}");

                Tensor<long> prompts;
                if (val1 is Tensor<long> pl) prompts = pl.Clone();
                else throw new Exception($"Output 'prompts' is not Tensor<long>. Actual: {val1?.GetType().FullName}");

                if (DebugMode)
                {
                    var xArr = x.ToArray();
                    DebugLog($"[T2S Encoder] x shape: [{x.Dimensions[0]}, {x.Dimensions[1]}, {x.Dimensions[2]}]");
                    DebugLog($"[T2S Encoder] x mean: {xArr.Average()}");
                }

                // 3. Global retry loop (First Stage and Stage Decoder)
                // Reason: First Stage Decoder output is also non-deterministic (includes random sampling) and
                // sometimes produces a "bad" state that makes the Stage Decoder fail immediately.
                // The whole decoding process must therefore be retried.

                // Dynamic threshold: language-aware multiplier
                int minExpectedTokens = CalculateMinExpectedTokens(textSeq, languageTags, options);
                if (DebugMode) DebugLog($"[T2S] Min expected tokens: {minExpectedTokens} (textSeq.Length={textSeq.Length})");

                List<long> generatedTokens = new List<long>();
                int maxRetries = 10;
                int retryCount = 0;
                bool success = false;

                while (retryCount < maxRetries)
                {
                    generatedTokens.Clear();

                    // --- A. Run First Stage Decoder ---
                    // Re-run on every retry to obtain fresh random sampling state
                    var firstStageInputs = new List<NamedOnnxValue>
                 {
                     NamedOnnxValue.CreateFromTensor("x", x),
                     NamedOnnxValue.CreateFromTensor("prompts", prompts)
                 };

                    using var firstStageResults = _firstStageDecoder.Run(firstStageInputs);

                    var yTensor = firstStageResults.First(r => r.Name == "y").AsTensor<long>();
                    var yEmbTensor = firstStageResults.First(r => r.Name == "y_emb").AsTensor<float>();

                    if (DebugMode)
                    {
                        var yArr = yTensor.ToArray();
                        DebugLog($"[FirstStage] Attempt {retryCount} y mean: {yArr.Average()}");

                        // Check whether the First Stage result already contains EOS
                        if (yArr.Any(t => t >= 1024))
                        {
                            DebugLog($"[T2S Warning] First Stage output already contains EOS. Retry {retryCount} likely to fail.");
                        }
                    }

                    // Capture First Stage Token (The first semantic token)
                    var firstDense = (DenseTensor<long>)yTensor;
                    generatedTokens.Add(firstDense.Buffer.Span[firstDense.Buffer.Length - 1]);

                    // KV cache references the First Stage result tensors directly (no Clone);
                    // firstStageResults stays alive via using until this decoding step completes.
                    var currentKvCache = new Tensor<float>[KvLayerCount * 2];
                    for (int layer = 0; layer < KvLayerCount; layer++)
                    {
                        currentKvCache[layer * 2] = firstStageResults.First(r => r.Name == PresentKNames[layer]).AsTensor<float>();
                        currentKvCache[layer * 2 + 1] = firstStageResults.First(r => r.Name == PresentVNames[layer]).AsTensor<float>();
                    }

                    // --- B. Run Stage Decoder Loop ---
                    var currentYTensor = yTensor;
                    var currentYEmbTensor = yEmbTensor;
                    int numLayers = KvLayerCount;

                    if (_stageDecoderLoop != null)
                    {
                        // Merged decoding: one Run performs the whole autoregressive loop (trip=500).
                        // Measured no significant gain (1.02× / 0.98×); kept as an optional optimization baseline.
                        var loopInputs = new List<NamedOnnxValue>(2 + 2 + KvLayerCount * 2)
                        {
                            NamedOnnxValue.CreateFromTensor("loop_max_steps",
                                new DenseTensor<long>(new[] { 500L }, Array.Empty<int>())),
                            NamedOnnxValue.CreateFromTensor("loop_cond_init",
                                new DenseTensor<bool>(new[] { true }, Array.Empty<int>())),
                            NamedOnnxValue.CreateFromTensor("iy", currentYTensor),
                            NamedOnnxValue.CreateFromTensor("iy_emb", currentYEmbTensor),
                        };
                        for (int layer = 0; layer < numLayers; layer++)
                        {
                            loopInputs.Add(NamedOnnxValue.CreateFromTensor(PastKNames[layer], currentKvCache[layer * 2]));
                            loopInputs.Add(NamedOnnxValue.CreateFromTensor(PastVNames[layer], currentKvCache[layer * 2 + 1]));
                        }

                        using var loopResults = _stageDecoderLoop.Run(loopInputs);
                        var finalY = loopResults.First(r => r.Name == "y_out").AsTensor<long>();
                        var finalDense = (DenseTensor<long>)finalY;

                        // New tokens = final y minus the history length at loop entry
                        int historyLen = (int)currentYTensor.Dimensions[1];
                        for (int i = historyLen; i < finalDense.Buffer.Length; i++)
                            generatedTokens.Add(finalDense.Buffer.Span[i]);

                        stageSteps += generatedTokens.Count;
                    }
                    else
                    {
                    // Lifetime: hold the previous step's result set until the next Run has consumed its tensors.
                    IDisposable? stageHolder = null;
                    var stageInputs = new List<NamedOnnxValue>(2 + KvLayerCount * 2);

                    try
                    {
                        for (int i = 0; i < 500; i++)
                        {
                            stageInputs.Clear();
                            stageInputs.Add(NamedOnnxValue.CreateFromTensor("iy", currentYTensor));
                            stageInputs.Add(NamedOnnxValue.CreateFromTensor("iy_emb", currentYEmbTensor));

                            for (int layer = 0; layer < numLayers; layer++)
                            {
                                stageInputs.Add(NamedOnnxValue.CreateFromTensor(PastKNames[layer], currentKvCache[layer * 2]));
                                stageInputs.Add(NamedOnnxValue.CreateFromTensor(PastVNames[layer], currentKvCache[layer * 2 + 1]));
                            }

                            var stageResults = _stageDecoder.Run(stageInputs);
                            stageSteps++;

                            // The old results (this step's inputs) are fully consumed by this Run, so they can now be released
                            stageHolder?.Dispose();
                            stageHolder = stageResults;

                            currentYTensor = stageResults.First(r => r.Name == "y").AsTensor<long>();
                            currentYEmbTensor = stageResults.First(r => r.Name == "y_emb").AsTensor<float>();
                            var stopCondition = stageResults.First(r => r.Name == "stop_condition_tensor").AsTensor<bool>().GetValue(0);

                            for (int layer = 0; layer < numLayers; layer++)
                            {
                                currentKvCache[layer * 2] = stageResults.First(r => r.Name == PresentKNames[layer]).AsTensor<float>();
                                currentKvCache[layer * 2 + 1] = stageResults.First(r => r.Name == PresentVNames[layer]).AsTensor<float>();
                            }

                            // Read the tensor buffer tail directly to avoid ToArray()
                            var lastDense = (DenseTensor<long>)currentYTensor;
                            var lastToken = lastDense.Buffer.Span[lastDense.Buffer.Length - 1];
                            generatedTokens.Add(lastToken);

                            if (stopCondition) break;
                        }
                    }
                    finally
                    {
                        stageHolder?.Dispose();
                    }
                    }

                    // --- C. Validate the result ---
                    // Use the dynamic threshold: success when the generated token count meets expectations
                    if (generatedTokens.Count >= minExpectedTokens)
                    {
                        success = true;
                        break;
                    }

                    retryCount++;
                    if (DebugMode) DebugLog($"[T2S] Insufficient tokens ({generatedTokens.Count} < {minExpectedTokens}). Retrying {retryCount}/{maxRetries}...");

                }

                if (!success)
                {
                    // Degrade gracefully: like the other paths, when retries are exhausted return whatever tokens
                    // are available instead of failing the whole sentence over sampling quality.
                    if (generatedTokens.Count > 0)
                    {
                        Console.WriteLine(
                            $"⚠️ [T2S] Retries exhausted, degrading to the generated content " +
                            $"(tokens={generatedTokens.Count}, expected={minExpectedTokens})");
                    }
                    else
                    {
                        throw new Exception($"T2S inference failed: no valid semantic sequence after {maxRetries} retries. Check the model state.");
                    }
                }

                DebugLog($"[T2S] Total tokens in list: {generatedTokens.Count}");

                // Strip EOS
                var eosIndex = generatedTokens.FindIndex(t => t >= 1024);
                long[] finalSemantic = (eosIndex >= 0)
                    ? generatedTokens.Take(eosIndex).ToArray()
                    : generatedTokens.ToArray();

                // Match Python: y[0, -1] = 0
                if (finalSemantic.Length > 0)
                {
                    finalSemantic[finalSemantic.Length - 1] = 0;
                    DebugLog("[T2S] Set last token to 0 (matching Python)");
                }

                return finalSemantic;
            }
            finally
            {
                ReturnFlat(refBertFlat);
                ReturnFlat(textBertFlat);
                if (phaseSw != null) PerfStats.AddT2S(phaseSw, textSeq.Length, stageSteps);
            }
        }

        /// <summary>
        /// Streaming T2S inference: invoke the vocoder and emit an audio chunk every chunkSize tokens during decoding.
        /// </summary>
        /// <param name="textSeq">Text sequence</param>
        /// <param name="textBert">Text BERT features</param>
        /// <param name="refSeq">Reference sequence</param>
        /// <param name="refBert">Reference BERT features</param>
        /// <param name="sslContent">SSL content</param>
        /// <param name="refAudio32k">Reference audio (32kHz)</param>
        /// <param name="ge">Optional: Prompt Encoder ge output</param>
        /// <param name="geAdvanced">Optional: Prompt Encoder ge_advanced output</param>
        /// <param name="chunkSize">Token count between audio callbacks</param>
        /// <param name="onAudioChunk">Audio chunk callback (32kHz sample rate)</param>
        public void RunT2SStreaming(
            long[] textSeq, float[,] textBert,
            long[] refSeq, float[,] refBert, float[] sslContent,
            float[] refAudio32k, float[]? ge, float[]? geAdvanced,
            int chunkSize,
            Action<float[], bool> onAudioChunk,
            float speed = 1.0f,
            PhoneLanguage[]? languageTags = null,
            TtsOptions? options = null)
        {
            options = ApplyConfig(options);
            if (_t2sEncoder == null || _firstStageDecoder == null || _stageDecoder == null || _vocoder == null)
                throw new InvalidOperationException("Models not loaded.");

            // === 1. T2S Encoder ===
            float[] refBertFlat = Flatten(refBert);
            float[] textBertFlat = Flatten(textBert);
            try
            {
                var encoderInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("ref_seq", new DenseTensor<long>(refSeq, new[] { 1, refSeq.Length })),
                    NamedOnnxValue.CreateFromTensor("text_seq", new DenseTensor<long>(textSeq, new[] { 1, textSeq.Length })),
                    NamedOnnxValue.CreateFromTensor("ref_bert", new DenseTensor<float>(refBertFlat.AsMemory(0, refBert.Length), new[] { refBert.GetLength(0), refBert.GetLength(1) })),
                    NamedOnnxValue.CreateFromTensor("text_bert", new DenseTensor<float>(textBertFlat.AsMemory(0, textBert.Length), new[] { textBert.GetLength(0), textBert.GetLength(1) })),
                    NamedOnnxValue.CreateFromTensor("ssl_content", new DenseTensor<float>(sslContent, new[] { 1, 768, sslContent.Length / 768 }))
                };

                using var encoderResults = _t2sEncoder.Run(encoderInputs);
                var x = encoderResults.First(r => r.Name == "x").AsTensor<float>().Clone();
                var prompts = encoderResults.First(r => r.Name == "prompts").AsTensor<long>().Clone();

                // === 2. T2S Decoder Loop with Streaming and Retry ===
                // Dynamic threshold: language-aware multiplier
                int minExpectedTokens = CalculateMinExpectedTokens(textSeq, languageTags, options);
                int maxRetries = options.MaxRetries!.Value;
                int retryCount = 0;
                bool success = false;

                while (retryCount < maxRetries)
                {
                    List<long> generatedTokens = new List<long>();
                    int lastVocoderPosition = 0;

                    // First Stage Decoder
                    var firstStageInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("x", x),
                    NamedOnnxValue.CreateFromTensor("prompts", prompts)
                };
                    using var firstStageResults = _firstStageDecoder.Run(firstStageInputs);

                    var yTensor = firstStageResults.First(r => r.Name == "y").AsTensor<long>().Clone();
                    var yEmbTensor = firstStageResults.First(r => r.Name == "y_emb").AsTensor<float>().Clone();

                    // Check whether First Stage output EOS directly
                    var firstToken = yTensor.ToArray().Last();
                    if (firstToken >= 1024)
                    {
                        retryCount++;
                        if (DebugMode) DebugLog($"[Streaming] First Stage output EOS directly, retrying {retryCount}/{maxRetries}...");
                        continue;
                    }

                    generatedTokens.Add(firstToken);

                    // KV Cache
                    var kvCache = new List<Tensor<float>>();
                    for (int layer = 0; layer < 24; layer++)
                    {
                        kvCache.Add(firstStageResults.First(r => r.Name == $"present_k_layer_{layer}").AsTensor<float>().Clone());
                        kvCache.Add(firstStageResults.First(r => r.Name == $"present_v_layer_{layer}").AsTensor<float>().Clone());
                    }

                    // Stage Decoder Loop
                    var currentYTensor = yTensor.Clone();
                    var currentYEmbTensor = yEmbTensor.Clone();
                    int numLayers = kvCache.Count / 2;

                    for (int i = 0; i < 500; i++)
                    {
                        var stageInputs = new List<NamedOnnxValue>();
                        stageInputs.Add(NamedOnnxValue.CreateFromTensor("iy", currentYTensor));
                        stageInputs.Add(NamedOnnxValue.CreateFromTensor("iy_emb", currentYEmbTensor));

                        for (int layer = 0; layer < numLayers; layer++)
                        {
                            stageInputs.Add(NamedOnnxValue.CreateFromTensor($"past_k_layer_{layer}", kvCache[layer * 2]));
                            stageInputs.Add(NamedOnnxValue.CreateFromTensor($"past_v_layer_{layer}", kvCache[layer * 2 + 1]));
                        }

                        using var stageResults = _stageDecoder.Run(stageInputs);
                        currentYTensor = stageResults.First(r => r.Name == "y").AsTensor<long>().Clone();
                        currentYEmbTensor = stageResults.First(r => r.Name == "y_emb").AsTensor<float>().Clone();
                        var stopCondition = stageResults.First(r => r.Name == "stop_condition_tensor").AsTensor<bool>().First();

                        kvCache.Clear();
                        for (int layer = 0; layer < numLayers; layer++)
                        {
                            kvCache.Add(stageResults.First(r => r.Name == $"present_k_layer_{layer}").AsTensor<float>().Clone());
                            kvCache.Add(stageResults.First(r => r.Name == $"present_v_layer_{layer}").AsTensor<float>().Clone());
                        }

                        var lastToken = currentYTensor.ToArray().Last();
                        generatedTokens.Add(lastToken);

                        // === Streaming output check ===
                        int currentTokenCount = generatedTokens.Count;
                        bool shouldEmitChunk = (currentTokenCount - lastVocoderPosition) >= chunkSize;
                        bool isEOS = stopCondition || lastToken >= 1024;

                        if (shouldEmitChunk || isEOS)
                        {
                            // Prepare semantic tokens (strip EOS)
                            int tokenLen = 0;
                            while (tokenLen < generatedTokens.Count && generatedTokens[tokenLen] < 1024) tokenLen++;

                            if (tokenLen > 0)
                            {
                                long[] semanticTokens = new long[tokenLen];
                                for (int k = 0; k < tokenLen; k++) semanticTokens[k] = generatedTokens[k];

                                // Set the last token to 0 only on the final chunk (matching Python)
                                if (isEOS)
                                {
                                    semanticTokens[semanticTokens.Length - 1] = 0;
                                }

                                // Invoke the vocoder with the speed parameter
                                var audio = RunVocoder(textSeq, semanticTokens, refAudio32k, out int audioLen, ge, geAdvanced, speed);
                                try
                                {
                                    // Extract the current audio segment and copy it (RunT2SStreaming's non-zero-GC path keeps old behavior but fixes the length)
                                    float[] chunk = new float[audioLen];
                                    Array.Copy(audio, 0, chunk, 0, audioLen);
                                    onAudioChunk(chunk, isEOS);
                                }
                                finally
                                {
                                    ReturnAudioBuffer(audio);
                                }
                                lastVocoderPosition = currentTokenCount;
                            }
                        }

                        if (isEOS) break;
                    }

                    // Validate the result: use the dynamic threshold
                    if (generatedTokens.Count >= minExpectedTokens)
                    {
                        success = true;
                        break;
                    }

                    retryCount++;
                    if (DebugMode) DebugLog($"[Streaming] Insufficient tokens ({generatedTokens.Count} < {minExpectedTokens}), retrying {retryCount}/{maxRetries}...");
                }

                if (!success)
                {
                    throw new Exception($"Streaming T2S inference failed: no valid semantic sequence after {options.MaxRetries} retries.");
                }
            }
            finally
            {
                ReturnFlat(refBertFlat);
                ReturnFlat(textBertFlat);
            }
        }

        /// <summary>
        /// Async streaming T2S: emits token chunks instead of audio; the caller drives the vocoder on a separate thread.
        /// </summary>
        public void RunT2SStreamingTokens(
            long[] textSeq, float[,] textBert,
            long[] refSeq, float[,] refBert, float[] sslContent,
            int chunkSize,
            Action<long[], bool> onTokenChunk,  // (tokensSoFar, isFinal)
            Action? onRetry = null,
            PhoneLanguage[]? languageTags = null,  // languageTags added
            TtsOptions? options = null)
        {
            options = ApplyConfig(options);
            if (_t2sEncoder == null || _firstStageDecoder == null || _stageDecoder == null)
                throw new InvalidOperationException("T2S models not loaded.");

            float[] refBertFlat = Flatten(refBert);
            float[] textBertFlat = Flatten(textBert);
            try
            {
                var encoderInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("ref_seq", new DenseTensor<long>(refSeq, new[] { 1, refSeq.Length })),
                    NamedOnnxValue.CreateFromTensor("text_seq", new DenseTensor<long>(textSeq, new[] { 1, textSeq.Length })),
                    NamedOnnxValue.CreateFromTensor("ref_bert", new DenseTensor<float>(refBertFlat.AsMemory(0, refBert.Length), new[] { (int)refBert.GetLength(0), (int)refBert.GetLength(1) })),
                    NamedOnnxValue.CreateFromTensor("text_bert", new DenseTensor<float>(textBertFlat.AsMemory(0, textBert.Length), new[] { (int)textBert.GetLength(0), (int)textBert.GetLength(1) })),
                    NamedOnnxValue.CreateFromTensor("ssl_content", new DenseTensor<float>(sslContent, new[] { 1, 768, sslContent.Length / 768 }))
                };

                using var encoderResults = _t2sEncoder.Run(encoderInputs);
                var x = encoderResults.First(r => r.Name == "x").AsTensor<float>().Clone();
                var prompts = encoderResults.First(r => r.Name == "prompts").AsTensor<long>().Clone();

                // Dynamic threshold: language-aware multiplier
                int minExpectedTokens = CalculateMinExpectedTokens(textSeq, languageTags, options);
                int maxRetries = options.MaxRetries!.Value;
                int retryCount = 0;
                bool success = false;

                // A first token of EOS is sampling noise (in-graph RandomNormalLike) and is not charged to the
                // "insufficient generation" retry budget; cap it separately to avoid long sentences failing on repeated EOS.
                int eosFirstTokenCount = 0;
                const int maxEosFirstTokenRetries = 12;

                // Degradation fallback: record the longest generation across attempts.
                long[] lastGoodTokens = Array.Empty<long>();
                int lastGoodTokenCount = 0;

                while (retryCount < maxRetries)
                {
                    // Notify the caller to clear tokens from the previous failed attempt on retry
                    if (retryCount > 0)
                    {
                        onRetry?.Invoke();
                        if (DebugMode) DebugLog($"[T2S Streaming] Retry {retryCount}/{maxRetries}, clearing previous tokens...");
                    }

                    List<long> generatedTokens = new List<long>();
                    int lastEmitPosition = 0;


                    // Deferred container release: holds references to the KV Cache data
                    IDisposable? activeResults = null;

                    // First Stage Setup
                    var firstStageInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("x", x),
                    NamedOnnxValue.CreateFromTensor("prompts", prompts)
                };

                    // Manually manage the firstStageResults lifetime
                    var firstStageResults = _firstStageDecoder.Run(firstStageInputs);
                    activeResults = firstStageResults; // Hold it

                    var yTensor = firstStageResults.First(r => r.Name == "y").AsTensor<long>();
                    // Note: no Clone here; relies on activeResults staying alive
                    var yEmbTensor = firstStageResults.First(r => r.Name == "y_emb").AsTensor<float>();

                    // Safe access because activeResults is alive
                    var firstToken = ((DenseTensor<long>)yTensor).Buffer.Span[((DenseTensor<long>)yTensor).Buffer.Length - 1];
                    if (firstToken >= 1024)
                    {
                        // Sampling noise made the first token EOS: resample without consuming the formal retry budget.
                        eosFirstTokenCount++;
                        activeResults.Dispose();
                        if (eosFirstTokenCount >= maxEosFirstTokenRetries)
                        {
                            throw new Exception(
                                $"Streaming T2S first token was EOS {eosFirstTokenCount} times in a row (sampling anomaly). " +
                                $"textSeq.Length={textSeq.Length}, minExpectedTokens={minExpectedTokens}");
                        }
                        continue;
                    }

                    generatedTokens.Add(firstToken);

                    // Initialize KV Cache (No Clone, references memory in firstStageResults)
                    var kvCache = new Tensor<float>[KvLayerCount * 2];
                    for (int layer = 0; layer < KvLayerCount; layer++)
                    {
                        kvCache[layer * 2] = firstStageResults.First(r => r.Name == PresentKNames[layer]).AsTensor<float>();
                        kvCache[layer * 2 + 1] = firstStageResults.First(r => r.Name == PresentVNames[layer]).AsTensor<float>();
                    }

                    var currentYTensor = yTensor;
                    var currentYEmbTensor = yEmbTensor;
                    int numLayers = KvLayerCount;

                    if (_stageDecoderLoop != null)
                    {
                        // Merged decoding (streaming batches): each Run advances chunkSize steps and feeds the KV
                        // back into the next call, preserving chunked output while merging chunkSize Runs into 1.
                        var batchInputs = new List<NamedOnnxValue>(4 + KvLayerCount * 2);
                        int stepBudget = 500;
                        while (stepBudget > 0)
                        {
                            int trip = Math.Min(chunkSize, stepBudget);
                            batchInputs.Clear();
                            batchInputs.Add(NamedOnnxValue.CreateFromTensor("loop_max_steps",
                                new DenseTensor<long>(new[] { (long)trip }, Array.Empty<int>())));
                            batchInputs.Add(NamedOnnxValue.CreateFromTensor("loop_cond_init",
                                new DenseTensor<bool>(new[] { true }, Array.Empty<int>())));
                            batchInputs.Add(NamedOnnxValue.CreateFromTensor("iy", currentYTensor));
                            batchInputs.Add(NamedOnnxValue.CreateFromTensor("iy_emb", currentYEmbTensor));
                            for (int layer = 0; layer < numLayers; layer++)
                            {
                                batchInputs.Add(NamedOnnxValue.CreateFromTensor(PastKNames[layer], kvCache[layer * 2]));
                                batchInputs.Add(NamedOnnxValue.CreateFromTensor(PastVNames[layer], kvCache[layer * 2 + 1]));
                            }

                            var batchResults = _stageDecoderLoop.Run(batchInputs);
                            activeResults?.Dispose();
                            activeResults = batchResults;

                            int historyLen = (int)currentYTensor.Dimensions[1];
                            var newY = batchResults.First(r => r.Name == "y_out").AsTensor<long>();
                            int newLen = (int)newY.Dimensions[1];
                            if (newLen <= historyLen) break;

                            var newDense = (DenseTensor<long>)newY;
                            int produced = newLen - historyLen;
                            bool isEOS = false;
                            for (int k = historyLen; k < newLen; k++)
                            {
                                long tok = newDense.Buffer.Span[k];
                                generatedTokens.Add(tok);
                                if (tok >= 1024) isEOS = true;
                            }
                            stepBudget -= produced;

                            currentYTensor = newY;
                            currentYEmbTensor = batchResults.First(r => r.Name == "y_emb_out").AsTensor<float>();
                            for (int layer = 0; layer < numLayers; layer++)
                            {
                                kvCache[layer * 2] = batchResults.First(r => r.Name == PresentKNames[layer] + "_out").AsTensor<float>();
                                kvCache[layer * 2 + 1] = batchResults.First(r => r.Name == PresentVNames[layer] + "_out").AsTensor<float>();
                            }

                            int currentCount = generatedTokens.Count;
                            bool shouldEmit = (currentCount - lastEmitPosition) >= chunkSize;
                            if (shouldEmit || isEOS)
                            {
                                int tokenLen = 0;
                                while (tokenLen < generatedTokens.Count && generatedTokens[tokenLen] < 1024) tokenLen++;
                                if (tokenLen > 0)
                                {
                                    long[] tokens = new long[tokenLen];
                                    for (int k = 0; k < tokenLen; k++) tokens[k] = generatedTokens[k];
                                    if (isEOS) tokens[tokens.Length - 1] = 0;
                                    onTokenChunk(tokens, isEOS);
                                    lastEmitPosition = currentCount;
                                }
                            }

                            if (isEOS) break;
                        }
                    }
                    else
                    {
                    // Reuse the List to avoid per-step allocations
                    var stageInputs = new List<NamedOnnxValue>(2 + KvLayerCount * 2);

                    try
                    {
                        for (int i = 0; i < 500; i++)
                        {
                            stageInputs.Clear();
                            stageInputs.Add(NamedOnnxValue.CreateFromTensor("iy", currentYTensor));
                            stageInputs.Add(NamedOnnxValue.CreateFromTensor("iy_emb", currentYEmbTensor));

                            for (int layer = 0; layer < numLayers; layer++)
                            {
                                stageInputs.Add(NamedOnnxValue.CreateFromTensor(PastKNames[layer], kvCache[layer * 2]));
                                stageInputs.Add(NamedOnnxValue.CreateFromTensor(PastVNames[layer], kvCache[layer * 2 + 1]));
                            }

                            // Run Inference
                            var stageResults = _stageDecoder.Run(stageInputs);

                            // Critical: Dispose the OLD results now that we have NEW results
                            // The old kvCache/yTensor inputs are no longer needed
                            activeResults?.Dispose();
                            activeResults = stageResults; // Update active holder

                            // Update references (No Clone)
                            currentYTensor = stageResults.First(r => r.Name == "y").AsTensor<long>();
                            currentYEmbTensor = stageResults.First(r => r.Name == "y_emb").AsTensor<float>();

                            var stopCondition = stageResults.First(r => r.Name == "stop_condition_tensor").AsTensor<bool>().GetValue(0);

                            // Update KV Cache pointers for next iteration
                            for (int layer = 0; layer < numLayers; layer++)
                            {
                                kvCache[layer * 2] = stageResults.First(r => r.Name == PresentKNames[layer]).AsTensor<float>();
                                kvCache[layer * 2 + 1] = stageResults.First(r => r.Name == PresentVNames[layer]).AsTensor<float>();
                            }

                            // Safe access
                            var denseY = (DenseTensor<long>)currentYTensor;
                            var lastToken = denseY.Buffer.Span[denseY.Buffer.Length - 1];
                            generatedTokens.Add(lastToken);

                            int currentCount = generatedTokens.Count;
                            bool shouldEmit = (currentCount - lastEmitPosition) >= chunkSize;
                            bool isEOS = stopCondition || lastToken >= 1024;

                            if (shouldEmit || isEOS)
                            {
                                // Optimization: avoid TakeWhile().ToArray(), build directly from the List
                                int tokenLen = 0;
                                while (tokenLen < generatedTokens.Count && generatedTokens[tokenLen] < 1024) tokenLen++;

                                if (tokenLen > 0)
                                {
                                    long[] tokens = new long[tokenLen];
                                    for (int k = 0; k < tokenLen; k++) tokens[k] = generatedTokens[k];

                                    if (isEOS) tokens[tokens.Length - 1] = 0;
                                    onTokenChunk(tokens, isEOS);  // Emit tokens only; no vocoder call
                                    lastEmitPosition = currentCount;
                                }
                            }

                            if (isEOS) break;
                        }
                    }
                    finally
                    {
                        activeResults?.Dispose();
                    }
                    } // end else (per-token path)

                        if (generatedTokens.Count >= minExpectedTokens)
                        {
                            success = true;
                            break;
                        }

                        // Insufficient generation: consume the formal retry budget, otherwise the while condition never advances (infinite loop).
                        retryCount++;
                        if (generatedTokens.Count > lastGoodTokenCount)
                        {
                            lastGoodTokenCount = generatedTokens.Count;
                            lastGoodTokens = generatedTokens.ToArray();
                        }
                        if (DebugMode) DebugLog($"[T2S Streaming] Incomplete generation ({generatedTokens.Count} < {minExpectedTokens}), retry {retryCount}/{maxRetries}");
                        onRetry?.Invoke();
                }

                if (!success)
                {
                    // Degrade gracefully: do not throw once retries are exhausted if usable tokens exist (avoids whole-sentence streaming failure).
                    // Typical case is occasional sampling anomalies on long sentences; returning incomplete audio beats a 500 error.
                    if (lastGoodTokenCount > 0)
                    {
                        // Must match the normal path: strip EOS and everything after, and zero the last token.
                        // Otherwise the vocoder receives tokens >=1024 and the quantizer Gather goes out of bounds.
                        int eosIdx = Array.FindIndex(lastGoodTokens, t => t >= 1024);
                        long[] fallback = eosIdx >= 0
                            ? lastGoodTokens.Take(eosIdx).ToArray()
                            : lastGoodTokens;

                        if (fallback.Length > 0)
                        {
                            fallback[^1] = 0;
                            Console.WriteLine(
                                $"⚠️ [T2S Streaming] Retries exhausted, degrading to the generated content " +
                                $"(tokens={fallback.Length}, expected={minExpectedTokens})");
                            onTokenChunk?.Invoke(fallback, true);
                            return;
                        }
                    }
                    throw new Exception($"Streaming T2S inference failed: expected token count ({minExpectedTokens}) not reached after {options.MaxRetries} retries");
                }
            }
            finally
            {
                ReturnFlat(refBertFlat);
                ReturnFlat(textBertFlat);
            }
        }


        public float[] RunVocoder(long[] textSeq, long[] predSemantic, float[] refAudio32k, out int audioLength, float[]? ge = null, float[]? geAdvanced = null, float speed = 1.0f)
        {
            var phaseSw = PerfStats.Enabled ? System.Diagnostics.Stopwatch.StartNew() : null;
            audioLength = 0;
            if (_vocoder == null) throw new InvalidOperationException("Vocoder not loaded.");

            long[] stretchedSemantic = predSemantic;
            // Speed control: interpolate semantic tokens
            if (Math.Abs(speed - 1.0f) > 0.01f)
            {
                DebugLog($"[Vocoder] Applying semantic-token interpolation speed change: {speed}x");
                stretchedSemantic = StretchSemanticTokens(predSemantic, speed);
            }

            try
            {
                var vocoderInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("text_seq", new DenseTensor<long>(textSeq, new[] { 1, textSeq.Length })),
                    NamedOnnxValue.CreateFromTensor("pred_semantic", new DenseTensor<long>(stretchedSemantic, new[] { 1, 1, stretchedSemantic.Length }))
                };

                if (_promptEncoder != null && ge != null && geAdvanced != null)
                {
                    vocoderInputs.Add(NamedOnnxValue.CreateFromTensor("ge", new DenseTensor<float>(ge, new[] { 1, ge.Length, 1 })));
                    vocoderInputs.Add(NamedOnnxValue.CreateFromTensor("ge_advanced", new DenseTensor<float>(geAdvanced, new[] { 1, geAdvanced.Length, 1 })));
                }
                else
                {
                    vocoderInputs.Add(NamedOnnxValue.CreateFromTensor("ref_audio", new DenseTensor<float>(refAudio32k, new[] { 1, refAudio32k.Length })));
                }

                using var results = _vocoder.Run(vocoderInputs);
                var audioTensor = (DenseTensor<float>)results.First().AsTensor<float>();

                audioLength = audioTensor.Buffer.Length;
                // Optimization: copy directly from the tensor Buffer into the rented array, avoiding the large object allocation of ToArray()
                float[] audio = ArrayPool<float>.Shared.Rent(audioLength);
                audioTensor.Buffer.Span.CopyTo(audio);
                return audio;
            }
            finally
            {
                // No need to return stretchedSemantic: it is now a regularly allocated array
                if (phaseSw != null) PerfStats.AddVocoder(phaseSw, audioLength);
            }
        }

        private float[] Flatten(float[,] array)
        {
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);
            int len = rows * cols;
            float[] flat = ArrayPool<float>.Shared.Rent(len);
            Buffer.BlockCopy(array, 0, flat, 0, len * sizeof(float));
            return flat;
        }

        private void ReturnFlat(float[] flat)
        {
            if (flat != null) ArrayPool<float>.Shared.Return(flat);
        }

        /// <summary>
        /// Semantic token interpolation (nearest neighbor)
        /// </summary>
        private long[] StretchSemanticTokens(long[] tokens, float speed)
        {
            if (tokens == null || tokens.Length == 0) return Array.Empty<long>();

            int newLen = (int)Math.Round(tokens.Length / speed);
            if (newLen < 1) newLen = 1;

            // Core fix: do not use ArrayPool.Rent, because Rent may return an array longer than newLen.
            // When passed to DenseTensor, stretchedSemantic.Length includes the extra zero padding,
            // making the vocoder believe there are many more semantic tokens and producing tail hallucination noise ("ah/uh" sounds).
            long[] result = new long[newLen];
            for (int i = 0; i < newLen; i++)
            {
                int oldIdx = (int)Math.Floor(i * speed);
                if (oldIdx >= tokens.Length) oldIdx = tokens.Length - 1;
                result[i] = tokens[oldIdx];
            }
            return result;
        }

        public void ReturnAudioBuffer(float[] buffer)
        {
            if (buffer != null) ArrayPool<float>.Shared.Return(buffer);
        }

        public void Dispose()
        {
            if (IsShared) return;

            _t2sEncoder?.Dispose();
            _firstStageDecoder?.Dispose();
            _stageDecoder?.Dispose();
            _stageDecoderLoop?.Dispose();
            _vocoder?.Dispose();
            _promptEncoder?.Dispose();
            _hubert?.Dispose();
            _svModel?.Dispose();
        }
        /// <summary>
        /// Dynamically compute the minimum expected token count from the phonemes' language attributes.
        /// </summary>
        private int CalculateMinExpectedTokens(long[] textSeq, PhoneLanguage[]? languageTags, TtsOptions options)
        {
            if (languageTags == null || languageTags.Length == 0)
            {
                return Math.Max(8, (int)(textSeq.Length * options.MinTokenMultiplierChinese!.Value));
            }

            double expected = 0;
            for (int i = 0; i < textSeq.Length; i++)
            {
                var lang = i < languageTags.Length ? languageTags[i] : PhoneLanguage.Chinese;
                double multiplier = lang switch
                {
                    PhoneLanguage.Chinese => options.MinTokenMultiplierChinese!.Value,
                    PhoneLanguage.Japanese => options.MinTokenMultiplierJapanese!.Value,
                    PhoneLanguage.English => options.MinTokenMultiplierEnglish!.Value,
                    _ => options.MinTokenMultiplierOther!.Value // Punctuation/Other
                };
                expected += multiplier;
            }

            return Math.Max(8, (int)expected);
        }

        private TtsOptions ApplyConfig(TtsOptions? options)
        {
            options ??= TtsOptions.Default;
            if (_config == null) return options;

            options.DebugMode ??= _config.DebugMode;
            options.MaxRetries ??= _config.Inference.MaxRetries;
            options.MinTokenMultiplierChinese ??= _config.Inference.MinTokenMultiplierChinese;
            options.MinTokenMultiplierJapanese ??= _config.Inference.MinTokenMultiplierJapanese;
            options.MinTokenMultiplierEnglish ??= _config.Inference.MinTokenMultiplierEnglish;
            options.MinTokenMultiplierOther ??= _config.Inference.MinTokenMultiplierOther;

            // Sync the DebugMode instance state so DebugLog works
            this.DebugMode = options.DebugMode.Value;

            return options;
        }
    }
}
