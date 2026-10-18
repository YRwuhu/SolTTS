using System.Buffers;
using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SolTTS.Core.Frontend.G2P.Common;

namespace SolTTS.Core.Core
{
    /// <summary>
    /// New inference engine (based on the GPT-SoVITS_minimal_inference architecture)
    /// Supports:
    /// - New model layout (gpt_encoder + gpt_step + sovits)
    /// - KV-Cache preallocation
    /// - Speed adjustment
    /// - Streaming inference
    /// </summary>
    public class InferenceEngineV2 : IDisposable
    {
        public bool DebugMode { get; set; } = false;

        /// <summary>
        /// Whether shared mode is enabled. In shared mode Dispose does not release sessions.
        /// </summary>
        public bool IsShared { get; set; } = false;

        private void DebugLog(string message)
        {
            if (DebugMode) Console.WriteLine($"[V2] {message}");
        }

        private SolTTS.Core.Config.TTSConfig? _config;

        // ============================================================
        // ONNX Sessions
        // ============================================================
        private InferenceSession? _sslSession;       // ssl.onnx (HuBERT)
        private InferenceSession? _bertSession;      // bert.onnx
        private InferenceSession? _vqEncoderSession; // vq_encoder.onnx
        private InferenceSession? _gptEncoderSession;// gpt_encoder.onnx
        private InferenceSession? _gptStepSession;   // gpt_step.onnx
        private InferenceSession? _sovitsSession;    // sovits.onnx
        private InferenceSession? _svSession;        // sv.onnx (ERes2NetV2, V2ProPlus)

        // ============================================================
        // Model configuration
        // ============================================================
        private Config.V2ModelConfig? _modelConfig;
        private int _maxKvLen = 2000; // Default changed to 2000 to match minimal_inference
        private int[] _kvCacheShape = { 24, 1, 2000, 512 }; // Fetched dynamically from the model
        private string _modelVersion = "v2";
        private bool _isProVersion = false;

        // ============================================================
        // Random sampler
        // ============================================================
        private readonly Random _random = new Random();

        // ============================================================
        // Properties
        // ============================================================
        public int SamplingRate => _modelConfig?.Data?.SamplingRate ?? 32000;
        public string ModelVersion => _modelVersion;
        public bool IsLoaded => _sslSession != null && _sovitsSession != null;

        // ============================================================
        // Session configuration
        // ============================================================
        public SessionOptions CreateSessionOptions(SolTTS.Core.Config.TTSConfig config)
        {
            var options = new SessionOptions();

            bool isMac = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX);

            // Thread configuration (consistent with Win/Linux; explicitly set by the user in config)
            options.IntraOpNumThreads = config.IntraOpNumThreads;
            options.InterOpNumThreads = config.InterOpNumThreads;
            options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;

            // Graph optimization level
            options.GraphOptimizationLevel = config.GraphOptimizationLevel switch
            {
                0 => GraphOptimizationLevel.ORT_DISABLE_ALL,
                1 => GraphOptimizationLevel.ORT_ENABLE_BASIC,
                2 => GraphOptimizationLevel.ORT_ENABLE_ALL,
                _ => GraphOptimizationLevel.ORT_DISABLE_ALL // V2 disables by default for legacy model compatibility
            };

            // Memory optimization
            options.EnableCpuMemArena = true;

            // macOS scheduling latency is high; enabling spinning actually reduces latency
            options.AddSessionConfigEntry("session.intra_op.allow_spinning", isMac ? "1" : "0");

            if (isMac && config.UseCoreML)
            {
                options.EnableMemoryPattern = false;
                try
                {
                    var coremlOptions = new Dictionary<string, string>
                    {
                        ["EnableOnSubgraphs"] = (config.CoreMLFlags & 0x2) != 0 ? "1" : "0",
                        ["MLComputeUnits"] = (config.CoreMLFlags & 0x4) != 0 ? "ANEOnly" : "ALL",
                        ["ModelFormat"] = (config.CoreMLFlags & 0x10) != 0 ? "MLProgram" : "NeuralNetwork",
                        ["RequireStaticInputShapes"] = (config.CoreMLFlags & 0x8) != 0 ? "1" : "0",
                    };
                    options.AppendExecutionProvider("CoreML", coremlOptions);
                    DebugLog($"CoreML acceleration enabled (V2/macOS) flags=0x{config.CoreMLFlags:X}");
                }
                catch (Exception ex)
                {
                    DebugLog($"⚠️ Failed to enable CoreML acceleration, fell back to CPU: {ex.Message}");
                    options.EnableMemoryPattern = true;
                }
            }
            else
            {
                options.EnableMemoryPattern = true;
            }

            return options;
        }

        private static int GetMacPerformanceCoreCount()
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "sysctl",
                    Arguments = "-n hw.perflevel0.physicalcpu",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };
                using var p = Process.Start(psi);
                if (p != null)
                {
                    string output = p.StandardOutput.ReadToEnd().Trim();
                    p.WaitForExit(500);
                    if (int.TryParse(output, out int v) && v > 0) return v;
                }
            }
            catch { }
            return Math.Max(1, Environment.ProcessorCount / 2);
        }

        // ============================================================
        // Model loading
        // ============================================================
        public void LoadModels(string modelDir, SolTTS.Core.Config.TTSConfig config)
        {
            _config = config;
            this.DebugMode = config.DebugMode;
            var options = CreateSessionOptions(config);

            var sw = Stopwatch.StartNew();
            Console.WriteLine($"[V2] Loading model directory: {modelDir}");

            // Load config
            var configPath = Path.Combine(modelDir, "config.json");
            _modelConfig = Config.V2ModelConfig.Load(configPath);
            _maxKvLen = _modelConfig.ExportOptions?.MaxLen ?? 1000;
            _modelVersion = _modelConfig.Model?.Version ?? "v2";
            _isProVersion = _modelVersion.Contains("Pro", StringComparison.OrdinalIgnoreCase);

            Console.WriteLine($"[V2] Model version: {_modelVersion}, KV-Cache max length: {_maxKvLen}");

            // Load all models in parallel
            var tasks = new List<Action>
            {
                () => _sslSession = new InferenceSession(Path.Combine(modelDir, "ssl.onnx"), options),
                () => _bertSession = new InferenceSession(Path.Combine(modelDir, "bert.onnx"), options),
                () => _vqEncoderSession = new InferenceSession(Path.Combine(modelDir, "vq_encoder.onnx"), options),
                () => _gptEncoderSession = new InferenceSession(Path.Combine(modelDir, "gpt_encoder.onnx"), options),
                () => _gptStepSession = new InferenceSession(Path.Combine(modelDir, "gpt_step.onnx"), options),
                () => _sovitsSession = new InferenceSession(Path.Combine(modelDir, "sovits.onnx"), options),
            };

            Parallel.Invoke(tasks.ToArray());

            // Load SV model (required by V2ProPlus)
            if (_isProVersion)
            {
                var svPath = Path.Combine(modelDir, "sv.onnx");
                if (File.Exists(svPath))
                {
                    _svSession = new InferenceSession(svPath, options);
                    Console.WriteLine("[V2] SV model (ERes2NetV2) loaded");
                }
                else
                {
                    Console.WriteLine("[V2] ⚠️ sv.onnx not found; V2ProPlus voice-cloning features may be limited");
                }
            }

            // Read the actual KV Cache shape from the gpt_step model
            try
            {
                var kCacheInput = _gptStepSession!.InputMetadata["k_cache"];
                var dims = kCacheInput.Dimensions;
                if (dims.Length == 4)
                {
                    // Handle dynamic dimensions: ONNX uses -1 for dynamic dimensions; substitute actual values
                    int layers = dims[0] > 0 ? dims[0] : 24;      // Layer count
                    int batchSize = dims[1] > 0 ? dims[1] : 1;    // Batch size, normally 1
                    int maxLen = dims[2] > 0 ? dims[2] : 2000;    // Maximum sequence length
                    int headDim = dims[3] > 0 ? dims[3] : 512;    // Attention head dimension

                    _kvCacheShape = new[] { layers, batchSize, maxLen, headDim };
                    _maxKvLen = maxLen;
                    Console.WriteLine($"[V2] KV Cache shape: [{layers}, {batchSize}, {maxLen}, {headDim}] (raw: [{dims[0]}, {dims[1]}, {dims[2]}, {dims[3]}])");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[V2] ⚠️ Failed to read KV Cache shape from the model, using defaults: {ex.Message}");
            }

            sw.Stop();
            Console.WriteLine($"[V2] Model loading complete, elapsed: {sw.ElapsedMilliseconds}ms");
        }

        /// <summary>
        /// Share model sessions from another instance
        /// </summary>
        public void ShareModelsFrom(InferenceEngineV2 other)
        {
            _sslSession = other._sslSession;
            _bertSession = other._bertSession;
            _vqEncoderSession = other._vqEncoderSession;
            _gptEncoderSession = other._gptEncoderSession;
            _gptStepSession = other._gptStepSession;
            _sovitsSession = other._sovitsSession;
            _svSession = other._svSession;
            _modelConfig = other._modelConfig;
            _maxKvLen = other._maxKvLen;
            _kvCacheShape = other._kvCacheShape;
            _modelVersion = other._modelVersion;
            _isProVersion = other._isProVersion;
            _config = other._config;
            IsShared = true;
        }

        // ============================================================
        // SSL feature extraction (HuBERT)
        // ============================================================
        public float[] ExtractSSL(float[] audio16k)
        {
            if (_sslSession == null) throw new InvalidOperationException("SSL model not loaded");

            var inputTensor = new DenseTensor<float>(audio16k, new[] { 1, audio16k.Length });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("audio", inputTensor)
            };

            using var results = _sslSession.Run(inputs);
            var output = results.First().AsTensor<float>();

            // Output shape: [1, 768, T]
            DebugLog($"SSL output shape: [1, {output.Dimensions[1]}, {output.Dimensions[2]}]");

            return output.ToArray();
        }

        // ============================================================
        // VQ encoding
        // ============================================================
        public long[] EncodeVQ(float[] sslContent)
        {
            if (_vqEncoderSession == null) throw new InvalidOperationException("VQ Encoder not loaded");

            int T = sslContent.Length / 768;
            var inputTensor = new DenseTensor<float>(sslContent, new[] { 1, 768, T });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("ssl_content", inputTensor)
            };

            using var results = _vqEncoderSession.Run(inputs);
            var codes = results.First().AsTensor<long>();

            DebugLog($"VQ encoding output: {codes.Dimensions[2]} tokens");

            // Return the codes shaped [T]
            return codes.ToArray();
        }

        // ============================================================
        // BERT feature extraction
        // ============================================================
        public float[] ExtractBERT(long[] inputIds, long[] attentionMask, long[] tokenTypeIds)
        {
            if (_bertSession == null) throw new InvalidOperationException("BERT model not loaded");

            int seqLen = inputIds.Length;
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(inputIds, new[] { 1, seqLen })),
                NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(attentionMask, new[] { 1, seqLen })),
                NamedOnnxValue.CreateFromTensor("token_type_ids", new DenseTensor<long>(tokenTypeIds, new[] { 1, seqLen }))
            };

            using var results = _bertSession.Run(inputs);
            var hiddenStates = results.First().AsTensor<float>();

            DebugLog($"BERT output shape: [{hiddenStates.Dimensions[0]}, {hiddenStates.Dimensions[1]}, {hiddenStates.Dimensions[2]}]");

            return hiddenStates.ToArray();
        }

        // ============================================================
        // Compute FBank features (Kaldi-style 80-bin)
        // ============================================================
        /// <summary>
        /// Compute Kaldi-style FBank features for the SV model.
        /// </summary>
        /// <param name="audio16k">16kHz mono audio</param>
        /// <returns>FBank features [T, 80]</returns>
        public float[,] ComputeFBank(float[] audio16k)
        {
            // FBank parameters (Kaldi defaults)
            const int sampleRate = 16000;
            const int numMelBins = 80;
            const float frameLengthMs = 25.0f;
            const float frameShiftMs = 10.0f;
            const int nFft = 512;

            int frameLength = (int)(sampleRate * frameLengthMs / 1000);  // 400
            int frameShift = (int)(sampleRate * frameShiftMs / 1000);    // 160
            int numFrames = (audio16k.Length - frameLength) / frameShift + 1;

            if (numFrames <= 0)
            {
                return new float[1, numMelBins];  // Audio too short
            }

            var fbank = new float[numFrames, numMelBins];

            // Simplified implementation: basic FFT and Mel filters.
            // A production implementation should match the Python/Kaldi reference.
            var melFilters = CreateMelFilterBank(nFft / 2 + 1, numMelBins, sampleRate, 20, sampleRate / 2);
            var window = CreateHannWindow(frameLength);

            for (int frame = 0; frame < numFrames; frame++)
            {
                int offset = frame * frameShift;

                // Windowing
                var windowed = new double[nFft];
                for (int i = 0; i < frameLength && offset + i < audio16k.Length; i++)
                {
                    windowed[i] = audio16k[offset + i] * window[i];
                }

                // FFT (simplified - MathNet or another library performs better)
                var spectrum = ComputePowerSpectrum(windowed);

                // Mel filter bank
                for (int mel = 0; mel < numMelBins; mel++)
                {
                    double sum = 0;
                    for (int k = 0; k < spectrum.Length; k++)
                    {
                        sum += spectrum[k] * melFilters[mel, k];
                    }
                    fbank[frame, mel] = (float)Math.Log(Math.Max(sum, 1e-10));
                }
            }

            DebugLog($"FBank features: [{numFrames}, {numMelBins}]");
            return fbank;
        }

        // Create a Hann window
        private static double[] CreateHannWindow(int length)
        {
            var window = new double[length];
            for (int i = 0; i < length; i++)
            {
                window[i] = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (length - 1)));
            }
            return window;
        }

        // Create a Mel filter bank
        private static double[,] CreateMelFilterBank(int numFftBins, int numMelBins, int sampleRate, double lowFreq, double highFreq)
        {
            var filters = new double[numMelBins, numFftBins];

            // Hz to Mel
            Func<double, double> hzToMel = hz => 1127.0 * Math.Log(1 + hz / 700.0);
            Func<double, double> melToHz = mel => 700.0 * (Math.Exp(mel / 1127.0) - 1);

            double melLow = hzToMel(lowFreq);
            double melHigh = hzToMel(highFreq);
            double melStep = (melHigh - melLow) / (numMelBins + 1);

            var melPoints = new double[numMelBins + 2];
            for (int i = 0; i < melPoints.Length; i++)
            {
                melPoints[i] = melLow + i * melStep;
            }

            var hzPoints = melPoints.Select(m => melToHz(m)).ToArray();
            var binPoints = hzPoints.Select(hz => (int)Math.Floor((numFftBins - 1) * 2 * hz / sampleRate)).ToArray();

            for (int m = 0; m < numMelBins; m++)
            {
                int left = binPoints[m];
                int center = binPoints[m + 1];
                int right = binPoints[m + 2];

                for (int k = left; k <= center && k < numFftBins; k++)
                {
                    if (center != left)
                        filters[m, k] = (double)(k - left) / (center - left);
                }
                for (int k = center; k <= right && k < numFftBins; k++)
                {
                    if (right != center)
                        filters[m, k] = (double)(right - k) / (right - center);
                }
            }

            return filters;
        }

        // Compute the power spectrum (simplified FFT)
        private static double[] ComputePowerSpectrum(double[] signal)
        {
            int n = signal.Length;
            int halfN = n / 2 + 1;
            var spectrum = new double[halfN];

            // Simplified DFT (production code should use an FFT library)
            for (int k = 0; k < halfN; k++)
            {
                double realSum = 0, imagSum = 0;
                for (int t = 0; t < n; t++)
                {
                    double angle = -2 * Math.PI * k * t / n;
                    realSum += signal[t] * Math.Cos(angle);
                    imagSum += signal[t] * Math.Sin(angle);
                }
                spectrum[k] = (realSum * realSum + imagSum * imagSum) / n;
            }

            return spectrum;
        }

        // ============================================================
        // Speaker embedding extraction (SV / ERes2NetV2)
        // ============================================================
        /// <summary>
        /// Extract speaker embedding (20480-dim) from 16kHz audio.
        /// Only applicable to V2ProPlus models.
        /// </summary>
        public float[]? ExtractSpeakerEmbedding(float[] audio16k)
        {
            if (_svSession == null)
            {
                DebugLog("SV model not loaded; cannot extract speaker embedding");
                return null;
            }

            // 1. Compute FBank features
            var fbank = ComputeFBank(audio16k);
            int numFrames = fbank.GetLength(0);
            int numMelBins = fbank.GetLength(1);

            // 2. Run the SV model
            var fbankTensor = new DenseTensor<float>(Flatten(fbank), new[] { 1, numFrames, numMelBins });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("fbank_feat", fbankTensor)
            };

            using var results = _svSession.Run(inputs);
            var svEmb = results.First().AsTensor<float>().ToArray();

            DebugLog($"Speaker embedding extraction complete: {svEmb.Length} dims");

            // 3. Handle dimensions (ensure 20480 dims)
            const int expectedDim = 20480;
            if (svEmb.Length != expectedDim)
            {
                var padded = new float[expectedDim];
                Array.Copy(svEmb, padded, Math.Min(svEmb.Length, expectedDim));
                return padded;
            }

            return svEmb;
        }

        // ============================================================
        // Top-K sampling
        // ============================================================
        private long SampleTopK(float[] topkValues, long[] topkIndices, float temperature = 1.0f)
        {
            int k = topkValues.Length;

            // Apply temperature
            if (Math.Abs(temperature - 1.0f) > 0.001f)
            {
                for (int i = 0; i < k; i++)
                {
                    topkValues[i] /= temperature;
                }
            }

            // Softmax
            float maxVal = topkValues.Max();
            float sum = 0;
            for (int i = 0; i < k; i++)
            {
                topkValues[i] = MathF.Exp(topkValues[i] - maxVal);
                sum += topkValues[i];
            }
            for (int i = 0; i < k; i++)
            {
                topkValues[i] /= sum;
            }

            // Sample
            float r = (float)_random.NextDouble();
            float cumulative = 0;
            for (int i = 0; i < k; i++)
            {
                cumulative += topkValues[i];
                if (r <= cumulative)
                {
                    return topkIndices[i];
                }
            }

            return topkIndices[k - 1];
        }

        // ============================================================
        // Streaming inference (generator pattern)
        // ============================================================
        /// <summary>
        /// Streaming inference: generate audio chunks.
        /// </summary>
        /// <param name="phonemeIds">Phoneme ID sequence</param>
        /// <param name="bertFeature">BERT features [1024, T]</param>
        /// <param name="promptCodes">VQ codes of the reference audio</param>
        /// <param name="referSpec">Spectrogram of the reference audio</param>
        /// <param name="svEmb">Speaker embedding (required by V2ProPlus)</param>
        /// <param name="topK">Top-K sampling</param>
        /// <param name="temperature">Sampling temperature</param>
        /// <param name="noiseScale">Noise scale</param>
        /// <param name="speed">Speech rate (0.5-2.0)</param>
        /// <param name="chunkSize">Number of tokens between outputs</param>
        /// <param name="onAudioChunk">Audio chunk callback</param>
        public async IAsyncEnumerable<float[]> InferStream(
            long[] refPhonemeIds, float[,] refBertFeature,
            long[] phonemeIds, float[,] bertFeature,
            long[] promptCodes, float[,] referSpec, float[]? svEmb,
            PhoneLanguage[]? languageTags = null, TtsOptions? options = null,
            int topK = 15, float temperature = 1.0f, float noiseScale = 0.35f,
            float speed = 1.0f, int chunkSize = 24)
        {
            options = ApplyConfig(options);
            if (_gptEncoderSession == null || _gptStepSession == null || _sovitsSession == null)
                throw new InvalidOperationException("Models not loaded");

            var sw = Stopwatch.StartNew();

            // Concatenate reference and target token IDs (for SoVITS)
            var generatedTokens = new List<long>();

            // ========== 1. GPT Encoder ==========
            DebugLog("Running GPT Encoder...");

            // Concatenate reference and target phonemes (Python: phones1 + phones2)
            var allPhonemeIds = refPhonemeIds.Concat(phonemeIds).ToArray();

            // Concatenate reference and target BERT features (Python: concatenate([bert1, bert2], axis=1))
            int bertDim = bertFeature.GetLength(0);  // 1024
            int refBertLen = refBertFeature.GetLength(1);
            int targetBertLen = bertFeature.GetLength(1);
            int totalBertLen = refBertLen + targetBertLen;

            var allBertFeature = new float[bertDim, totalBertLen];
            // Copy reference BERT
            for (int d = 0; d < bertDim; d++)
                for (int t = 0; t < refBertLen; t++)
                    allBertFeature[d, t] = refBertFeature[d, t];
            // Copy target BERT
            for (int d = 0; d < bertDim; d++)
                for (int t = 0; t < targetBertLen; t++)
                    allBertFeature[d, refBertLen + t] = bertFeature[d, t];

            DebugLog($"Phoneme concat: ref={refPhonemeIds.Length} + target={phonemeIds.Length} = {allPhonemeIds.Length}");
            DebugLog($"BERT concat: ref={refBertLen} + target={targetBertLen} = {totalBertLen}");

            int textLen = allPhonemeIds.Length;
            int promptLen = promptCodes.Length;

            var encoderInputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("phoneme_ids", new DenseTensor<long>(allPhonemeIds, new[] { 1, textLen })),
                NamedOnnxValue.CreateFromTensor("prompts", new DenseTensor<long>(promptCodes, new[] { 1, promptLen })),
                NamedOnnxValue.CreateFromTensor("bert_feature", new DenseTensor<float>(Flatten(allBertFeature), new[] { 1, bertDim, totalBertLen }))
            };

            float[] topkValues, kCache, vCache;
            long[] topkIndices, xLen, yLen;

            using (var encoderResults = _gptEncoderSession.Run(encoderInputs))
            {
                topkValues = encoderResults.First(r => r.Name == "topk_values").AsTensor<float>().ToArray();
                topkIndices = encoderResults.First(r => r.Name == "topk_indices").AsTensor<long>().ToArray();
                kCache = encoderResults.First(r => r.Name == "k_cache").AsTensor<float>().ToArray();
                vCache = encoderResults.First(r => r.Name == "v_cache").AsTensor<float>().ToArray();
                xLen = encoderResults.First(r => r.Name == "x_len").AsTensor<long>().ToArray();
                yLen = encoderResults.First(r => r.Name == "y_len").AsTensor<long>().ToArray();
            }

            DebugLog($"GPT Encoder complete, elapsed: {sw.ElapsedMilliseconds}ms");

            // ========== 2. GPT Step Loop with Retry ==========
            int minExpectedTokens = CalculateMinExpectedTokens(phonemeIds, languageTags, options);
            int maxRetries = options.MaxRetries!.Value;
            int retryCount = 0;
            bool success = false;

            while (retryCount < maxRetries)
            {
                retryCount++;
                generatedTokens.Clear();
                GPT_KV_Cache? gptKvCache = null;
                try
                {
                    if (DebugMode) DebugLog($"[V2 Stream] Retry {retryCount}/{maxRetries}, clearing previous audio...");

                    long currentToken = SampleTopK(topkValues, topkIndices, temperature);
                    generatedTokens.Add(currentToken);

                    int lastEmitPosition = 0;
                    int lastAudioLength = 0;
                    const int maxSteps = 1500;
                    const long EOS_TOKEN = 1024;

                    gptKvCache = new GPT_KV_Cache(kCache, vCache, GetKvCacheShape());

                    for (int step = 0; step < maxSteps; step++)
                    {
                        var stepInputs = new List<NamedOnnxValue>
                        {
                            NamedOnnxValue.CreateFromTensor("samples", new DenseTensor<long>(new[] { currentToken }, new[] { 1, 1 })),
                            NamedOnnxValue.CreateFromTensor("k_cache", gptKvCache.KCacheTensor),
                            NamedOnnxValue.CreateFromTensor("v_cache", gptKvCache.VCacheTensor),
                            NamedOnnxValue.CreateFromTensor("x_len", new DenseTensor<long>(xLen, new[] { 1 })),
                            NamedOnnxValue.CreateFromTensor("y_len", new DenseTensor<long>(yLen, new[] { 1 })),
                            NamedOnnxValue.CreateFromTensor("idx", new DenseTensor<long>(new[] { (long)step }, new[] { 1 }))
                        };

                        using var stepResults = _gptStepSession.Run(stepInputs);

                        var topkValuesNext = stepResults.First(r => r.Name == "topk_values").AsTensor<float>().ToArray();
                        var topkIndicesNext = stepResults.First(r => r.Name == "topk_indices").AsTensor<long>().ToArray();

                        var newKCache = stepResults.First(r => r.Name == "k_cache_new").AsTensor<float>();
                        var newVCache = stepResults.First(r => r.Name == "v_cache_new").AsTensor<float>();
                        gptKvCache.Update(newKCache, newVCache);

                        currentToken = SampleTopK(topkValuesNext, topkIndicesNext, temperature);
                        generatedTokens.Add(currentToken);

                        // Check whether an audio chunk should be emitted
                        bool isEOS = currentToken >= EOS_TOKEN;
                        bool shouldEmit = (generatedTokens.Count - lastEmitPosition) >= chunkSize;

                        if ((shouldEmit || isEOS))
                        {
                            var tokensForVocoder = generatedTokens.Where(t => t < EOS_TOKEN).ToArray();
                            if (tokensForVocoder.Length > 0)
                            {
                                var fullAudio = RunSoVITS(tokensForVocoder, phonemeIds, referSpec, svEmb, noiseScale, speed);
                                int newSamplesCount = fullAudio.Length - lastAudioLength;
                                if (newSamplesCount > 0)
                                {
                                    float[] chunk = new float[newSamplesCount];
                                    Array.Copy(fullAudio, lastAudioLength, chunk, 0, newSamplesCount);
                                    await foreach (var item in AsyncEnumerable.Repeat(chunk, 1))
                                    {
                                        yield return item;
                                    }
                                    lastAudioLength = fullAudio.Length;
                                }
                                else if (isEOS && (generatedTokens.Count >= minExpectedTokens))
                                {
                                    await foreach (var item in AsyncEnumerable.Repeat(Array.Empty<float>(), 1))
                                    {
                                        yield return item;
                                    }
                                }

                                lastEmitPosition = generatedTokens.Count;
                            }
                        }

                        if (isEOS) break;
                    }

                    if (generatedTokens.Count >= minExpectedTokens)
                    {
                        success = true;
                        break;
                    }
                    if (DebugMode) DebugLog($"[InferStream] Insufficient tokens ({generatedTokens.Count} < {minExpectedTokens}), retrying {retryCount}/{maxRetries}...");
                }
                finally
                {
                    gptKvCache?.Dispose();
                }
            }

            if (!success)
            {
                throw new Exception($"Streaming inference failed: no valid semantic sequence after {options.MaxRetries} retries.");
            }

            sw.Stop();
            DebugLog($"Inference complete, total elapsed: {sw.ElapsedMilliseconds}ms, generated {generatedTokens.Count} tokens");
        }

        /// <summary>
        /// Non-streaming inference: return all semantic tokens at once.
        /// </summary>
        /// <param name="refPhonemeIds">Reference text phoneme IDs</param>
        /// <param name="refBertFeature">Reference text BERT features [1024, T_ref]</param>
        /// <param name="phonemeIds">Target text phoneme IDs</param>
        /// <param name="bertFeature">Target text BERT features [1024, T_target]</param>
        /// <param name="promptCodes">VQ codes of the reference audio</param>
        /// <param name="topK">Top-K sampling</param>
        /// <param name="temperature">Sampling temperature</param>
        public long[] InferTokens(
            long[] refPhonemeIds, float[,] refBertFeature,
            long[] phonemeIds, float[,] bertFeature,
            long[] promptCodes,
            PhoneLanguage[]? languageTags = null, TtsOptions? options = null,
            int topK = 15, float temperature = 1.0f)
        {
            options = ApplyConfig(options);
            if (_gptEncoderSession == null || _gptStepSession == null)
                throw new InvalidOperationException("Models not loaded");

            // Concatenate reference and target phonemes
            var allPhonemeIds = refPhonemeIds.Concat(phonemeIds).ToArray();

            // Concatenate reference and target BERT features
            int bertDim = bertFeature.GetLength(0);  // 1024
            int refBertLen = refBertFeature.GetLength(1);
            int targetBertLen = bertFeature.GetLength(1);
            int totalBertLen = refBertLen + targetBertLen;

            var allBertFeature = new float[bertDim, totalBertLen];
            for (int d = 0; d < bertDim; d++)
                for (int t = 0; t < refBertLen; t++)
                    allBertFeature[d, t] = refBertFeature[d, t];
            for (int d = 0; d < bertDim; d++)
                for (int t = 0; t < targetBertLen; t++)
                    allBertFeature[d, refBertLen + t] = bertFeature[d, t];

            int textLen = allPhonemeIds.Length;
            int promptLen = promptCodes.Length;

            var encoderInputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("phoneme_ids", new DenseTensor<long>(allPhonemeIds, new[] { 1, textLen })),
                NamedOnnxValue.CreateFromTensor("prompts", new DenseTensor<long>(promptCodes, new[] { 1, promptLen })),
                NamedOnnxValue.CreateFromTensor("bert_feature", new DenseTensor<float>(Flatten(allBertFeature), new[] { 1, bertDim, totalBertLen }))
            };

            float[] topkValues, kCache, vCache;
            long[] topkIndices, xLen, yLen;

            using (var encoderResults = _gptEncoderSession.Run(encoderInputs))
            {
                topkValues = encoderResults.First(r => r.Name == "topk_values").AsTensor<float>().ToArray();
                topkIndices = encoderResults.First(r => r.Name == "topk_indices").AsTensor<long>().ToArray();
                kCache = encoderResults.First(r => r.Name == "k_cache").AsTensor<float>().ToArray();
                vCache = encoderResults.First(r => r.Name == "v_cache").AsTensor<float>().ToArray();
                xLen = encoderResults.First(r => r.Name == "x_len").AsTensor<long>().ToArray();
                yLen = encoderResults.First(r => r.Name == "y_len").AsTensor<long>().ToArray();
            }

            int minExpectedTokens = CalculateMinExpectedTokens(phonemeIds, languageTags, options);
            int maxRetries = options.MaxRetries!.Value;
            int retryCount = 0;
            bool success = false;
            long[]? finalResult = null;

            while (retryCount < maxRetries)
            {
                retryCount++;
                var generatedTokens = new List<long>();
                long currentToken = SampleTopK(topkValues.ToArray(), topkIndices.ToArray(), temperature);
                generatedTokens.Add(currentToken);

                using var gptKvCache = new GPT_KV_Cache(kCache, vCache, GetKvCacheShape());

                const int maxSteps = 1500;
                const long EOS_TOKEN = 1024;

                for (int step = 0; step < maxSteps; step++)
                {
                    var stepInputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("samples", new DenseTensor<long>(new[] { currentToken }, new[] { 1, 1 })),
                        NamedOnnxValue.CreateFromTensor("k_cache", gptKvCache.KCacheTensor),
                        NamedOnnxValue.CreateFromTensor("v_cache", gptKvCache.VCacheTensor),
                        NamedOnnxValue.CreateFromTensor("x_len", new DenseTensor<long>(xLen, new[] { 1 })),
                        NamedOnnxValue.CreateFromTensor("y_len", new DenseTensor<long>(yLen, new[] { 1 })),
                        NamedOnnxValue.CreateFromTensor("idx", new DenseTensor<long>(new[] { (long)step }, new[] { 1 }))
                    };

                    using var stepResults = _gptStepSession.Run(stepInputs);

                    var topkValuesNext = stepResults.First(r => r.Name == "topk_values").AsTensor<float>().ToArray();
                    var topkIndicesNext = stepResults.First(r => r.Name == "topk_indices").AsTensor<long>().ToArray();

                    var newKCache = stepResults.First(r => r.Name == "k_cache_new").AsTensor<float>();
                    var newVCache = stepResults.First(r => r.Name == "v_cache_new").AsTensor<float>();
                    gptKvCache.Update(newKCache, newVCache);

                    currentToken = SampleTopK(topkValuesNext, topkIndicesNext, temperature);
                    if (currentToken >= EOS_TOKEN) break;
                    generatedTokens.Add(currentToken);
                }

                if (generatedTokens.Count >= minExpectedTokens)
                {
                    success = true;
                    finalResult = generatedTokens.ToArray();
                    break;
                }
                if (DebugMode) DebugLog($"[InferTokens] Insufficient tokens ({generatedTokens.Count} < {minExpectedTokens}), retrying {retryCount}/{maxRetries}...");
            }

            if (!success || finalResult == null)
            {
                throw new Exception($"Inference failed: no valid semantic sequence after {options.MaxRetries} retries.");
            }

            // Set the last token to 0
            if (finalResult.Length > 0) finalResult[finalResult.Length - 1] = 0;
            return finalResult;
        }

        // ============================================================
        // SoVITS audio synthesis
        // ============================================================
        public float[] RunSoVITS(long[] predSemantic, long[] textSeq, float[,] referSpec, float[]? svEmb, float noiseScale = 0.35f, float speed = 1.0f)
        {
            if (_sovitsSession == null) throw new InvalidOperationException("SoVITS not loaded");

            if (Math.Abs(speed - 1.0f) > 0.01f)
            {
                DebugLog($"Applying semantic-token interpolation speed change: {speed}x");
                predSemantic = StretchSemanticTokens(predSemantic, speed);
            }

            int specDim = referSpec.GetLength(0);
            int specLen = referSpec.GetLength(1);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("pred_semantic", new DenseTensor<long>(predSemantic, new[] { 1, 1, predSemantic.Length })),
                NamedOnnxValue.CreateFromTensor("text_seq", new DenseTensor<long>(textSeq, new[] { 1, textSeq.Length })),
                NamedOnnxValue.CreateFromTensor("refer_spec", new DenseTensor<float>(Flatten(referSpec), new[] { 1, specDim, specLen })),
                NamedOnnxValue.CreateFromTensor("noise_scale", new DenseTensor<float>(new[] { noiseScale }, new[] { 1 }))
            };

            if (_sovitsSession.InputMetadata.ContainsKey("speed"))
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor("speed", new DenseTensor<float>(new[] { 1.0f }, new[] { 1 })));
            }

            if (_sovitsSession.InputMetadata.ContainsKey("sv_emb"))
            {
                const int svEmbDim = 20480;
                var emb = (svEmb != null && svEmb.Length == svEmbDim) ? svEmb : new float[svEmbDim];
                inputs.Add(NamedOnnxValue.CreateFromTensor("sv_emb", new DenseTensor<float>(emb, new[] { 1, svEmbDim })));
            }

            using var results = _sovitsSession.Run(inputs);
            return results.First().AsTensor<float>().ToArray();
        }

        // ============================================================
        // Helpers and classes
        // ============================================================
        private int[] GetKvCacheShape() => _kvCacheShape;

        private static float[] Flatten(float[,] array)
        {
            int rows = array.GetLength(0), cols = array.GetLength(1);
            float[] flat = new float[rows * cols];
            Buffer.BlockCopy(array, 0, flat, 0, flat.Length * sizeof(float));
            return flat;
        }

        private long[] StretchSemanticTokens(long[] tokens, float speed)
        {
            if (tokens == null || tokens.Length == 0) return tokens!;
            int newLen = Math.Max(1, (int)Math.Round(tokens.Length / speed));
            long[] result = new long[newLen];
            for (int i = 0; i < newLen; i++)
            {
                int oldIdx = (int)Math.Floor(i * speed);
                result[i] = tokens[Math.Min(oldIdx, tokens.Length - 1)];
            }
            return result;
        }

        /// <summary>
        /// Dynamically compute the minimum expected token count from the phonemes' language attributes.
        /// </summary>
        private int CalculateMinExpectedTokens(long[] phonemeIds, PhoneLanguage[]? languageTags, TtsOptions options)
        {
            if (languageTags == null || languageTags.Length == 0)
                return Math.Max(8, (int)(phonemeIds.Length * (double)options.MinTokenMultiplierChinese!.Value));

            double expected = 0;
            for (int i = 0; i < phonemeIds.Length; i++)
            {
                var lang = i < languageTags.Length ? languageTags[i] : PhoneLanguage.Chinese;
                expected += lang switch
                {
                    PhoneLanguage.Chinese => (double)options.MinTokenMultiplierChinese!.Value,
                    PhoneLanguage.Japanese => (double)options.MinTokenMultiplierJapanese!.Value,
                    PhoneLanguage.English => (double)options.MinTokenMultiplierEnglish!.Value,
                    _ => (double)options.MinTokenMultiplierOther!.Value
                };
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

        private class GPT_KV_Cache : IDisposable
        {
            public DenseTensor<float> KCacheTensor { get; private set; }
            public DenseTensor<float> VCacheTensor { get; private set; }

            public GPT_KV_Cache(float[] kData, float[] vData, int[] shape)
            {
                KCacheTensor = new DenseTensor<float>(kData.ToArray(), shape);
                VCacheTensor = new DenseTensor<float>(vData.ToArray(), shape);
            }

            public void Update(Tensor<float> newK, Tensor<float> newV)
            {
                KCacheTensor = newK.Clone() as DenseTensor<float> ?? throw new Exception("KV Cache clone failed");
                VCacheTensor = newV.Clone() as DenseTensor<float> ?? throw new Exception("KV Cache clone failed");
            }

            public void Dispose() { }
        }

        public void Dispose()
        {
            if (IsShared) return;

            _sslSession?.Dispose();
            _bertSession?.Dispose();
            _vqEncoderSession?.Dispose();
            _gptEncoderSession?.Dispose();
            _gptStepSession?.Dispose();
            _sovitsSession?.Dispose();
            _svSession?.Dispose();
        }
    }
}
