using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SolTTS.Core.Core;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SolTTS.Core.Frontend.BERT
{
    public class RobertaFeatureExtractor : IDisposable
    {
        private readonly Tokenizers.HuggingFace.Tokenizer.Tokenizer _tokenizer;
        private readonly InferenceSession _session;
        private readonly BertFeatureCache? _cache;
        public bool DebugMode { get; set; } = false;

        /// <summary>
        /// Whether to enable feature caching. When enabled, repeated calls with the same text + alignment return the cached result directly,
        /// skipping one BERT forward pass (measured at about 280–390 ms).
        /// The cached array is shared read-only: callers must not modify the returned float[,].
        /// </summary>
        public bool UseCache { get; set; } = true;

        public RobertaFeatureExtractor(string modelPath, string tokenizerJsonPath, SessionOptions? options = null, bool useCache = true)
        {
            if (!File.Exists(modelPath)) throw new FileNotFoundException(modelPath);
            if (!File.Exists(tokenizerJsonPath)) throw new FileNotFoundException(tokenizerJsonPath);

            _tokenizer = Tokenizers.HuggingFace.Tokenizer.Tokenizer.FromFile(tokenizerJsonPath);

            options ??= new SessionOptions();
            _session = new InferenceSession(modelPath, options);

            UseCache = useCache;
            _cache = useCache ? new BertFeatureCache() : null;
        }

        /// <summary>
        /// Experimental-only constructor: falls back to CPU when no usable ggml session options are available.
        /// The return value indicates whether the actual backend is GPU.
        /// </summary>
        public static RobertaFeatureExtractor CreateWithGpuFallback(
            string modelPath, string tokenizerJsonPath, SessionOptions cpuOptions, bool useCache,
            Func<SessionOptions?> gpuOptionsFactory)
        {
            var gpuOptions = gpuOptionsFactory();
            if (gpuOptions != null)
            {
                try
                {
                    var extractor = new RobertaFeatureExtractor(modelPath, tokenizerJsonPath, gpuOptions, useCache)
                    {
                        IsUsingGgml = true
                    };
                    // Note: for unsupported graphs the plugin reports "claiming nothing" and silently falls back to CPU,
                    // so successful construction does not mean execution actually happens on GPU; check the plugin logs for the real backend.
                    Console.WriteLine("[BERT] Session created with the ggml EP (falls back to CPU execution if the plugin rejects the whole graph)");
                    return extractor;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"⚠️ [BERT] ggml backend failed to load; falling back to CPU: {ex.Message}");
                }
                finally
                {
                    gpuOptions.Dispose();
                }
            }

            return new RobertaFeatureExtractor(modelPath, tokenizerJsonPath, cpuOptions, useCache);
        }

        /// <summary>Whether this extractor currently runs on the ggml/Vulkan backend (diagnostics only).</summary>
        public bool IsUsingGgml { get; private set; }

        public float[,] Extract(string text, int[] word2ph)
        {
            // 0. Cache lookup (key includes text and alignment; a hit skips tokenization and inference)
            string? cacheKey = null;
            if (UseCache && _cache != null)
            {
                cacheKey = BertFeatureCache.MakeKey(text, word2ph);
                if (_cache.TryGet(cacheKey, out var cachedFeat))
                {
                    if (DebugMode)
                        Console.WriteLine($"[BERT Cache] Hit: '{text}' (rows={cachedFeat.GetLength(0)}), cumulative hits={_cache.Hits} misses={_cache.Misses}");
                    return cachedFeat;
                }
            }

            // 1. Tokenize - get token IDs and word mapping
            var encodings = _tokenizer.Encode(text, addSpecialTokens: true);
            var encoding = encodings.First();
            var ids = encoding.Ids.Select(x => (long)x).ToArray();
            var mask = ids.Select(_ => 1L).ToArray();

            // 2. Build the repeats array (maps words to tokens)
            // Rule: word2ph is the phoneme count for each word. If a word is split into multiple tokens, assign the phoneme count to the first token and 0 to the rest.
            long[] fullRepeats = new long[ids.Length];
            int lastWordIdx = -1;

            if (encoding.Words != null && encoding.Words.Count == ids.Length)
            {
                for (int i = 0; i < ids.Length; i++)
                {
                    uint wIdx = encoding.Words[i];
                    int wordIdx = (wIdx == uint.MaxValue) ? -1 : (int)wIdx;

                    if (wordIdx >= 0 && wordIdx < word2ph.Length)
                    {
                        if (wordIdx != lastWordIdx)
                        {
                            fullRepeats[i] = word2ph[wordIdx];
                            lastWordIdx = wordIdx;
                        }
                        else
                        {
                            fullRepeats[i] = 0; // Subword tokens are set to 0
                        }
                    }
                    else
                    {
                        fullRepeats[i] = 0; // Special tokens (e.g. [CLS], [SEP]) are set to 0
                    }
                }
            }
            else
            {
                // Fallback: when word mapping is unavailable, try a simple one-to-one mapping
                // Assume the structure is [CLS] w1 w2 ... [SEP]
                for (int i = 0; i < Math.Min(ids.Length - 2, word2ph.Length); i++)
                {
                    fullRepeats[i + 1] = word2ph[i];
                }
            }

            // 3. Slice out the content part (remove the repeated counts for [CLS] and [SEP])
            // This particular ONNX model handles alignment via the repeats input and strips special tokens internally.
            long[] inferenceRepeats = new long[Math.Max(0, ids.Length - 2)];
            if (ids.Length > 2)
            {
                Array.Copy(fullRepeats, 1, inferenceRepeats, 0, ids.Length - 2);
            }

            if (DebugMode)
            {
                Console.WriteLine($"[BERT Debug] Text: {text}");
                Console.WriteLine($"[BERT Debug] IDs: {string.Join(",", ids)}");
                Console.WriteLine($"[BERT Debug] Repeats (Content): {string.Join(",", inferenceRepeats)}");
            }

            // 4. ONNX inference
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(ids, new[] { 1, ids.Length })),
                NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(mask, new[] { 1, mask.Length })),
                NamedOnnxValue.CreateFromTensor("repeats", new DenseTensor<long>(inferenceRepeats, new[] { inferenceRepeats.Length }))
            };

            // ONNX Runtime's InferenceSession allows concurrent Run calls from multiple threads.
            using var _results = _session.Run(inputs);
            var outputValue = _results.First().Value;

            // 5. Parse output dimensions and convert (the model aligns internally, so read directly)
            ReadOnlySpan<int> dims = outputValue switch
            {
                DenseTensor<Float16> f16 => f16.Dimensions,
                DenseTensor<float> f32 => f32.Dimensions,
                _ => throw new Exception($"Unsupported BERT output type: {outputValue?.GetType().FullName}")
            };

            int rows = dims[dims.Length - 2];
            int hidden = dims[dims.Length - 1];
            float[,] features = new float[rows, hidden];

            if (outputValue is DenseTensor<float> f32Tensor)
            {
                var flatData = f32Tensor.ToArray();
                Buffer.BlockCopy(flatData, 0, features, 0, flatData.Length * sizeof(float));
            }
            else if (outputValue is DenseTensor<Float16> f16Tensor)
            {
                var flatData = f16Tensor.ToArray();
                for (int i = 0; i < flatData.Length; i++)
                {
                    features[i / hidden, i % hidden] = (float)flatData[i];
                }
            }

            if (DebugMode)
            {
                float sum = 0, sumSq = 0;
                int total = features.Length;
                foreach (var f in features) { sum += f; sumSq += f * f; }
                float mean = total > 0 ? sum / total : 0;
                float std = total > 0 ? (float)Math.Sqrt(Math.Max(0, sumSq / total - mean * mean)) : 0;
                Console.WriteLine($"[BERT] Final Output Shape: ({rows}, {hidden}), mean={mean}, std={std}");
            }

            // 6. Write to the cache (idempotent; occasional duplicate computation for the same key under high concurrency is acceptable)
            if (UseCache && _cache != null && cacheKey != null)
            {
                _cache.Set(cacheKey, features);
            }

            return features;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
