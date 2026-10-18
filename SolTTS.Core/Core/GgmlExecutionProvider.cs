using Microsoft.ML.OnnxRuntime;
using System.Runtime.InteropServices;

namespace SolTTS.Core.Core
{
    /// <summary>
    /// C# wrapper for the onnxruntime-ggml plugin execution provider.
    ///
    /// This provider compiles ONNX graphs to ggml kernels (Vulkan on Windows/Linux, Metal on macOS)
    /// and can deliver several-fold speedups for large parallel convolution models such as the vits vocoder.
    /// Measured: the vits vocoder is about 5x faster than CPU on an AMD RX 5500 XT, with spectral envelope correlation 0.994.
    ///
    /// Mechanism: ggml registers as a plugin EP with ORT (requires the ORT 1.29+ plugin API).
    /// Once registered it appears as an OrtEpDevice named "ggml" and is attached to a session via
    /// SessionOptions.AppendExecutionProvider(env, devices, options).
    ///
    /// Note: the provider's graph-claiming strategy is all-or-nothing.
    /// Graphs containing unsupported operators (e.g. ArgMax) are rejected wholesale and silently fall back to the CPU EP.
    /// </summary>
    public static class GgmlExecutionProvider
    {
        private static readonly object _lock = new();
        private static bool _registered;
        private static string? _libraryPath;

        /// <summary>
        /// Whether registration has succeeded (or been attempted).
        /// </summary>
        public static bool IsRegistered
        {
            get { lock (_lock) return _registered; }
        }

        /// <summary>
        /// Register the ggml EP library (only once per process). Returns false if the library is missing or registration fails; callers fall back to CPU.
        /// </summary>
        public static bool EnsureRegistered(string? libraryPath = null)
        {
            lock (_lock)
            {
                if (_registered) return true;

                try
                {
                    string path = libraryPath ?? ResolveDefaultLibraryPath();
                    if (!File.Exists(path))
                    {
                        return false;
                    }

                    OrtEnv.Instance().RegisterExecutionProviderLibrary("ggml", path);
                    _libraryPath = path;
                    _registered = true;
                    return true;
                }
                catch
                {
                    // Registration failed (missing Vulkan runtime, ORT version mismatch, etc.) — degrade silently; callers use CPU.
                    return false;
                }
            }
        }

        /// <summary>
        /// Get the ggml EP device list (empty when not registered or unavailable).
        /// </summary>
        public static IReadOnlyList<OrtEpDevice> GetDevices()
        {
            if (!_registered) return System.Array.Empty<OrtEpDevice>();
            try
            {
                var devices = new List<OrtEpDevice>();
                foreach (var d in OrtEnv.Instance().GetEpDevices())
                {
                    if (string.Equals(d.EpName, "ggml", System.StringComparison.OrdinalIgnoreCase))
                        devices.Add(d);
                }
                return devices;
            }
            catch
            {
                return System.Array.Empty<OrtEpDevice>();
            }
        }

        /// <summary>
        /// Build SessionOptions with the ggml EP attached; returns null when unavailable (callers fall back to CPU).
        /// </summary>
        /// <param name="device">gpu / cpu / auto. gpu requires a Vulkan device.</param>
        /// <param name="gpuMinWeightMb">
        /// Graphs with weights below this threshold are forced onto ggml's CPU kernels (small graphs suffer from GPU startup overhead).
        /// The vocoder weighs about 250MB; set 0 to force GPU.
        /// </param>
        public static SessionOptions? TryCreateSessionOptions(string device = "gpu", int gpuMinWeightMb = 0, bool partial = false, bool profile = false)
        {
            if (!EnsureRegistered()) return null;

            var devices = GetDevices();
            if (devices.Count == 0) return null;

            try
            {
                var so = new SessionOptions();
                var options = new Dictionary<string, string>
                {
                    ["device"] = device,
                    ["gpu_min_weight_mb"] = gpuMinWeightMb.ToString(),
                };
                if (partial) options["partial"] = "1";
                if (profile) options["profile"] = "1";

                so.AppendExecutionProvider(OrtEnv.Instance(), devices, options);
                so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC;
                so.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
                return so;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Default search paths for the ggml runtime:
        ///   1. SOLTTS_GGML_LIBRARY environment variable (explicit override)
        ///   2. onnxruntime_ggml.dll next to the assembly (shipped with the release package)
        /// </summary>
        private static string ResolveDefaultLibraryPath()
        {
            string? env = Environment.GetEnvironmentVariable("SOLTTS_GGML_LIBRARY");
            if (!string.IsNullOrWhiteSpace(env)) return env;

            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string name = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? "onnxruntime_ggml.dll"
                : RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
                    ? "libonnxruntime_ggml.dylib"
                    : "libonnxruntime_ggml.so";
            return Path.Combine(baseDir, name);
        }
    }
}
