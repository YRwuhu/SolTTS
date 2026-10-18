using System.Text.Json;

namespace SolTTS.Core.Config
{
    /// <summary>
    /// Model configuration (corresponds to config.json)
    /// </summary>
    public class V2ModelConfig
    {
        public DataConfig? Data { get; set; }
        public ModelInfo? Model { get; set; }
        public ExportOptions? ExportOptions { get; set; }
        public Dictionary<string, int>? SymbolToId { get; set; }

        public static V2ModelConfig Load(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"Model config file not found: {path}");

            var json = File.ReadAllText(path);
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
            };

            return JsonSerializer.Deserialize<V2ModelConfig>(json, options)
                   ?? throw new InvalidDataException("Failed to parse model config file");
        }
    }

    /// <summary>
    /// Audio data configuration
    /// </summary>
    public class DataConfig
    {
        public int SamplingRate { get; set; } = 32000;
        public int FilterLength { get; set; } = 2048;
        public int HopLength { get; set; } = 640;
        public int WinLength { get; set; } = 2048;
        public int NSpeakers { get; set; } = 300;
    }

    /// <summary>
    /// Model information
    /// </summary>
    public class ModelInfo
    {
        public string Version { get; set; } = "v2";
        public string SemanticFrameRate { get; set; } = "25hz";
    }

    /// <summary>
    /// Export options
    /// </summary>
    public class ExportOptions
    {
        public int MaxLen { get; set; } = 1000;
        public int OpsetVersion { get; set; } = 17;
        public string Target { get; set; } = "directml";
    }
}
