using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace SolTTS.Core.Frontend.G2P.Chinese
{
    /// <summary>
    /// Polyphone correction processor.
    /// Loads the polyphonic.json dictionary for forced corrections.
    /// </summary>
    public class CorrectPronunciation
    {
        private readonly Dictionary<string, string[]> _dict;

        public CorrectPronunciation(string dictPath)
        {
            _dict = new Dictionary<string, string[]>();
            if (File.Exists(dictPath))
            {
                try
                {
                    string json = File.ReadAllText(dictPath);
                    _dict = JsonSerializer.Deserialize<Dictionary<string, string[]>>(json) ?? new Dictionary<string, string[]>();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[CorrectPronunciation] Error loading dict: {ex.Message}");
                }
            }
        }

        /// <summary>
        /// Correct the final list for the given word.
        /// </summary>
        public List<string> Correct(string word, List<string> finals)
        {
            if (_dict.TryGetValue(word, out string[]? correctedFinals))
            {
                if (correctedFinals.Length == finals.Count)
                {
                    return new List<string>(correctedFinals);
                }
            }
            return finals;
        }
    }
}
