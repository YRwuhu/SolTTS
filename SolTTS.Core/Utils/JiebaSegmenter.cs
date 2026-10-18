using System.Collections.Generic;
using System.Linq;
using JiebaNet.Segmenter;
using JiebaNet.Segmenter.PosSeg;

namespace SolTTS.Core.Utils
{
    /// <summary>
    /// Segmenter based on jieba.NET, with part-of-speech tagging (POS tagging).
    /// </summary>
    public class JiebaSegmenter
    {
        private readonly PosSegmenter _posSeg;

        public JiebaSegmenter(string dictDir)
        {
            if (!string.IsNullOrEmpty(dictDir))
            {
                ConfigManager.ConfigFileBaseDir = dictDir;
            }
            _posSeg = new PosSegmenter();
        }

        /// <summary>
        /// Segment text and return words with their part-of-speech tags.
        /// </summary>
        public List<(string word, string pos)> Cut(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new List<(string word, string pos)>();

            var result = _posSeg.Cut(text);
            return result.Select(p => (p.Word, p.Flag)).ToList();
        }

        /// <summary>
        /// Also supports search-mode segmentation (used for subword checks in the PreMerge logic).
        /// </summary>
        public List<string> CutForSearch(string text)
        {
            // Note: PosSegmenter doesn't have CutForSearch directly with POS.
            // But we can use the regular segmenter if needed.
            // For now, simple Cut is enough for G2P.
            return _posSeg.Cut(text).Select(p => p.Word).ToList();
        }
    }
}
