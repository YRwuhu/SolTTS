using System.Collections.Generic;

namespace SolTTS.Core.Frontend.G2P.Common
{
    /// <summary>
    /// Language type associated with a phoneme
    /// </summary>
    public enum PhoneLanguage : byte
    {
        Chinese,
        English,
        Japanese,
        Other  // SP, punctuation, etc.
    }

    /// <summary>
    /// Language segment information, used for segmented BERT processing
    /// </summary>
    public struct LanguageSegment
    {
        /// <summary>Original text segment</summary>
        public string Text;
        /// <summary>Language type</summary>
        public PhoneLanguage Language;
        /// <summary>Starting index in the Phones list</summary>
        public int StartPhoneIndex;
        /// <summary>Number of phonemes</summary>
        public int PhoneCount;
        /// <summary>Word2Ph array for this segment</summary>
        public int[] Word2Ph;
    }

    public struct G2PResult
    {
        public string NormalizedText;
        public List<string> Phones;
        public long[] PhoneIds;
        public int[] Word2Ph;

        /// <summary>
        /// Language tag for each phoneme, used for segmented BERT processing.
        /// Same length as Phones.
        /// </summary>
        public PhoneLanguage[]? LanguageTags;

        /// <summary>
        /// List of language segments with text and position information for each segment.
        /// Used for segmented BERT processing.
        /// </summary>
        public List<LanguageSegment>? Segments;
    }

    public interface IG2P
    {
        G2PResult Process(string text, int? priorityMode = null);
    }
}
