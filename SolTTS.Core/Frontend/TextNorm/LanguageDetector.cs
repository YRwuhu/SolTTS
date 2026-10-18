using System.Linq;

namespace SolTTS.Core.Frontend.TextNorm
{
    /// <summary>
    /// Language detection and text preprocessing utilities
    /// </summary>
    public static class LanguageDetector
    {
        /// <summary>
        /// Language mode
        /// </summary>
        public enum LanguageMode
        {
            Chinese,  // Pure Chinese
            English,  // Pure English
            Mixed,    // Chinese-English mixed
            Japanese  // Contains Japanese kana
        }

        /// <summary>
        /// Detect the language mode of the text
        /// </summary>
        public static LanguageMode DetectMode(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return LanguageMode.English;

            bool hasChinese = false;
            bool hasEnglish = false;
            bool hasJapanese = false;

            foreach (char c in text)
            {
                // Japanese hiragana (U+3040-U+309F)
                if (c >= 0x3040 && c <= 0x309F)
                {
                    hasJapanese = true;
                }
                // Japanese katakana (U+30A0-U+30FF)
                else if (c >= 0x30A0 && c <= 0x30FF)
                {
                    hasJapanese = true;
                }
                // Chinese characters (CJK Unified Ideographs + Extension A)
                else if ((c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF))
                {
                    hasChinese = true;
                }
                // English letters
                else if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
                {
                    hasEnglish = true;
                }
                // Other characters (digits, punctuation, etc.) do not affect detection
            }

            // Japanese has highest priority (any kana is treated as Japanese)
            if (hasJapanese) return LanguageMode.Japanese;

            if (hasChinese && hasEnglish) return LanguageMode.Mixed;
            if (hasChinese && !hasEnglish) return LanguageMode.Chinese;
            if (hasEnglish && !hasChinese) return LanguageMode.English;

            // Neither Chinese nor English (digits/punctuation only): default to English mode
            return LanguageMode.English;
        }

        /// <summary>
        /// Convert full-width punctuation to half-width
        /// </summary>
        public static string NormalizePunctuation(string text)
        {
            if (string.IsNullOrEmpty(text)) return text;

            return text
                .Replace('\uFF0C', ',')  // \uFF0C = \uFF0C
                .Replace('\u3002', '.')  // \u3002 = \u3002
                .Replace('\uFF01', '!')  // \uFF01 = \uFF01
                .Replace('\uFF1F', '?')  // \uFF1F = \uFF1F
                .Replace('\uFF1A', ':')  // \uFF1A = \uFF1A
                .Replace('\uFF1B', ';')  // \uFF1B = \uFF1B
                .Replace('\u201C', '"')  // \u201C = \u201C
                .Replace('\u201D', '"')  // \u201D = \u201D
                .Replace('\u2018', '\'')  // \u2018 = \u2018
                .Replace('\u2019', '\'')  // \u2019 = \u2019
                .Replace('\uFF08', '(')  // \uFF08 = \uFF08
                .Replace('\uFF09', ')')  // \uFF09 = \uFF09
                .Replace('\u3010', '[')  // \u3010 = \u3010
                .Replace('\u3011', ']')  // \u3011 = \u3011
                .Replace('\u300A', '<')  // \u300A = \u300A
                .Replace('\u300B', '>'); // \u300B = \u300B
        }

        /// <summary>
        /// Split long text into sentences, supporting Chinese/English punctuation and newlines
        /// </summary>
        public static System.Collections.Generic.List<string> SplitSentences(string text)
        {
            var sentences = new System.Collections.Generic.List<string>();
            if (string.IsNullOrWhiteSpace(text)) return sentences;

            var sb = new System.Text.StringBuilder();
            char[] splitChars = { '。', '！', '？', '.', '!', '?', '\n', '\r' };
            bool inTag = false;

            for (int i = 0; i < text.Length; i++)
            {
                char c = text[i];

                // Track whether we are inside a tag {}
                if (c == '{') inTag = true;
                else if (c == '}') inTag = false;

                sb.Append(c);

                if (!inTag && System.Array.IndexOf(splitChars, c) >= 0)
                {
                    // Decimal point logic: if . is surrounded by digits, it is not a sentence terminator
                    if (c == '.')
                    {
                        bool isDigitBefore = i > 0 && char.IsDigit(text[i - 1]);
                        bool isDigitAfter = i < text.Length - 1 && char.IsDigit(text[i + 1]);
                        if (isDigitBefore && isDigitAfter) continue;
                    }

                    string s = sb.ToString().Trim();
                    if (!string.IsNullOrEmpty(s))
                    {
                        sentences.Add(s);
                    }
                    sb.Clear();
                }
            }

            string remaining = sb.ToString().Trim();
            if (!string.IsNullOrEmpty(remaining))
            {
                sentences.Add(remaining);
            }

            return sentences;
        }
    }
}
