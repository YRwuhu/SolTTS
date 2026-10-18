using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace SolTTS.Core.Frontend.G2P.English
{
    public enum TextSegmentType { Normal, Native }

    public class TextSegment
    {
        public string Text { get; set; } = string.Empty;
        public string? Language { get; set; }
        public TextSegmentType Type { get; set; } = TextSegmentType.Normal;
    }

    /// <summary>
    /// Parses tagged text, recognizing {zh content} language switches and {ni3}{hao3} native phoneme blocks.
    /// </summary>
    public static class TaggedTextParser
    {
        // Matches {lang content}, where lang is 2-3 letters and content may contain spaces (non-greedy)
        private static readonly Regex UnifiedTagRegex = new Regex(@"{([a-zA-Z]{2,3})\s+(.*?)}", RegexOptions.Compiled | RegexOptions.Singleline);
        // Matches {ni3} native phonemes, dispatched by MixedLanguageG2P
        private static readonly Regex NativeTagRegex = new Regex(@"({[^{}\s]+})", RegexOptions.Compiled);

        public static List<TextSegment> Parse(string text)
        {
            var result = new List<TextSegment>();
            if (string.IsNullOrEmpty(text)) return result;

            int lastIdx = 0;
            var matches = UnifiedTagRegex.Matches(text);



            foreach (Match match in matches)
            {
                if (match.Index > lastIdx)
                {
                    ProcessNative(text.Substring(lastIdx, match.Index - lastIdx), null, result);
                }

                string lang = match.Groups[1].Value;
                string content = match.Groups[2].Value;



                // Content inside a language tag may itself contain native phoneme blocks, e.g. {zh GPT {ni3}}
                // For simplicity, nesting is not supported yet; such content is treated as plain text
                result.Add(new TextSegment { Text = content, Language = lang, Type = TextSegmentType.Normal });

                lastIdx = match.Index + match.Length;
            }

            if (lastIdx < text.Length)
            {
                ProcessNative(text.Substring(lastIdx), null, result);
            }

            return result;
        }

        private static void ProcessNative(string text, string? lang, List<TextSegment> result)
        {
            if (string.IsNullOrEmpty(text)) return;

            int lastIdx = 0;
            var matches = NativeTagRegex.Matches(text);
            foreach (Match match in matches)
            {
                if (match.Index > lastIdx)
                {
                    result.Add(new TextSegment { Text = text.Substring(lastIdx, match.Index - lastIdx), Language = lang, Type = TextSegmentType.Normal });
                }
                result.Add(new TextSegment { Text = match.Value, Language = lang, Type = TextSegmentType.Native });
                lastIdx = match.Index + match.Length;
            }
            if (lastIdx < text.Length)
            {
                result.Add(new TextSegment { Text = text.Substring(lastIdx), Language = lang, Type = TextSegmentType.Normal });
            }
        }
    }
}
