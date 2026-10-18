using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace SolTTS.Core.Frontend.TextNorm
{
    /// <summary>
    /// Japanese text normalizer
    /// </summary>
    public static class JapaneseTextNormalizer
    {
        // Simplified Chinese -> common Japanese kanji variant mapping (avoids OpenJTalk dictionary lookup failures)
        private static readonly Dictionary<char, char> KanjiVariantMap = new Dictionary<char, char>
        {
            { '场', '場' }, { '语', '語' }, { '发', '発' }, { '经', '経' },
            { '过', '過' }, { '进', '進' }, { '开', '開' }, { '实', '実' },
            { '现', '現' }, { '对', '対' }, { '说', '説' }, { '话', '話' },
            { '长', '長' }, { '门', '門' }, { '间', '間' }, { '问', '問' },
            { '题', '題' }, { '东', '東' }, { '应', '応' }, { '该', '該' },
        };

        // Regex to collapse consecutive punctuation (prevents overly long pauses from breaking the model)
        private static readonly Regex ConsecutivePunctuation = new Regex(@"([,./?!~…・。、！？：；〜～])\1+", RegexOptions.Compiled);

        /// <summary>
        /// Normalize Japanese text
        /// </summary>
        public static string Normalize(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return text;

            // 1. Fast symbol replacement (avoids regex overhead)
            text = text.Replace("%", "パーセント").Replace("％", "パーセント")
                       .Replace("&", "アンド").Replace("＆", "アンド");

            // 2. Kanji variant conversion (Simplified Chinese -> Japanese kanji) and punctuation restoration (full-width)
            // Use StringBuilder to reduce allocations
            var sb = new StringBuilder(text.Length);
            foreach (char c in text)
            {
                if (KanjiVariantMap.TryGetValue(c, out char replaced)) sb.Append(replaced);
                else
                {
                    // Normalize punctuation to full-width; OpenJTalk handles full-width punctuation better
                    char target = c switch
                    {
                        '.' => '。',
                        ',' => '、',
                        '!' => '！',
                        '?' => '？',
                        ':' => '：',
                        ';' => '；',
                        _ => c
                    };
                    sb.Append(target);
                }
            }
            text = sb.ToString();

            // 3. Collapse consecutive punctuation
            text = ConsecutivePunctuation.Replace(text, "$1");

            return text;
        }

        /// <summary>
        /// Punctuation post-replacement (used during phoneme mapping)
        /// </summary>
        public static string PostReplacePunctuation(string phoneme)
        {
            return phoneme switch
            {
                "：" or "；" or "，" or "、" or "·" => ",",
                "。" => ".",
                "！" => "!",
                "？" => "?",
                "\n" => ".",
                "..." => "…",
                _ => phoneme
            };
        }
    }
}
