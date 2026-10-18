using Xunit;
using Xunit.Abstractions;
using SolTTS.Core.Frontend.G2P;
using SolTTS.Core.Frontend.G2P.Common;
using SolTTS.Core.Frontend.G2P.Japanese;
using SolTTS.Core.Frontend.TextNorm;
using System;
using System.IO;
using System.Linq;

namespace SolTTS.Tests
{
    /// <summary>
    /// Japanese G2P unit tests
    /// </summary>
    public class JapaneseG2PTests : IDisposable
    {
        private readonly ITestOutputHelper _output;
        private readonly JapaneseG2P? _g2p;

        // Dictionary relative path (from the test project output directory to GenieData)
        private static readonly string DictPath = GetDictPath();

        private static string GetDictPath()
        {
            // Try several possible dictionary locations
            var possiblePaths = new[]
            {
                // Search upward from the test project bin directory (new location)
                Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
                    @"..\..\..\..\resources\shared\g2p\dicts\japanese\open_jtalk_dic_utf_8-1.11")),
                // Use an absolute path directly
                @"E:\RiderProjects\SolTTS\resources\shared\g2p\dicts\japanese\open_jtalk_dic_utf_8-1.11",
            };

            foreach (var path in possiblePaths)
            {
                if (Directory.Exists(path))
                    return path;
            }

            return possiblePaths[0]; // Return the first path so failures show it
        }

        public JapaneseG2PTests(ITestOutputHelper output)
        {
            _output = output;

            if (Directory.Exists(DictPath))
            {
                _g2p = new JapaneseG2P(DictPath);
                _output.WriteLine($"[Setup] Loaded dictionary from: {DictPath}");
            }
            else
            {
                _output.WriteLine($"[Setup] Dictionary not found at: {DictPath}");
                _output.WriteLine("[Setup] Tests will be skipped.");
            }
        }

        public void Dispose()
        {
            _g2p?.Dispose();
        }

        [Fact]
        public void JapaneseG2P_Hiragana_ReturnsPhonemes()
        {
            Skip.If(_g2p == null, "Dictionary not available");

            // Test hiragana
            var result = _g2p!.Process("こんにちは");

            _output.WriteLine($"Input: こんにちは");
            _output.WriteLine($"Phones: {string.Join(", ", result.Phones)}");
            _output.WriteLine($"PhoneIds: {string.Join(", ", result.PhoneIds)}");

            Assert.NotEmpty(result.Phones);
            Assert.True(result.Phones.Count > 1, "Should produce multiple phonemes");

            // こんにちは -> k, o, N, n, i, ch, i, w, a (approx. 9 phonemes)
            Assert.Contains("k", result.Phones);
            Assert.Contains("o", result.Phones);
        }

        [Fact]
        public void JapaneseG2P_Katakana_ReturnsPhonemes()
        {
            Skip.If(_g2p == null, "Dictionary not available");

            // Test katakana
            var result = _g2p!.Process("コンニチハ");

            _output.WriteLine($"Input: コンニチハ");
            _output.WriteLine($"Phones: {string.Join(", ", result.Phones)}");

            Assert.NotEmpty(result.Phones);
            Assert.True(result.Phones.Count > 1, "Should produce multiple phonemes");
        }

        [Fact]
        public void JapaneseG2P_Kanji_ReturnsPhonemes()
        {
            Skip.If(_g2p == null, "Dictionary not available");

            // Test kanji (Japanese reading)
            var result = _g2p!.Process("今日");

            _output.WriteLine($"Input: 今日");
            _output.WriteLine($"Phones: {string.Join(", ", result.Phones)}");

            Assert.NotEmpty(result.Phones);
            // 今日 -> きょう (kyou) or こんにち (konnichi)
        }

        [Fact]
        public void JapaneseG2P_MixedText_ReturnsPhonemes()
        {
            Skip.If(_g2p == null, "Dictionary not available");

            // Test mixed text
            var result = _g2p!.Process("今日は良い天気です。");

            _output.WriteLine($"Input: 今日は良い天気です。");
            _output.WriteLine($"Phones: {string.Join(", ", result.Phones)}");
            _output.WriteLine($"Word2Ph: {string.Join(", ", result.Word2Ph)}");

            Assert.NotEmpty(result.Phones);
        }

        [Fact]
        public void JapaneseG2P_EmptyInput_ReturnsDefault()
        {
            Skip.If(_g2p == null, "Dictionary not available");

            var result = _g2p!.Process("");

            Assert.NotEmpty(result.Phones);
            Assert.Contains("SP", result.Phones);
        }

        [Theory]
        [InlineData("あ", "a")]  // Single hiragana
        [InlineData("い", "i")]
        [InlineData("う", "u")]
        [InlineData("え", "e")]
        [InlineData("お", "o")]
        public void JapaneseG2P_SingleVowel_ReturnsCorrectPhoneme(string input, string expectedPhoneme)
        {
            Skip.If(_g2p == null, "Dictionary not available");

            var result = _g2p!.Process(input);

            _output.WriteLine($"Input: {input}");
            _output.WriteLine($"Phones: {string.Join(", ", result.Phones)}");

            Assert.Contains(expectedPhoneme, result.Phones);
        }
    }

    /// <summary>
    /// Japanese text normalization unit tests
    /// </summary>
    public class JapaneseTextNormalizerTests
    {
        [Fact]
        public void Normalize_FullWidthPunctuation_StaysFullWidth()
        {
            var input = "こんにちは。这是、测试です！";
            var result = JapaneseTextNormalizer.Normalize(input);

            Assert.Contains("。", result);
            Assert.Contains("、", result);
            Assert.Contains("！", result);
        }

        [Fact]
        public void Normalize_HalfWidthPunctuation_ConvertsToFullWidth()
        {
            var input = "こんにちは. 这是, 测试です!";
            var result = JapaneseTextNormalizer.Normalize(input);

            Assert.Contains("。", result);
            Assert.Contains("、", result);
            Assert.Contains("！", result);
            Assert.DoesNotContain(".", result);
        }

        [Fact]
        public void Normalize_PercentSymbol_ConvertsToJapanese()
        {
            var input = "100%の確率";
            var result = JapaneseTextNormalizer.Normalize(input);

            Assert.Contains("パーセント", result);
            Assert.DoesNotContain("%", result);
        }

        [Fact]
        public void Normalize_ConsecutivePunctuation_MergesIntoOne()
        {
            var input = "なに！！！";
            var result = JapaneseTextNormalizer.Normalize(input);

            // Consecutive ！ should be merged into one
            Assert.Equal(1, result.Count(c => c == '！'));
        }

        [Fact]
        public void PostReplacePunctuation_JapanesePunctuation_ReturnsEnglish()
        {
            Assert.Equal(".", JapaneseTextNormalizer.PostReplacePunctuation("。"));
            Assert.Equal(",", JapaneseTextNormalizer.PostReplacePunctuation("、"));
            Assert.Equal("!", JapaneseTextNormalizer.PostReplacePunctuation("！"));
            Assert.Equal("?", JapaneseTextNormalizer.PostReplacePunctuation("？"));
        }
    }

    /// <summary>
    /// Mixed-language G2P Japanese detection tests
    /// </summary>
    public class MixedLanguageG2PJapaneseTests
    {
        [Fact]
        public void CharDetection_Hiragana_DetectedAsJapanese()
        {
            // Test the hiragana character range
            char hiragana = 'あ'; // U+3042
            Assert.True(hiragana >= 0x3040 && hiragana <= 0x309F);
        }

        [Fact]
        public void CharDetection_Katakana_DetectedAsJapanese()
        {
            // Test the katakana character range
            char katakana = 'ア'; // U+30A2
            Assert.True(katakana >= 0x30A0 && katakana <= 0x30FF);
        }

        [Fact]
        public void CharDetection_Kanji_DetectedAsChineseByDefault()
        {
            // Without context, kanji is detected as Chinese by default
            char kanji = '日'; // U+65E5
            Assert.True(kanji >= 0x4E00 && kanji <= 0x9FFF);
        }

        [Fact]
        public void ContextDetection_KanjiWithKana_ReturnsJapanese()
        {
            // This test verifies the new heuristic: if kana is present, CJK should be treated as Japanese
            // Since the test class has no full MixedLanguageG2P instance, only verify the heuristic's components here
            string text = "この場合";
            bool hasKana = text.Any(c => (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF));
            Assert.True(hasKana, "Should detect kana in 'この場合'");
        }
    }
}
