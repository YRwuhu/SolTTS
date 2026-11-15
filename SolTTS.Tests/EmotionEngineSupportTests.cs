using SolTTS.Core.Core;
using Xunit;

namespace SolTTS.Tests
{
    /// <summary>
    /// The V2 engine has no emotion path. An explicit emotion request must fail loudly rather than
    /// silently producing neutral audio that looks like it worked.
    /// </summary>
    public class EmotionEngineSupportTests
    {
        [Theory]
        [InlineData("happy")]
        [InlineData("高兴")]
        [InlineData("lovestruck")]
        public void V2_RejectsExplicitEmotion(string emotion)
        {
            var options = new TtsOptions { EmotionId = emotion };
            Assert.Throws<NotSupportedException>(() => TtsEngineV2.RejectUnsupportedEmotion(options));
        }

        [Theory]
        [InlineData(null)]
        [InlineData("")]
        [InlineData("calm")]
        [InlineData("neutral")]
        [InlineData("平静")]
        public void V2_AllowsBaselineOrOmitted(string? emotion)
        {
            var options = new TtsOptions { EmotionId = emotion };
            TtsEngineV2.RejectUnsupportedEmotion(options); // must not throw
        }

        [Fact]
        public void V2_RejectionMessageNamesTheEmotionAndTheFix()
        {
            var ex = Assert.Throws<NotSupportedException>(
                () => TtsEngineV2.RejectUnsupportedEmotion(new TtsOptions { EmotionId = "害羞" }));
            Assert.Contains("shy", ex.Message);
            Assert.Contains("V2", ex.Message);
            Assert.Contains("UseEngineV2=false", ex.Message);
        }
    }
}