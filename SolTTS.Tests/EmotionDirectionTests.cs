using System;
using System.IO;
using System.Linq;
using SolTTS.Core.Core;
using Xunit;

namespace SolTTS.Tests
{
    /// <summary>
    /// Unit tests for the emotion direction vector controller: catalog mapping, strict loading,
    /// α=0 invariants and rejection of unusable directions.
    /// </summary>
    public class EmotionDirectionTests
    {
        private static string WriteTempTauDir(params (string name, float[] vec)[] taus)
        {
            var dir = Path.Combine(Path.GetTempPath(), "sol_tau_test_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            foreach (var (name, vec) in taus)
            {
                using var fs = File.Create(Path.Combine(dir, $"tau_{name}.f32"));
                var bytes = new byte[vec.Length * 4];
                Buffer.BlockCopy(vec, 0, bytes, 0, bytes.Length);
                fs.Write(bytes, 0, bytes.Length);
            }
            return dir;
        }

        /// <summary>Write a full-dimension unit vector (20480) with only the first component non-zero.</summary>
        private static string WriteFullDimDir(params string[] names)
        {
            var dir = Path.Combine(Path.GetTempPath(), "sol_tau_full_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            foreach (var name in names)
            {
                var vec = new float[EmotionDirectionController.ExpectedDim];
                vec[0] = 1f;
                using var fs = File.Create(Path.Combine(dir, $"tau_{name}.f32"));
                var bytes = new byte[vec.Length * 4];
                Buffer.BlockCopy(vec, 0, bytes, 0, bytes.Length);
                fs.Write(bytes, 0, bytes.Length);
            }
            return dir;
        }

        // ------------------------------------------------------------------
        // Catalog
        // ------------------------------------------------------------------

        [Fact]
        public void Catalog_HasExactlyTheNineteenRequestedLabels()
        {
            Assert.Equal(19, EmotionCatalog.All.Count);
            Assert.Equal(
                new[] { "高兴", "平静", "厌恶", "害羞", "害怕", "生气", "认真", "紧张", "慌张",
                        "疑惑", "兴奋", "无奈", "担心", "惊讶", "哭泣", "心动", "难为情", "自信", "调皮" },
                EmotionCatalog.All.Select(e => e.ZhName).ToArray());
        }

        [Fact]
        public void Catalog_BaselineIsCalmAndHasNoDirection()
        {
            Assert.Equal("calm", EmotionCatalog.Baseline);
            Assert.DoesNotContain(EmotionCatalog.Directional, e => e.Id == "calm");
            Assert.Equal(18, EmotionCatalog.Directional.Count);
        }

        [Theory]
        [InlineData("happy", "happy")]
        [InlineData("高兴", "happy")]
        [InlineData("高兴 ", "happy")]
        [InlineData("lovestruck", "lovestruck")]
        [InlineData("心动", "lovestruck")]
        [InlineData("Lovestruck", "lovestruck")]
        public void Catalog_NormalizesIdsAndChineseNames(string input, string expected)
        {
            Assert.True(EmotionCatalog.TryNormalize(input, out var id));
            Assert.Equal(expected, id);
        }

        [Theory]
        [InlineData("neutral", "calm")]
        [InlineData("none", "calm")]
        [InlineData("sad", "crying")]
        [InlineData("fear", "fearful")]
        [InlineData("surprise", "surprised")]
        [InlineData("playfulness", "playful")]
        public void Catalog_MapsLegacyAliases(string legacy, string expected)
        {
            Assert.True(EmotionCatalog.TryNormalize(legacy, out var id));
            Assert.Equal(expected, id);
        }

        [Theory]
        [InlineData("bogus")]
        [InlineData("")]
        [InlineData("  ")]
        [InlineData(null)]
        public void Catalog_RejectsUnknownOrEmpty(string? input)
        {
            Assert.False(EmotionCatalog.TryNormalize(input, out _));
        }

        // ------------------------------------------------------------------
        // Loading
        // ------------------------------------------------------------------

        [Fact]
        public void MissingDirectory_LoadsNothing_AndIsSilentlyDisabled()
        {
            var c = new EmotionDirectionController(Path.Combine(Path.GetTempPath(), "sol_missing_" + Guid.NewGuid().ToString("N")));
            Assert.False(c.Loaded);
            Assert.Empty(c.Emotions);
            Assert.Null(c.GetTau("happy"));
            Assert.NotEmpty(c.LoadErrors);
        }

        [Fact]
        public void FullDimensionVector_LoadsAndIsUnitNormalized()
        {
            var dir = WriteFullDimDir("happy");
            try
            {
                var c = new EmotionDirectionController(dir);
                Assert.True(c.Loaded);
                Assert.Equal(EmotionDirectionController.ExpectedDim, c.Dim);
                var tau = c.GetTau("happy");
                Assert.NotNull(tau);
                Assert.Equal(1f, tau![0], 4);
                Assert.Equal(EmotionReadiness.Ready, c.Describe("happy")!.Readiness);
            }
            finally { Directory.Delete(dir, true); }
        }

        [Fact]
        public void WrongDimensionVector_IsRejectedWithAReason()
        {
            // 3-dim legacy vectors are no longer accepted: they cannot be injected into a 20480-dim embedding.
            var dir = WriteTempTauDir(("happy", new float[] { 3f, 4f, 0f }));
            try
            {
                var c = new EmotionDirectionController(dir);
                Assert.False(c.Loaded);
                Assert.Null(c.GetTau("happy"));
                Assert.Contains(c.LoadErrors, e => e.Contains("dimension"));
                Assert.Equal(EmotionReadiness.Unavailable, c.Describe("happy")!.Readiness);
            }
            finally { Directory.Delete(dir, true); }
        }

        [Fact]
        public void ZeroVector_IsRejected()
        {
            var dir = WriteTempTauDir(("happy", new float[EmotionDirectionController.ExpectedDim]));
            try
            {
                var c = new EmotionDirectionController(dir);
                Assert.False(c.Loaded);
                Assert.Contains(c.LoadErrors, e => e.Contains("zero"));
            }
            finally { Directory.Delete(dir, true); }
        }

        [Fact]
        public void NonFiniteVector_IsRejected()
        {
            var vec = new float[EmotionDirectionController.ExpectedDim];
            vec[0] = 1f;
            vec[10] = float.NaN;
            var dir = WriteTempTauDir(("happy", vec));
            try
            {
                var c = new EmotionDirectionController(dir);
                Assert.False(c.Loaded);
                Assert.Contains(c.LoadErrors, e => e.Contains("NaN") || e.Contains("Infinity"));
            }
            finally { Directory.Delete(dir, true); }
        }

        [Fact]
        public void ManifestBasedLoading_Works()
        {
            var dir = WriteFullDimDir("sad");
            try
            {
                File.WriteAllText(Path.Combine(dir, "manifest.json"),
                    "{\"dim\":20480,\"emotions\":{\"sad\":{\"file\":\"tau_sad.f32\",\"n_samples\":40,\"consistency\":0.4}}}");

                var c = new EmotionDirectionController(dir);
                Assert.True(c.Loaded);
                Assert.Equal(20480, c.Dim);
                Assert.NotNull(c.GetTau("sad"));
                var info = c.Describe("sad")!;
                Assert.Equal(40, info.SampleCount);
                Assert.Equal(EmotionReadiness.Ready, info.Readiness);
            }
            finally { Directory.Delete(dir, true); }
        }

        [Fact]
        public void Manifest_LowSampleCountMarksExperimental()
        {
            var dir = WriteFullDimDir("crying");
            try
            {
                File.WriteAllText(Path.Combine(dir, "manifest.json"),
                    "{\"dim\":20480,\"emotions\":{\"crying\":{\"file\":\"tau_crying.f32\",\"n_samples\":5}}}");
                var c = new EmotionDirectionController(dir);
                Assert.Equal(EmotionReadiness.Experimental, c.Describe("crying")!.Readiness);
            }
            finally { Directory.Delete(dir, true); }
        }

        [Fact]
        public void Manifest_BaselineVectorIsIgnored()
        {
            var dir = WriteFullDimDir("calm");
            try
            {
                File.WriteAllText(Path.Combine(dir, "manifest.json"),
                    "{\"dim\":20480,\"emotions\":{\"calm\":{\"file\":\"tau_calm.f32\"}}}");
                var c = new EmotionDirectionController(dir);
                Assert.False(c.Loaded);
                Assert.Equal(EmotionReadiness.Baseline, c.Describe("calm")!.Readiness);
                Assert.Contains(c.LoadErrors, e => e.Contains("baseline"));
            }
            finally { Directory.Delete(dir, true); }
        }

        [Fact]
        public void Manifest_PathTraversalFileNameIsRejected()
        {
            var dir = WriteFullDimDir("happy");
            try
            {
                File.WriteAllText(Path.Combine(dir, "manifest.json"),
                    "{\"dim\":20480,\"emotions\":{\"happy\":{\"file\":\"../../evil.f32\"}}}");
                var c = new EmotionDirectionController(dir);
                Assert.False(c.Loaded);
                Assert.Contains(c.LoadErrors, e => e.Contains("illegal file name"));
            }
            finally { Directory.Delete(dir, true); }
        }

        [Fact]
        public void MissingManifestEntry_IsReportedNotSilentlyIgnored()
        {
            var dir = WriteFullDimDir("happy");
            try
            {
                File.WriteAllText(Path.Combine(dir, "manifest.json"),
                    "{\"dim\":20480,\"emotions\":{\"happy\":{\"file\":\"tau_happy.f32\"}," +
                    "\"angry\":{\"file\":\"tau_angry.f32\"}}}");
                var c = new EmotionDirectionController(dir);
                Assert.True(c.Loaded);
                Assert.NotNull(c.GetTau("happy"));
                Assert.Null(c.GetTau("angry"));
                Assert.Contains(c.LoadErrors, e => e.Contains("angry"));
                Assert.Equal(EmotionReadiness.Unavailable, c.Describe("angry")!.Readiness);
                // The other 16 labels must still be visible as unavailable, not vanish from the catalog.
                Assert.Equal(19, c.Catalog().Count);
            }
            finally { Directory.Delete(dir, true); }
        }

        [Fact]
        public void Manifest_ExposesAllNineteenLabels()
        {
            var dir = WriteFullDimDir("happy");
            try
            {
                var c = new EmotionDirectionController(dir);
                var catalog = c.Catalog();
                Assert.Equal(19, catalog.Count);
                Assert.Equal(19, catalog.Select(e => e.Id).Distinct().Count());
                Assert.Single(catalog, e => e.IsBaseline);
                Assert.Equal(EmotionReadiness.Baseline, catalog.Single(e => e.IsBaseline).Readiness);
            }
            finally { Directory.Delete(dir, true); }
        }

        // ------------------------------------------------------------------
        // Resolution
        // ------------------------------------------------------------------

        [Fact]
        public void Resolve_BaselineAndEmptyMeanNoInjection()
        {
            var dir = WriteFullDimDir("happy");
            try
            {
                var c = new EmotionDirectionController(dir);
                foreach (var baseline in new[] { "calm", "neutral", "平静", "off", null })
                {
                    var (id, tau, error) = c.Resolve(baseline);
                    Assert.Null(id);
                    Assert.Null(tau);
                    Assert.Null(error);
                }
            }
            finally { Directory.Delete(dir, true); }
        }

        [Fact]
        public void Resolve_UnknownLabelReturnsError()
        {
            var dir = WriteFullDimDir("happy");
            try
            {
                var c = new EmotionDirectionController(dir);
                var (id, tau, error) = c.Resolve("bogus");
                Assert.Null(id);
                Assert.Null(tau);
                Assert.NotNull(error);
                Assert.Contains("unknown", error);
            }
            finally { Directory.Delete(dir, true); }
        }

        [Fact]
        public void Resolve_KnownButUnloadedReturnsError()
        {
            var dir = WriteFullDimDir("happy");
            try
            {
                var c = new EmotionDirectionController(dir);
                var (id, tau, error) = c.Resolve("心动");
                Assert.Null(id);
                Assert.Null(tau);
                Assert.NotNull(error);
                Assert.Contains("lovestruck", error);
            }
            finally { Directory.Delete(dir, true); }
        }

        [Fact]
        public void Resolve_ChineseNameReturnsCanonicalId()
        {
            var dir = WriteFullDimDir("lovestruck");
            try
            {
                var c = new EmotionDirectionController(dir);
                var (id, tau, error) = c.Resolve("心动");
                Assert.Null(error);
                Assert.Equal("lovestruck", id);
                Assert.NotNull(tau);
            }
            finally { Directory.Delete(dir, true); }
        }

        // ------------------------------------------------------------------
        // Injection math
        // ------------------------------------------------------------------

        [Fact]
        public void Apply_WithZeroAlpha_ReturnsBitwiseIdenticalCopy()
        {
            var x = new float[] { 1.5f, -2.25f, 0.125f, 42f };
            var tau = new float[] { 0.6f, 0.8f, 0f, 0f };
            var y = EmotionDirectionController.Apply(x, tau, 0f);
            Assert.Equal(x.Length, y.Length);
            for (int i = 0; i < x.Length; i++)
                Assert.Equal(x[i], y[i]); // Exactly equal (no floating-point error)
        }

        [Fact]
        public void Apply_ScalesByAlphaTimesNorm()
        {
            var x = new float[] { 3f, 4f };      // ||x|| = 5
            var tau = new float[] { 1f, 0f };    // Unit vector
            var y = EmotionDirectionController.Apply(x, tau, 0.2f);
            Assert.Equal(4f, y[0], 5);
            Assert.Equal(4f, y[1], 5);
        }

        [Fact]
        public void Apply_RejectsDimensionMismatch()
        {
            var x = new float[] { 1f, 2f, 3f, 4f };
            var tau = new float[] { 1f, 0f };
            Assert.Throws<ArgumentException>(() => EmotionDirectionController.Apply(x, tau, 1f));
        }

        [Theory]
        [InlineData(float.NaN)]
        [InlineData(float.PositiveInfinity)]
        [InlineData(-0.1f)]
        public void Apply_RejectsInvalidAlpha(float alpha)
        {
            var x = new float[] { 1f, 2f };
            var tau = new float[] { 1f, 0f };
            Assert.Throws<ArgumentOutOfRangeException>(() => EmotionDirectionController.Apply(x, tau, alpha));
        }
    }
}