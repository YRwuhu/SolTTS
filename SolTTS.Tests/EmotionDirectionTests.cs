using System;
using System.IO;
using SolTTS.Core.Core;
using Xunit;

namespace SolTTS.Tests
{
    /// <summary>
    /// Unit tests for the emotion direction vector controller: loading, normalization, α=0 consistency, dimension handling.
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

        [Fact]
        public void MissingDirectory_LoadsNothing_AndIsSilentlyDisabled()
        {
            var c = new EmotionDirectionController(Path.Combine(Path.GetTempPath(), "sol_missing_" + Guid.NewGuid().ToString("N")));
            Assert.False(c.Loaded);
            Assert.Empty(c.Emotions);
            Assert.Null(c.GetTau("happy"));
        }

        [Fact]
        public void LoadsTauFile_AndNormalizesToUnitVector()
        {
            var vec = new float[] { 3f, 4f, 0f };
            var dir = WriteTempTauDir(("happy", vec));
            try
            {
                var c = new EmotionDirectionController(dir);
                Assert.True(c.Loaded);
                var tau = c.GetTau("happy");
                Assert.NotNull(tau);
                float norm = MathF.Sqrt(tau![0] * tau[0] + tau[1] * tau[1] + tau[2] * tau[2]);
                Assert.InRange(norm, 0.9999f, 1.0001f);
                Assert.Equal(0.6f, tau[0], 4);
                Assert.Equal(0.8f, tau[1], 4);
            }
            finally { Directory.Delete(dir, true); }
        }

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
            // x_new = x + 0.2*5*tau = [3+1, 4+0] = [4, 4]
            Assert.Equal(4f, y[0], 5);
            Assert.Equal(4f, y[1], 5);
        }

        [Fact]
        public void Apply_HandlesDimensionMismatch_Gracefully()
        {
            var x = new float[] { 1f, 2f, 3f, 4f };
            var tau = new float[] { 1f, 0f };  // Shorter
            var y = EmotionDirectionController.Apply(x, tau, 1f);
            Assert.Equal(4, y.Length);
            Assert.Equal(x[2], y[2]); // Tail not covered by τ keeps its original value
            Assert.Equal(x[3], y[3]);
        }

        [Fact]
        public void Neutral_AndUnknownEmotion_ReturnNull()
        {
            var dir = WriteTempTauDir(("happy", new float[] { 1f, 0f }));
            try
            {
                var c = new EmotionDirectionController(dir);
                Assert.Null(c.GetTau("neutral")); // neutral does not exist in τ
                Assert.Null(c.GetTau("bogus"));
                Assert.NotNull(c.GetTau("HAPPY")); // Case-insensitive
            }
            finally { Directory.Delete(dir, true); }
        }

        [Fact]
        public void ManifestBasedLoading_Works()
        {
            var dir = Path.Combine(Path.GetTempPath(), "sol_tau_manifest_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                var vec = new float[] { 1f, 0f, 0f };
                var bytes = new byte[vec.Length * 4];
                Buffer.BlockCopy(vec, 0, bytes, 0, bytes.Length);
                File.WriteAllBytes(Path.Combine(dir, "tau_sad.f32"), bytes);
                File.WriteAllText(Path.Combine(dir, "manifest.json"),
                    "{\"dim\":3,\"emotions\":{\"sad\":{\"file\":\"tau_sad.f32\"}}}");

                var c = new EmotionDirectionController(dir);
                Assert.True(c.Loaded);
                Assert.Equal(3, c.Dim);
                Assert.NotNull(c.GetTau("sad"));
            }
            finally { Directory.Delete(dir, true); }
        }
    }
}
