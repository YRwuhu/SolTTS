using System.Buffers;
using System.Runtime.InteropServices;
using SolTTS.Core.Core;
using SolTTS.Core.Utils;
using NAudio.Wave;

namespace SolTTS.Tests;

/// <summary>
/// Equivalence tests between native audio primitives and the managed implementation.
/// The whole group is skipped when the native DLL is missing (SkippableFact), so environments without a Rust toolchain are unaffected.
/// </summary>
public class NativeAudioTests
{
    private static bool Native => NativeAudio.IsAvailable;

    [SkippableFact]
    public void RingBuffer_ProducesIdenticalBytesToManaged()
    {
        Skip.IfNot(Native, $"native sol_audio unavailable: {NativeAudio.UnavailableReason}");

        const int chunk = 4096;
        const int rounds = 64;
        var format = new WaveFormat(32000, 16, 1);
        const int capacity = 65536 * 8;

        var managed = new LockFreeWaveProvider(format, capacity);
        var native = new LockFreeWaveProvider(format, capacity, useNative: true);
        Assert.False(managed.IsNative);
        Assert.True(native.IsNative);

        var rng = new Random(12345);
        var produced = new List<byte[]>();
        var managedOut = new List<byte>();
        var nativeOut = new List<byte>();
        var readBufM = new byte[chunk];
        var readBufN = new byte[chunk];

        for (int r = 0; r < rounds; r++)
        {
            var payload = new byte[chunk];
            rng.NextBytes(payload);
            produced.Add(payload);

            managed.AddSamples(payload, 0, payload.Length);
            native.AddSamples(payload, 0, payload.Length);

            Assert.Equal(managed.BufferedBytes, native.BufferedBytes);

            // Consumption length varies slightly each round to cover wrap-around paths
            int want = chunk - (r % 7) * 64;
            int gotM = managed.Read(readBufM, 0, want);
            int gotN = native.Read(readBufN, 0, want);
            Assert.Equal(gotM, gotN);
            managedOut.AddRange(readBufM.AsSpan(0, gotM).ToArray());
            nativeOut.AddRange(readBufN.AsSpan(0, gotN).ToArray());
        }

        Assert.Equal(managedOut.Count, nativeOut.Count);
        Assert.True(managedOut.SequenceEqual(nativeOut), "native ring bytes differ from managed");

        // Padding statistics should also match (with ReadFully=true, empty reads return silence)
        managed.Read(readBufM, 0, chunk);
        native.Read(readBufN, 0, chunk);
        Assert.Equal(managed.PaddingBytes, native.PaddingBytes);

        native.Dispose();
    }

    [SkippableFact]
    public void HighPassFilter_IsBitExact()
    {
        Skip.IfNot(Native, $"native sol_audio unavailable: {NativeAudio.UnavailableReason}");

        const int n = 32000;
        var managedInput = new float[n];
        var nativeInput = new float[n];
        var rng = new Random(7);
        for (int i = 0; i < n; i++)
        {
            float v = (float)(rng.NextDouble() * 2 - 1);
            managedInput[i] = v;
            nativeInput[i] = v;
        }
        // Add silent segments and extremes to cover state accumulation
        for (int i = 1000; i < 2000; i++) { managedInput[i] = 0; nativeInput[i] = 0; }
        managedInput[0] = 1f; nativeInput[0] = 1f;
        managedInput[1] = -1f; nativeInput[1] = -1f;

        var hpf = new HighPassFilter(32000, 20);
        hpf.Process(managedInput.AsSpan());

        float alpha = NativeAudio.HighPassAlpha(32000, 20);
        var state = new float[2];
        NativeAudio.HighPassProcess(nativeInput.AsSpan(), alpha, state);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(
                BitConverter.SingleToInt32Bits(managedInput[i]),
                BitConverter.SingleToInt32Bits(nativeInput[i]));
        }
    }

    [SkippableFact]
    public void CrossFade_MatchesManagedLoop()
    {
        Skip.IfNot(Native, $"native sol_audio unavailable: {NativeAudio.UnavailableReason}");

        const int n = 640; // 20ms @ 32k
        var previousTail = new float[n];
        var currentChunk = new float[n];
        var expected = new float[n];
        for (int i = 0; i < n; i++)
        {
            previousTail[i] = (float)Math.Sin(i * 0.01);
            currentChunk[i] = (float)Math.Cos(i * 0.013);
            float alpha = (float)i / n;
            // Consistent with AudioPipeline: previous tail fades out, current chunk fades in
            expected[i] = previousTail[i] * (1f - alpha) + currentChunk[i] * alpha;
        }

        var actual = (float[])currentChunk.Clone();
        NativeAudio.CrossFade(actual.AsSpan(), previousTail, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(actual[i]));
    }

    [SkippableFact]
    public void FloatToPcm16_MatchesManagedConversion()
    {
        Skip.IfNot(Native, $"native sol_audio unavailable: {NativeAudio.UnavailableReason}");

        var values = new HashSet<float> { -1f, 1f, 0f, -0f, 0.5f, -0.5f, 1.5f, -1.5f, 1f / 3f, float.Epsilon };
        for (int i = 0; i < 5000; i++) values.Add((float)(Random.Shared.NextDouble() * 2.4 - 1.2));
        var src = values.ToArray();

        var expected = new short[src.Length];
        for (int i = 0; i < src.Length; i++)
            expected[i] = (short)(Math.Clamp(src[i], -1f, 1f) * 32767);

        var actual = new short[src.Length];
        NativeAudio.FloatToPcm16(src, actual);

        Assert.Equal(expected, actual);
    }

    [SkippableFact]
    public void AudioPipeline_NativeAndManagedProduceSamePcm()
    {
        Skip.IfNot(Native, $"native sol_audio unavailable: {NativeAudio.UnavailableReason}");

        const int sampleRate = 32000;
        const int chunkSamples = 8000;
        var rng = new Random(99);
        var source = new float[chunkSamples * 6];
        for (int i = 0; i < source.Length; i++)
            source[i] = (float)(Math.Sin(i * 0.002) * 0.4 + (rng.NextDouble() - 0.5) * 0.05);

        using var managed = new AudioPipeline(sampleRate, 20, 20, useNative: false);
        using var native = new AudioPipeline(sampleRate, 20, 20, useNative: true);
        Assert.True(native.IsNative);

        for (int c = 0; c < 6; c++)
        {
            bool isFinal = c == 5;
            var managedChunk = (float[])source.Clone();
            var nativeChunk = (float[])source.Clone();

            var (pcmM, lenM) = managed.ProcessChunk(managedChunk, chunkSamples, isFinal);
            var (pcmN, lenN) = native.ProcessChunk(nativeChunk, chunkSamples, isFinal);

            try
            {
                Assert.Equal(lenM, lenN);
                var spanM = MemoryMarshal.Cast<byte, short>(pcmM.AsSpan(0, lenM));
                var spanN = MemoryMarshal.Cast<byte, short>(pcmN.AsSpan(0, lenN));
                Assert.True(spanM.SequenceEqual(spanN), $"PCM mismatch at chunk {c}");
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(pcmM);
                ArrayPool<byte>.Shared.Return(pcmN);
            }
        }
    }

    [SkippableFact]
    public void RingBuffer_ConcurrentProducerConsumer_LosesNoBytes()
    {
        Skip.IfNot(Native, $"native sol_audio unavailable: {NativeAudio.UnavailableReason}");

        var format = new WaveFormat(32000, 16, 1);
        using var provider = new LockFreeWaveProvider(format, 65536, useNative: true);

        const int totalChunks = 2000;
        const int chunkBytes = 128; // Small chunks create wrap-around pressure
        var received = new List<byte>(totalChunks * chunkBytes);
        var done = new ManualResetEventSlim();
        // The consumer does not use ReadFully (to avoid counting silence padding), so the producer must guarantee no dropped chunks:
        // the 64KB capacity is far larger than a single 128B chunk, so writing after checking available space suffices.
        provider.ReadFully = false;

        var consumer = new Thread(() =>
        {
            var buf = new byte[chunkBytes];
            while (received.Count < totalChunks * chunkBytes)
            {
                int got = provider.Read(buf, 0, buf.Length);
                if (got > 0)
                {
                    received.AddRange(buf.AsSpan(0, got).ToArray());
                }
                else
                {
                    Thread.SpinWait(10);
                }
            }
            done.Set();
        }) { IsBackground = true };
        consumer.Start();

        var payload = new byte[chunkBytes];
        const int capacity = 65536;
        for (int i = 0; i < totalChunks; i++)
        {
            BitConverter.GetBytes((short)(i % 32000)).CopyTo(payload, 0);
            for (int b = 2; b < payload.Length; b++) payload[b] = (byte)(i + b);

            // Wait for the consumer to free space when capacity is insufficient (AddSamples silently drops when full)
            var spin = new SpinWait();
            while (capacity - provider.BufferedBytes < chunkBytes) spin.SpinOnce();

            provider.AddSamples(payload, 0, payload.Length);
        }

        Assert.True(done.Wait(TimeSpan.FromSeconds(10)), "consumer did not finish");
        Assert.Equal(totalChunks * chunkBytes, received.Count);
    }
}
