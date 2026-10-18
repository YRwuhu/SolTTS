using System;
using System.Buffers;
using System.Runtime.InteropServices;
using SolTTS.Core.Utils;

namespace SolTTS.Core.Core
{
    /// <summary>
    /// Audio post-processing pipeline.
    /// Combines crossfade, high-pass filtering and PCM conversion for zero-GC operation.
    /// </summary>
    public class AudioPipeline : IDisposable
    {
        private readonly int _sampleRate;
        private readonly int _crossFadeSamples;
        private readonly HighPassFilter _hpFilter;
        private readonly bool _useNative;
        private readonly float _hpAlpha;
        private readonly float[] _hpState = new float[2];

        private float[]? _prevChunkTail;
        private int _prevTailLen;

        private readonly object _lock = new();

        public AudioPipeline(int sampleRate = 32000, int crossFadeMs = 20, int hpCutoff = 20, bool useNative = false)
        {
            _sampleRate = sampleRate;
            _crossFadeSamples = (sampleRate * crossFadeMs) / 1000;
            _hpFilter = new HighPassFilter(sampleRate, hpCutoff);
            _useNative = useNative && NativeAudio.IsAvailable;
            _hpAlpha = _useNative ? NativeAudio.HighPassAlpha(sampleRate, hpCutoff) : 0f;

            // Preallocate the tail buffer used for crossfade
            _prevChunkTail = ArrayPool<float>.Shared.Rent(_crossFadeSamples);
            _prevTailLen = 0;
        }

        /// <summary>Whether the native implementation is currently in use (diagnostics only).</summary>
        public bool IsNative => _useNative;

        /// <summary>
        /// Process one audio chunk.
        /// </summary>
        /// <param name="input">Input audio (float[])</param>
        /// <param name="isFinal">Whether this is the last chunk</param>
        /// <returns>Rented byte[] buffer (PCM16). The caller must return it to ArrayPool.</returns>
        public (byte[] pcmBuffer, int pcmLength) ProcessChunk(float[] input, int length, bool isFinal)
        {
            lock (_lock)
            {
                // 1. Apply high-pass filtering (in place)
                if (_useNative)
                    NativeAudio.HighPassProcess(input.AsSpan(0, length), _hpAlpha, _hpState);
                else
                    _hpFilter.Process(input.AsSpan(0, length));

                // 2. Crossfade processing
                int effectiveLen = length;

                // Blend with the previous chunk's tail
                if (_prevTailLen > 0 && effectiveLen > _crossFadeSamples)
                {
                    if (_useNative)
                    {
                        NativeAudio.CrossFade(input.AsSpan(0, _crossFadeSamples),
                            _prevChunkTail!.AsSpan(0, _crossFadeSamples), _crossFadeSamples);
                    }
                    else
                    {
                        for (int i = 0; i < _crossFadeSamples; i++)
                        {
                            float alpha = (float)i / _crossFadeSamples;
                            input[i] = _prevChunkTail![i] * (1f - alpha) + input[i] * alpha;
                        }
                    }
                }

                // Save the current chunk's tail for the next crossfade
                if (!isFinal && effectiveLen > _crossFadeSamples)
                {
                    Array.Copy(input, effectiveLen - _crossFadeSamples, _prevChunkTail!, 0, _crossFadeSamples);
                    _prevTailLen = _crossFadeSamples;
                    effectiveLen -= _crossFadeSamples; // Hide the tail; play it with the next chunk
                }
                else
                {
                    // Last chunk: apply fade-out and reset
                    if (isFinal && effectiveLen > _crossFadeSamples)
                    {
                        int start = effectiveLen - _crossFadeSamples;
                        if (_useNative)
                        {
                            NativeAudio.FadeOut(input.AsSpan(), start, _crossFadeSamples, _crossFadeSamples);
                        }
                        else
                        {
                            for (int i = 0; i < _crossFadeSamples; i++)
                            {
                                float fadeOut = 1f - (float)i / _crossFadeSamples;
                                input[start + i] *= fadeOut;
                            }
                        }
                    }
                    _prevTailLen = 0;
                }

                // 3. Convert to PCM16
                byte[] pcmBuf = ArrayPool<byte>.Shared.Rent(effectiveLen * 2);
                var pcmShorts = MemoryMarshal.Cast<byte, short>(pcmBuf.AsSpan(0, effectiveLen * 2));

                if (_useNative)
                {
                    NativeAudio.FloatToPcm16(input.AsSpan(0, effectiveLen), pcmShorts);
                }
                else
                {
                    for (int i = 0; i < effectiveLen; i++)
                    {
                        pcmShorts[i] = (short)(Math.Clamp(input[i], -1f, 1f) * 32767);
                    }
                }

                return (pcmBuf, effectiveLen * 2);
            }
        }

        public void Reset()
        {
            lock (_lock)
            {
                _prevTailLen = 0;
                if (_useNative)
                {
                    _hpState[0] = 0f;
                    _hpState[1] = 0f;
                }
            }
        }

        public void Dispose()
        {
            if (_prevChunkTail != null)
            {
                ArrayPool<float>.Shared.Return(_prevChunkTail);
                _prevChunkTail = null;
            }
        }
    }
}
