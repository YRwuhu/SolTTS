using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace SolTTS.Core.Core
{
    /// <summary>
    /// Managed bindings for the native real-time audio primitives (native/sol_audio, Rust cdylib).
    ///
    /// Design constraints:
    /// - Optional dependency: when the DLL is missing, version-mismatched or unsupported on the platform,
    ///   <see cref="IsAvailable"/> is false and callers automatically fall back to the managed implementation.
    /// - Buffers are caller-provided (pooled); the native side neither allocates nor holds managed memory.
    /// </summary>
    public static unsafe class NativeAudio
    {
        public const string LibraryName = "sol_audio";

        private const uint ExpectedVersion = (0u << 16) | (1u << 8) | 0u;

        /// <summary>Whether the DLL exists and the version handshake succeeded. Probed on first access, then cached.</summary>
        public static bool IsAvailable => Available.Value;

        /// <summary>Probe failure reason (null on success).</summary>
        public static string? UnavailableReason => Reason.Value;

        /// <summary>Native library version, encoded as major&lt;&lt;16 | minor&lt;&lt;8 | patch.</summary>
        public static uint Version => NativeVersion.Value;

        private static readonly Lazy<bool> Available = new(Probe);
        private static readonly Lazy<string?> Reason = new(() => { _ = Available.Value; return _reason; });
        private static readonly Lazy<uint> NativeVersion = new(() => { _ = Available.Value; return _version; });

        private static string? _reason;
        private static uint _version;

        private static bool Probe()
        {
            try
            {
                _version = NativeMethods.sol_audio_version();
                if (_version != ExpectedVersion)
                {
                    _reason = $"version 0x{_version:X} != expected 0x{ExpectedVersion:X}";
                    return false;
                }
                return true;
            }
            catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
            {
                _reason = ex.Message;
                return false;
            }
        }

        public static IntPtr RingCreate(int capacityBytes)
        {
            EnsureAvailable();
            if (capacityBytes <= 0) throw new ArgumentOutOfRangeException(nameof(capacityBytes));
            var handle = NativeMethods.sol_ring_create((nuint)capacityBytes);
            if (handle == IntPtr.Zero) throw new OutOfMemoryException("sol_ring_create failed");
            return handle;
        }

        public static void RingDestroy(IntPtr handle)
        {
            if (!IsAvailable || handle == IntPtr.Zero) return;
            NativeMethods.sol_ring_destroy(handle);
        }

        /// <summary>Write bytes; returns the actual count written (0 when the buffer is full, discarding the whole block).</summary>
        public static int RingWrite(IntPtr handle, ReadOnlySpan<byte> src)
        {
            EnsureAvailable();
            if (handle == IntPtr.Zero) throw new ArgumentException("null ring handle", nameof(handle));
            fixed (byte* p = src)
                return checked((int)NativeMethods.sol_ring_write(handle, p, (nuint)src.Length));
        }

        /// <summary>Read bytes; returns the actual count read.</summary>
        public static int RingRead(IntPtr handle, Span<byte> dst)
        {
            EnsureAvailable();
            if (handle == IntPtr.Zero) throw new ArgumentException("null ring handle", nameof(handle));
            fixed (byte* p = dst)
                return checked((int)NativeMethods.sol_ring_read(handle, p, (nuint)dst.Length));
        }

        public static int RingAvailable(IntPtr handle)
        {
            if (!IsAvailable || handle == IntPtr.Zero) return 0;
            return checked((int)NativeMethods.sol_ring_available(handle));
        }

        public static void RingClear(IntPtr handle)
        {
            if (!IsAvailable || handle == IntPtr.Zero) return;
            NativeMethods.sol_ring_clear(handle);
        }

        public static float HighPassAlpha(float sampleRate, float cutoffHz)
        {
            EnsureAvailable();
            return NativeMethods.sol_hpf_alpha(sampleRate, cutoffHz);
        }

        /// <summary>In-place high-pass filter; state is [prevInput, prevOutput], updated in place.</summary>
        public static void HighPassProcess(Span<float> samples, float alpha, Span<float> state)
        {
            EnsureAvailable();
            if (state.Length < 2) throw new ArgumentException("state needs 2 floats", nameof(state));
            fixed (float* p = samples)
            fixed (float* s = state)
                Check(NativeMethods.sol_hpf_process(p, (nuint)samples.Length, alpha, s), "hpf");
        }

        /// <summary>
        /// Crossfade: dst fades in from src, dst[i] = src[i]*(1-a) + dst[i]*a, a = i/denominator.
        /// Corresponds to blending prevTail (fading out) with input (fading in) in AudioPipeline.
        /// </summary>
        public static void CrossFade(Span<float> dst, ReadOnlySpan<float> src, int denominator)
        {
            EnsureAvailable();
            if (src.Length < dst.Length) throw new ArgumentException("src shorter than dst", nameof(src));
            fixed (float* d = dst)
            fixed (float* s = src)
                Check(NativeMethods.sol_crossfade(d, s, (nuint)dst.Length, (nuint)denominator), "crossfade");
        }

        public static void FadeOut(Span<float> samples, int start, int length, int denominator)
        {
            EnsureAvailable();
            if (length <= 0) return;
            fixed (float* p = samples)
                Check(NativeMethods.sol_fade_out(p, (nuint)start, (nuint)length, (nuint)denominator), "fade_out");
        }

        public static void FloatToPcm16(ReadOnlySpan<float> src, Span<short> dst)
        {
            EnsureAvailable();
            if (dst.Length < src.Length) throw new ArgumentException("dst too short", nameof(dst));
            fixed (float* s = src)
            fixed (short* d = dst)
                Check(NativeMethods.sol_f32_to_i16(s, d, (nuint)src.Length), "f32_to_i16");
        }

        private static void Check(int code, string op)
        {
            if (code != 0) throw new InvalidOperationException($"native sol_audio {op} failed: {code}");
        }

        private static void EnsureAvailable()
        {
            if (!IsAvailable)
                throw new InvalidOperationException($"native sol_audio unavailable: {UnavailableReason}");
        }

        private static class NativeMethods
        {
            [DllImport(LibraryName, EntryPoint = "sol_audio_version", CallingConvention = CallingConvention.Cdecl)]
            internal static extern uint sol_audio_version();

            [DllImport(LibraryName, EntryPoint = "sol_ring_create", CallingConvention = CallingConvention.Cdecl)]
            internal static extern IntPtr sol_ring_create(nuint capacityBytes);

            [DllImport(LibraryName, EntryPoint = "sol_ring_destroy", CallingConvention = CallingConvention.Cdecl)]
            internal static extern void sol_ring_destroy(IntPtr ring);

            [DllImport(LibraryName, EntryPoint = "sol_ring_write", CallingConvention = CallingConvention.Cdecl)]
            internal static extern nuint sol_ring_write(IntPtr ring, byte* src, nuint len);

            [DllImport(LibraryName, EntryPoint = "sol_ring_read", CallingConvention = CallingConvention.Cdecl)]
            internal static extern nuint sol_ring_read(IntPtr ring, byte* dst, nuint len);

            [DllImport(LibraryName, EntryPoint = "sol_ring_available", CallingConvention = CallingConvention.Cdecl)]
            internal static extern nuint sol_ring_available(IntPtr ring);

            [DllImport(LibraryName, EntryPoint = "sol_ring_clear", CallingConvention = CallingConvention.Cdecl)]
            internal static extern void sol_ring_clear(IntPtr ring);

            [DllImport(LibraryName, EntryPoint = "sol_hpf_alpha", CallingConvention = CallingConvention.Cdecl)]
            internal static extern float sol_hpf_alpha(float sampleRate, float cutoffHz);

            [DllImport(LibraryName, EntryPoint = "sol_hpf_process", CallingConvention = CallingConvention.Cdecl)]
            internal static extern int sol_hpf_process(float* samples, nuint len, float alpha, float* state);

            [DllImport(LibraryName, EntryPoint = "sol_crossfade", CallingConvention = CallingConvention.Cdecl)]
            internal static extern int sol_crossfade(float* dst, float* src, nuint n, nuint denom);

            [DllImport(LibraryName, EntryPoint = "sol_fade_out", CallingConvention = CallingConvention.Cdecl)]
            internal static extern int sol_fade_out(float* samples, nuint start, nuint n, nuint denom);

            [DllImport(LibraryName, EntryPoint = "sol_f32_to_i16", CallingConvention = CallingConvention.Cdecl)]
            internal static extern int sol_f32_to_i16(float* src, short* dst, nuint len);
        }
    }
}
