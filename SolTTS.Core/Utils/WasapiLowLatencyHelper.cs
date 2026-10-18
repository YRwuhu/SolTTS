using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using NAudio.CoreAudioApi;
using NAudio.Wave;

namespace SolTTS.Core.Utils
{
    /// <summary>
    /// WASAPI low-latency helper (audio anchor anchoring).
    /// Principle: create a tiny-period IAudioClient3 stream to force the Windows audio engine into low-latency scheduling mode (typically 10ms down to 2.67ms).
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class WasapiLowLatencyHelper : IDisposable
    {
        private object? _audioClient;
        private bool _isStarted;

        // IAudioClient3 interface ID
        private static readonly Guid IID_IAudioClient3 = new Guid("7ED4EE07-8E67-4CD4-8C1A-2B7A5987AD42");

        [ComImport]
        [Guid("7ED4EE07-8E67-4CD4-8C1A-2B7A5987AD42")]
        [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        internal interface IAudioClient3
        {
            [PreserveSig] int Initialize(int shareMode, int streamFlags, long hnsBufferDuration, long hnsPeriodicity, [In] WaveFormat pformat, [In] ref Guid audioSessionGuid);
            [PreserveSig] int GetBufferSize(out uint bufferSize);
            [PreserveSig] int GetStreamLatency(out long latency);
            [PreserveSig] int GetCurrentPadding(out uint currentPadding);
            [PreserveSig] int IsFormatSupported(int shareMode, [In] WaveFormat pFormat, out IntPtr ppClosestMatch);
            [PreserveSig] int GetMixFormat(out IntPtr ppDeviceFormat);
            [PreserveSig] int GetDevicePeriod(out long defaultPeriod, out long minimumPeriod);
            [PreserveSig] int Start();
            [PreserveSig] int Stop();
            [PreserveSig] int Reset();
            [PreserveSig] int SetEventHandle(IntPtr eventHandle);
            [PreserveSig] int GetService([In] ref Guid interfaceId, [MarshalAs(UnmanagedType.IUnknown)] out object interfacePointer);

            // IAudioClient3 specific methods
            [PreserveSig] int GetSharedModeEnginePeriod([In] WaveFormat pFormat, out uint pDefaultPeriodInFrames, out uint pFundamentalPeriodInFrames, out uint pMinPeriodInFrames, out uint pMaxPeriodInFrames);
            [PreserveSig] int GetCurrentSharedModeEnginePeriod(out IntPtr ppFormat, out uint pCurrentPeriodInFrames);
            [PreserveSig] int InitializeSharedAudioStream(uint StreamFlags, uint PeriodInFrames, [In] WaveFormat pFormat, [In] ref Guid AudioSessionGuid);
        }

        public void EnableLowLatency()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return;
            }
            if (_isStarted) return;

            try
            {
                Console.WriteLine("[Wasapi] Attempting to enable system-level low-latency mode (Audio Anchor)...");

                // 1. Get the default render device
                using var enumerator = new MMDeviceEnumerator();
                using var device = enumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);

                // 2. Activate IAudioClient3
                // NAudio's MMDevice does not expose Activate directly.
                // Obtain IAudioClient via device.AudioClient, then QueryInterface for IAudioClient3
                var audioClient = device.AudioClient;
                var iid = IID_IAudioClient3;
                Marshal.QueryInterface(Marshal.GetIUnknownForObject(audioClient), in iid, out var comPtr);

                if (comPtr == IntPtr.Zero)
                {
                    Console.WriteLine("   ⚠️ Failed to obtain the IAudioClient3 interface (system may be older than Win10)");
                    return;
                }

                var client3 = (IAudioClient3)Marshal.GetObjectForIUnknown(comPtr);
                _audioClient = client3;

                // 3. Get the mix format (engine native format)
                client3.GetMixFormat(out var formatPtr);
                var format = Marshal.PtrToStructure<WaveFormat>(formatPtr);
                if (format == null) return;

                // 4. Query the minimum period supported by the driver
                client3.GetSharedModeEnginePeriod(format, out _, out _, out uint minPeriod, out _);
                Console.WriteLine($"   Hardware-supported minimum period: {minPeriod} frames ({(minPeriod * 1000.0 / format.SampleRate):F2} ms)");

                // 5. Initialize the low-latency stream
                // AUDCLNT_STREAMFLAGS_EVENTCALLBACK (0x00040000)
                uint flags = 0x00040000;
                Guid sessionGuid = Guid.Empty;
                int hr = client3.InitializeSharedAudioStream(flags, minPeriod, format, ref sessionGuid);

                if (hr < 0)
                {
                    Console.WriteLine($"   ❌ Failed to initialize low-latency stream: 0x{hr:X8}");
                    return;
                }

                // An EventHandle must be associated before Start
                var hEvent = CreateEvent(IntPtr.Zero, false, false, null);
                client3.SetEventHandle(hEvent);

                // 6. Start the anchor stream
                client3.Start();
                _isStarted = true;

                Console.WriteLine("   ✅ Low-latency anchor started; the Windows audio engine is now in high-performance mode.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ⚠️ Failed to enable low latency: {ex.Message}");
            }
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern IntPtr CreateEvent(IntPtr lpEventAttributes, bool bManualReset, bool bInitialState, string? lpName);

        public void Dispose()
        {
            if (_audioClient is IAudioClient3 client)
            {
                client.Stop();
                Marshal.ReleaseComObject(client);
            }
            _audioClient = null;
        }
    }
}
