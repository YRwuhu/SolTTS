using System.Diagnostics;
using System.Threading;

namespace SolTTS.Core.Core
{
    /// <summary>
    /// Lightweight per-stage timing statistics (disabled by default; enabled via SOLTTS_PERF_STATS=1 or by the assembly).
    /// Only accumulates counters, with no allocations or lock contention hot spots; used to locate T2S / vocoder / frontend time shares.
    /// </summary>
    public static class PerfStats
    {
        public static bool Enabled { get; set; } =
            Environment.GetEnvironmentVariable("SOLTTS_PERF_STATS") == "1";

        private static long _t2sTicks, _t2sCalls, _t2sTextPhones, _t2sStageSteps;
        private static long _vocTicks, _vocCalls, _vocSamples;
        private static long _frontendTicks, _frontendCalls;
        private static readonly System.Collections.Concurrent.ConcurrentQueue<(int Index, int Tokens, int Samples)> _sentences = new();

        /// <summary>Record one sentence's output, used to verify sentence order and length after parallel decoding.</summary>
        public static void AddSentence(int index, int tokens, int samples)
        {
            if (!Enabled) return;
            _sentences.Enqueue((index, tokens, samples));
        }

        /// <summary>Drain and clear the per-sentence statistics (in enqueue order).</summary>
        public static List<(int Index, int Tokens, int Samples)> DrainSentences()
        {
            var list = new List<(int, int, int)>();
            while (_sentences.TryDequeue(out var item)) list.Add(item);
            return list;
        }

        public static void AddT2S(Stopwatch sw, int textPhones, int stageSteps)
        {
            if (!Enabled) return;
            Interlocked.Add(ref _t2sTicks, sw.ElapsedTicks);
            Interlocked.Increment(ref _t2sCalls);
            Interlocked.Add(ref _t2sTextPhones, textPhones);
            Interlocked.Add(ref _t2sStageSteps, stageSteps);
        }

        public static void AddVocoder(Stopwatch sw, int samples)
        {
            if (!Enabled) return;
            Interlocked.Add(ref _vocTicks, sw.ElapsedTicks);
            Interlocked.Increment(ref _vocCalls);
            Interlocked.Add(ref _vocSamples, samples);
        }

        public static void AddFrontend(Stopwatch sw)
        {
            if (!Enabled) return;
            Interlocked.Add(ref _frontendTicks, sw.ElapsedTicks);
            Interlocked.Increment(ref _frontendCalls);
        }

        public static (double T2sMs, long T2sCalls, double T2sPhonesPerCall, double T2sStageStepsPerCall,
                       double VocMs, long VocCalls, double VocSamplesPerCall,
                       double FrontendMs, long FrontendCalls) Snapshot()
        {
            long t2s = Interlocked.Read(ref _t2sTicks);
            long t2sCalls = Interlocked.Read(ref _t2sCalls);
            long phones = Interlocked.Read(ref _t2sTextPhones);
            long steps = Interlocked.Read(ref _t2sStageSteps);
            long voc = Interlocked.Read(ref _vocTicks);
            long vocCalls = Interlocked.Read(ref _vocCalls);
            long samples = Interlocked.Read(ref _vocSamples);
            long fe = Interlocked.Read(ref _frontendTicks);
            long feCalls = Interlocked.Read(ref _frontendCalls);

            double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;
            return (
                Ms(t2s), t2sCalls, t2sCalls == 0 ? 0 : (double)phones / t2sCalls, t2sCalls == 0 ? 0 : (double)steps / t2sCalls,
                Ms(voc), vocCalls, vocCalls == 0 ? 0 : (double)samples / vocCalls,
                Ms(fe), feCalls);
        }

        public static void Reset()
        {
            Interlocked.Exchange(ref _t2sTicks, 0);
            Interlocked.Exchange(ref _t2sCalls, 0);
            Interlocked.Exchange(ref _t2sTextPhones, 0);
            Interlocked.Exchange(ref _t2sStageSteps, 0);
            Interlocked.Exchange(ref _vocTicks, 0);
            Interlocked.Exchange(ref _vocCalls, 0);
            Interlocked.Exchange(ref _vocSamples, 0);
            Interlocked.Exchange(ref _frontendTicks, 0);
            Interlocked.Exchange(ref _frontendCalls, 0);
        }
    }
}
