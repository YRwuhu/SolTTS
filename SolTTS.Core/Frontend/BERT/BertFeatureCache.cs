using System;
using System.Collections.Generic;
using System.Linq;

namespace SolTTS.Core.Frontend.BERT
{
    /// <summary>
    /// FIFO implementation of a BERT feature cache (thread-safe).
    ///
    /// Motivation: each `RobertaFeatureExtractor.Extract` call takes about 280–390 ms (of which about 278 ms is
    /// CPU control flow through the Loop node in the graph). The output is fully deterministic for the same text
    /// (and identical word2ph), and typical usage repeatedly synthesizes the same reference text/phrase, so caching pays off.
    ///
    /// Semantics:
    ///   - Caches the `Extract` return value `float[rows, hidden]`; **callers must treat it as read-only**
    ///     (both V1 and V2 currently Buffer.BlockCopy and discard, without mutation).
    ///   - The key includes word2ph so the same text under different segmentation/alignment cannot collide.
    ///   - Eviction policy: FIFO with a fixed capacity to prevent unbounded growth.
    /// </summary>
    public sealed class BertFeatureCache
    {
        private readonly int _capacity;
        private readonly Dictionary<string, float[,]> _map;
        private readonly Queue<string> _order = new Queue<string>();
        private readonly object _lock = new object();

        private long _hits;
        private long _misses;

        public BertFeatureCache(int capacity = 256)
        {
            if (capacity < 1) capacity = 1;
            _capacity = capacity;
            _map = new Dictionary<string, float[,]>(capacity);
        }

        /// <summary>Cache capacity.</summary>
        public int Capacity => _capacity;

        /// <summary>Number of hits (statistics only).</summary>
        public long Hits => System.Threading.Interlocked.Read(ref _hits);

        /// <summary>Number of misses (statistics only).</summary>
        public long Misses => System.Threading.Interlocked.Read(ref _misses);

        /// <summary>Build a cache key. word2ph is part of the key so different alignments do not reuse the same result.</summary>
        public static string MakeKey(string text, int[] word2ph)
        {
            // Use separators that cannot appear in normal text to avoid concatenation ambiguity
            return text + "\u0001" + string.Join("\u0002", word2ph ?? Array.Empty<int>());
        }

        /// <summary>Try to get a cached entry. Returns true on hit; value is a shared read-only array.</summary>
        public bool TryGet(string key, out float[,] value)
        {
            lock (_lock)
            {
                if (_map.TryGetValue(key, out value))
                {
                    System.Threading.Interlocked.Increment(ref _hits);
                    return true;
                }
            }
            value = null!;
            System.Threading.Interlocked.Increment(ref _misses);
            return false;
        }

        /// <summary>Write to the cache (FIFO eviction when capacity is exceeded).</summary>
        public void Set(string key, float[,] value)
        {
            lock (_lock)
            {
                if (_map.ContainsKey(key)) return;
                if (_map.Count >= _capacity)
                {
                    // FIFO: evict the oldest key
                    while (_order.Count > 0)
                    {
                        var oldest = _order.Dequeue();
                        if (_map.Remove(oldest)) break;
                    }
                }
                _map[key] = value;
                _order.Enqueue(key);
            }
        }

        /// <summary>Clear the cache (e.g., after a configuration reload).</summary>
        public void Clear()
        {
            lock (_lock)
            {
                _map.Clear();
                _order.Clear();
            }
        }
    }
}
