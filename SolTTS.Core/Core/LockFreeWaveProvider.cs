using System;
using System.Runtime.InteropServices;
using System.Threading;
using NAudio.Wave;

namespace SolTTS.Core.Core
{
    /// <summary>
    /// A lock-free FIFO buffer for audio samples, supporting single-producer single-consumer.
    /// Uses Interlocked primitives to ensure thread safety without locks.
    /// </summary>
    public class LockFreeWaveProvider : IWaveProvider, IDisposable
    {
        private readonly byte[] _buffer;
        private readonly int _bufferSize;
        private readonly WaveFormat _waveFormat;
        
        private long _writePosition;
        private long _readPosition;
        private volatile bool _readFully = true; // If true, return silence when buffer is empty instead of returning 0

        /// <summary>Native SPSC ring buffer handle (0 means the managed implementation is used).</summary>
        private readonly IntPtr _nativeRing;
        private readonly bool _useNative;

        public LockFreeWaveProvider(WaveFormat format, int bufferSize = 65536 * 32, bool useNative = false)
        {
            _waveFormat = format;
            _bufferSize = bufferSize;
            _useNative = useNative && NativeAudio.IsAvailable;
            if (_useNative)
            {
                // Native buffer size is in bytes; it is internally rounded up to a power of two
                _nativeRing = NativeAudio.RingCreate(bufferSize);
                _buffer = Array.Empty<byte>();
            }
            else
            {
                _buffer = new byte[bufferSize];
            }
        }

        public WaveFormat WaveFormat => _waveFormat;
        
        public bool ReadFully 
        { 
            get => _readFully; 
            set => _readFully = value; 
        }
        
        // Track silence bytes padded due to buffer underrun, used to calibrate playback time
        public long PaddingBytes => Volatile.Read(ref _paddingBytes);
        private long _paddingBytes;

        public int BufferedBytes
        {
            get
            {
                if (_useNative) return NativeAudio.RingAvailable(_nativeRing);
                return (int)(Volatile.Read(ref _writePosition) - Volatile.Read(ref _readPosition));
            }
        }

        /// <summary>Total buffer capacity (bytes), used by the producer to check for free space.</summary>
        public int Capacity => _bufferSize;

        /// <summary>
        /// Adds samples to the buffer. This must only be called by the producer thread.
        /// </summary>
        public void AddSamples(byte[] buffer, int offset, int count)
        {
            if (_useNative)
            {
                NativeAudio.RingWrite(_nativeRing, buffer.AsSpan(offset, count));
                return;
            }

            long writePos = _writePosition; 
            long readPos = Volatile.Read(ref _readPosition);

            if (writePos - readPos + count > _bufferSize) return;

            int writeOffset = (int)(writePos % _bufferSize);
            int firstChunk = Math.Min(count, _bufferSize - writeOffset);
            
            // Optimization: Use Span for safe and fast memory copy
            var srcSpan = buffer.AsSpan(offset, count);
            var bufferSpan = _buffer.AsSpan();
            
            srcSpan.Slice(0, firstChunk).CopyTo(bufferSpan.Slice(writeOffset));
            if (firstChunk < count)
            {
                srcSpan.Slice(firstChunk).CopyTo(bufferSpan);
            }

            // Publish Write Position with memory barrier
            Volatile.Write(ref _writePosition, writePos + count);
        }

        /// <summary>
        /// Reads samples from the buffer. This must only be called by the consumer thread.
        /// </summary>
        public int Read(byte[] buffer, int offset, int count)
        {
            int toRead;
            if (_useNative)
            {
                toRead = NativeAudio.RingRead(_nativeRing, buffer.AsSpan(offset, count));
            }
            else
            {
                toRead = ReadManaged(buffer, offset, count);
            }

            // Fill the rest with silence if ReadFully is true
            if (toRead < count && _readFully)
            {
                int padding = count - toRead;
                buffer.AsSpan(offset + toRead, padding).Clear();
                Volatile.Write(ref _paddingBytes, padding);
                return count;
            }

            return toRead;
        }

        private int ReadManaged(byte[] buffer, int offset, int count)
        {
            long writePos = Volatile.Read(ref _writePosition);
            long readPos = _readPosition;

            int available = (int)(writePos - readPos);
            int toRead = Math.Min(available, count);

            if (toRead > 0)
            {
                int readOffset = (int)(readPos % _bufferSize);
                int firstChunk = Math.Min(toRead, _bufferSize - readOffset);

                // Optimization: Use Span for safe and fast memory copy
                var destSpan = buffer.AsSpan(offset, toRead);
                var bufferSpan = _buffer.AsSpan();

                bufferSpan.Slice(readOffset, firstChunk).CopyTo(destSpan.Slice(0, firstChunk));
                if (firstChunk < toRead)
                {
                    bufferSpan.Slice(0, toRead - firstChunk).CopyTo(destSpan.Slice(firstChunk));
                }
                
                Volatile.Write(ref _readPosition, readPos + toRead);
            }

            return toRead;
        }
        
        /// <summary>
        /// Resets the buffer
        /// </summary>
        public void Clear()
        {
            if (_useNative)
            {
                NativeAudio.RingClear(_nativeRing);
                return;
            }

            // Resetting is tricky in lock-free if threads are active. Use with caution or when stopped.
            _writePosition = 0;
            _readPosition = 0;
            _buffer.AsSpan().Clear();
        }
        /// <summary>Whether the native ring buffer is currently in use (diagnostics only).</summary>
        public bool IsNative => _useNative;

        public void Dispose()
        {
            if (_useNative)
                NativeAudio.RingDestroy(_nativeRing);
        }
    }
}
