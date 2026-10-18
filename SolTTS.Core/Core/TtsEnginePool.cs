using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using SolTTS.Core.Config;

namespace SolTTS.Core.Core
{
    /// <summary>
    /// TTS engine pool managing concurrent inference instances.
    /// </summary>
    public class TtsEnginePool : IDisposable
    {
        private readonly TTSConfig _config;
        private readonly ConcurrentStack<ITtsEngine> _pool = new();
        private readonly SemaphoreSlim _semaphore;
        private ITtsEngine? _templateEngine;
        private bool _isInitialized = false;

        public TtsEnginePool(TTSConfig config)
        {
            _config = config;
            int capacity = Math.Max(1, config.PoolCapacity);
            _semaphore = new SemaphoreSlim(capacity, capacity);
        }

        /// <summary>
        /// Initialize the pool. Loads the first engine as a template.
        /// </summary>
        public async Task InitializeAsync()
        {
            if (_isInitialized) return;

            // Create and load the first engine (template)
            _templateEngine = CreateEngine(_config);
            await _templateEngine.LoadAsync(_config);

            _pool.Push(_templateEngine);

            // If capacity > 1, create more instances
            int extraCount = _config.PoolCapacity - 1;
            if (extraCount > 0)
            {
                // Multiple instances sharing sessions cannot run safely concurrently (session state is not reentrant),
                // so when using ggml hybrid mode (whose vocoder session runs on the GPU EP), force independent loading.
                bool usesGpuSession = _config.UseVocoderGgml && !_config.UseEngineV2;
                bool shareAllowed = _config.ReuseMemory && !usesGpuSession;
                if (_config.ReuseMemory && usesGpuSession)
                {
                    Console.WriteLine("[TtsEnginePool] ggml hybrid mode does not support shared sessions (sessions are not concurrency-safe); switched to independent loading.");
                }

                for (int i = 0; i < extraCount; i++)
                {
                    var engine = CreateEngine(_config);
                    if (shareAllowed)
                    {
                        // Reuse mode: share resources from the template engine
                        ShareResources(engine, _templateEngine);
                    }
                    else
                    {
                        // Independent mode: load separately (uses more memory)
                        await engine.LoadAsync(_config);
                    }
                    _pool.Push(engine);
                }
            }

            _isInitialized = true;
            Console.WriteLine($"[TtsEnginePool] Initialized with {_config.PoolCapacity} instances (ReuseMemory={_config.ReuseMemory})");
        }

        private static ITtsEngine CreateEngine(TTSConfig config)
        {
            return config.UseEngineV2 ? new TtsEngineV2() : new TtsEngineV1();
        }

        private static void ShareResources(ITtsEngine target, ITtsEngine source)
        {
            if (target is TtsEngineV1 v1Target && source is TtsEngineV1 v1Source)
            {
                v1Target.ShareResourcesFrom(v1Source);
            }
            else if (target is TtsEngineV2 v2Target && source is TtsEngineV2 v2Source)
            {
                v2Target.ShareResourcesFrom(v2Source);
            }
        }

        /// <summary>
        /// Lease an engine instance. Dispose the returned Lease when done.
        /// </summary>
        public async Task<EngineLease> LeaseAsync(CancellationToken cancellationToken = default)
        {
            if (!_isInitialized) await InitializeAsync();

            await _semaphore.WaitAsync(cancellationToken);

            if (_pool.TryPop(out var engine))
            {
                return new EngineLease(this, engine);
            }

            // In theory this should not happen because the semaphore controls availability
            _semaphore.Release();
            throw new InvalidOperationException("No engine available in the pool and none could be created.");
        }

        internal void Return(ITtsEngine engine)
        {
            _pool.Push(engine);
            _semaphore.Release();
        }

        public void Dispose()
        {
            while (_pool.TryPop(out var engine))
            {
                engine.Dispose();
            }
            _semaphore.Dispose();
        }

        public int SamplingRate => _templateEngine?.SamplingRate ?? 32000;
    }

    /// <summary>
    /// Engine lease ensuring the engine is returned to the pool after use.
    /// </summary>
    public class EngineLease : IDisposable
    {
        private readonly TtsEnginePool _pool;
        private ITtsEngine? _engine;

        public ITtsEngine Engine => _engine ?? throw new ObjectDisposedException(nameof(EngineLease));

        internal EngineLease(TtsEnginePool pool, ITtsEngine engine)
        {
            _pool = pool;
            _engine = engine;
        }

        public void Dispose()
        {
            if (_engine != null)
            {
                _pool.Return(_engine);
                _engine = null;
            }
        }
    }
}
