using SolTTS.Core.Core;
using Microsoft.Extensions.Hosting;

namespace SolTTS.Web
{
    public class TtsWarmupService : IHostedService
    {
        private readonly SolTtsSdk _sdk;

        public TtsWarmupService(SolTtsSdk sdk)
        {
            _sdk = sdk;
        }

        public async Task StartAsync(CancellationToken cancellationToken)
        {
            Console.WriteLine("[SolTTS] Starting SDK Initialization...");
            await _sdk.InitializeAsync();
            Console.WriteLine("[SolTTS] SDK Initialized Successfully.");
        }

        public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
    }
}
