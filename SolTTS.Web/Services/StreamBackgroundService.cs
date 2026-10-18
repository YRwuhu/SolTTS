using SolTTS.Core.Core;

namespace SolTTS.Web.Services
{
    public class StreamBackgroundService : BackgroundService
    {
        private readonly StreamSessionManager _sessionManager;
        private readonly SolTtsSdk _sdk;
        private readonly ILogger<StreamBackgroundService> _logger;

        public StreamBackgroundService(StreamSessionManager sessionManager, SolTtsSdk sdk, ILogger<StreamBackgroundService> logger)
        {
            _sessionManager = sessionManager;
            _sdk = sdk;
            _logger = logger;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    await Task.Delay(1000, stoppingToken);

                    _sessionManager.CleanupOldSessions(TimeSpan.FromMinutes(5));
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error in StreamBackgroundService");
                }
            }
        }
    }
}
