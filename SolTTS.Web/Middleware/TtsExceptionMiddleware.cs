using System.Net;
using System.Text.Json;

namespace SolTTS.Web.Middleware
{
    /// <summary>
    /// TTS exception-handling middleware; maps language-restriction and similar exceptions from the core layer to HTTP 400.
    /// </summary>
    public class TtsExceptionMiddleware
    {
        private readonly RequestDelegate _next;
        private readonly ILogger<TtsExceptionMiddleware> _logger;

        public TtsExceptionMiddleware(RequestDelegate next, ILogger<TtsExceptionMiddleware> logger)
        {
            _next = next;
            _logger = logger;
        }

        public async Task InvokeAsync(HttpContext context)
        {
            try
            {
                await _next(context);
            }
            catch (InvalidOperationException ex)
            {
                // Language-restriction exception thrown by MixedLanguageG2P in the core layer
                _logger.LogWarning($"[SolTTS.Web] InvalidOperation: {ex.Message}");
                await HandleExceptionAsync(context, ex, HttpStatusCode.BadRequest);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[SolTTS.Web] Unhandled Exception");
                await HandleExceptionAsync(context, ex, HttpStatusCode.InternalServerError);
            }
        }

        private async Task HandleExceptionAsync(HttpContext context, Exception exception, HttpStatusCode code)
        {
            // Once a streaming response has started writing (headers sent), the status code and Content-Type can no longer change.
            // In that case we can only abort the connection so the client sees a truncated response, rather than throwing a secondary "Headers are read-only" exception.
            if (context.Response.HasStarted)
            {
                _logger.LogWarning(
                    $"[SolTTS.Web] Response already started; cannot write error body ({code}): {exception.Message}. Aborting the connection.");
                context.Abort();
                return;
            }

            context.Response.ContentType = "application/json";
            context.Response.StatusCode = (int)code;

            var result = JsonSerializer.Serialize(new
            {
                success = false,
                message = exception.Message,
                type = exception.GetType().Name
            });

            await context.Response.WriteAsync(result);
        }
    }
}
