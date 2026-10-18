using SolTTS.Core.Core;
using SolTTS.Core.Config;
using SolTTS.Web;
using SolTTS.Web.Services;
using Scalar.AspNetCore;

var builder = WebApplication.CreateBuilder(args);

// Parse config path from args
string? configPath = null;
for (int i = 0; i < args.Length; i++)
{
    string arg = args[i];
    if (arg.StartsWith("-"))
    {
        string flag = arg;
        string? value = null;

        if (arg.Contains('='))
        {
            var parts = arg.Split('=', 2);
            flag = parts[0];
            value = parts[1];
        }

        switch (flag.ToLower())
        {
            case "-c":
            case "--config":
                if (value == null)
                {
                    if (i + 1 < args.Length && !args[i + 1].StartsWith("-")) value = args[++i];
                    else
                    {
                        Console.WriteLine("Error: Missing value for config flag.");
                        return;
                    }
                }
                configPath = value;
                break;
                // Ignore other unknown arguments; they may be ASP.NET Core options (e.g., --urls, --environment)
        }
    }
}

var config = configPath != null ? TTSConfig.LoadOrCreate(configPath) : TTSConfig.Load();
Console.WriteLine($"[SolTTS.Web] Loading config from: {TTSConfig.LoadedPath}");

// Add services to the container.
builder.Services.AddControllers()
    .AddJsonOptions(options =>
    {
        options.JsonSerializerOptions.PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase;
        options.JsonSerializerOptions.WriteIndented = true;
    });

// Register StreamSessionManager as Singleton
builder.Services.AddSingleton<StreamSessionManager>();

// Register SolTTS.Core SDK as Singleton
builder.Services.AddSingleton<SolTtsSdk>(sp =>
{
    var sdk = new SolTtsSdk(config);
    // Async initialization (runs in the background or is awaited by app startup logic)
    // Note: In ASP.NET Core, heavy async initialization is typically done in a Startup Filter or HostedService
    // For simplicity, we ensure initialization completes before the first request
    return sdk;
});

// Use a HostedService for warm-up initialization
builder.Services.AddHostedService<TtsWarmupService>();
builder.Services.AddHostedService<StreamBackgroundService>();

// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();

var app = builder.Build();

// Configure the HTTP request pipeline.
app.UseMiddleware<SolTTS.Web.Middleware.TtsExceptionMiddleware>();
app.UseDefaultFiles();
app.UseStaticFiles();

app.MapOpenApi();
app.MapScalarApiReference(); // Mapping Scalar at /scalar
app.MapControllers();

app.Run();
