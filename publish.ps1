# SolTTS unified publish script
# Set console output encoding
chcp 65001 >$null
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Output test
Write-Host "--- SolTTS build and publish tool ---" -ForegroundColor Cyan

$PublishDir = "publish"
$Runtime = "win-x64" # Change as needed, e.g., win-arm64

if (Test-Path $PublishDir) {
    Write-Host "Cleaning the old publish directory..." -ForegroundColor Cyan
    Remove-Item -Path $PublishDir -Force -Recurse
}

Write-Host "🚀 Publishing SolTTS.Web (sol-server)..." -ForegroundColor Green
dotnet publish SolTTS.Web/SolTTS.Web.csproj -c Release -r $Runtime --self-contained true -o $PublishDir /p:PublishSingleFile=false /p:AllowMissingPrunePackageData=true

Write-Host "🚀 Publishing SolTTS.CLI (sol-cli)..." -ForegroundColor Green
dotnet publish SolTTS.CLI/SolTTS.CLI.csproj -c Release -r $Runtime --self-contained true -o $PublishDir /p:PublishSingleFile=false /p:AllowMissingPrunePackageData=true

# Remove the old config.json (if left over)
if (Test-Path "$PublishDir/config.json") {
    Remove-Item "$PublishDir/config.json" -Force
}

# Copy configuration templates
if (Test-Path "config.template.yaml") {
    Write-Host "📦 Copying configuration files..." -ForegroundColor Yellow
    Copy-Item "config.template.yaml" -Destination "$PublishDir/config.template.yaml"
    Copy-Item "config.template.yaml" -Destination "$PublishDir/config.yaml"
}

# Copy the minimal resource directory
if (Test-Path "resources-minimal") {
    Write-Host "📦 Copying resources (minimal)..." -ForegroundColor Yellow
    Copy-Item -Path "resources-minimal" -Destination "$PublishDir/resources" -Recurse -Force
}

# Copy the converter tooling only, matching publish-linux.sh and publish-mac.sh.
# Copying all of `tools` pulled in the probe projects and their bin/obj (~1 GB of
# experiment binaries) into the release.
if (Test-Path "tools/converter") {
    Write-Host "📦 Copying converter tools..." -ForegroundColor Yellow
    $toolsDst = Join-Path $PublishDir "tools/converter"
    New-Item -ItemType Directory -Force -Path $toolsDst | Out-Null
    Copy-Item -Path "tools/converter/*.py" -Destination $toolsDst -Force
    if (Test-Path "tools/converter/templates") {
        Copy-Item -Path "tools/converter/templates" -Destination $toolsDst -Recurse -Force
    }
    if (Test-Path "tools/converter/runtime") {
        Copy-Item -Path "tools/converter/runtime" -Destination $toolsDst -Recurse -Force
    }
}

# Native audio primitives (optional): sol_audio.dll is copied by the csproj None items during publish.
# This is only a verification hint to avoid silent fallback when the release uses UseNativeAudio=true.
$nativeDll = Join-Path $PublishDir "sol_audio.dll"
if (Test-Path $nativeDll) {
    Write-Host "✅ Native audio library included: sol_audio.dll" -ForegroundColor Green
} else {
    Write-Host "⚠️  sol_audio.dll not found (UseNativeAudio will fall back to the managed implementation)." -ForegroundColor Yellow
    Write-Host "    For the native implementation, run: cd native/sol_audio && cargo build --release" -ForegroundColor Yellow
}

Write-Host "`n✅ Publish complete! All files are in: $(Resolve-Path $PublishDir)" -ForegroundColor Green
Write-Host "Run hints:"
Write-Host "  - Run the Web service: ./$PublishDir/sol-server.exe"
Write-Host "  - Run the CLI: ./$PublishDir/sol-cli.exe"
