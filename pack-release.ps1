# SolTTS resource and release packaging script
# Set console output encoding to UTF8 to prevent garbled characters
chcp 65001 >$null
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Version = "v1.2.3"
$ReleaseDir = "releases"

if (!(Test-Path $ReleaseDir)) {
    New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null
}

$7z_exe = "7z"
if (!(Get-Command $7z_exe -ErrorAction SilentlyContinue)) {
    if (Test-Path "$env:ProgramFiles\7-Zip\7z.exe") {
        $7z_exe = "$env:ProgramFiles\7-Zip\7z.exe"
    } else {
        $7z_exe = $null
    }
}

function Compress-WithProgress {
    param([string]$SourcePath, [string]$DestinationPath)
    if ($7z_exe) {
        Write-Host "  👉 Compressing at high speed with 7-Zip..." -ForegroundColor DarkCyan
        & $7z_exe a -tzip "$DestinationPath" "$SourcePath" -bso0 -bsp1
    } else {
        Write-Host "  ⚠️ 7-Zip not found; using built-in compression (slower, no progress bar — please wait)..." -ForegroundColor DarkGray
        Compress-Archive -Path $SourcePath -DestinationPath $DestinationPath -Force
    }
}

Write-Host "📦 Packaging SolTTS $Version release files..." -ForegroundColor Cyan

# Package the Windows all-in-one bundle
if (Test-Path "publish") {
    $winZip = "$ReleaseDir/SolTTS-$Version-win64.zip"
    if (Test-Path $winZip) { Remove-Item $winZip -Force }
    Write-Host "`nCompressing Windows bundle -> $winZip ..." -ForegroundColor Yellow
    Compress-WithProgress -SourcePath "publish\*" -DestinationPath $winZip
    Write-Host "✅ Windows bundle packaging complete!" -ForegroundColor Green
} else {
    Write-Host "`n⚠️ publish directory not found; skipping Windows packaging." -ForegroundColor Yellow
}

# Package the standalone resource bundle (resources-minimal)
if (Test-Path "resources-minimal") {
    $resZip = "$ReleaseDir/SolTTS-resources-minimal-$Version.zip"
    if (Test-Path $resZip) { Remove-Item $resZip -Force }
    Write-Host "`nCompressing standalone resource bundle (resources-minimal) -> $resZip ..." -ForegroundColor Yellow
    Compress-WithProgress -SourcePath "resources-minimal" -DestinationPath $resZip
    Write-Host "✅ Standalone resource bundle packaging complete!" -ForegroundColor Green
} else {
    Write-Host "`n⚠️ resources-minimal directory not found; skipping core resource packaging." -ForegroundColor Yellow
}

Write-Host "`n🎉 All packaging jobs complete. Output: $(Resolve-Path $ReleaseDir)" -ForegroundColor Cyan
