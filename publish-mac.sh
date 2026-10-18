#!/bin/bash

# ==========================================
# SolTTS unified publish script (macOS)
# Usage:
#   ./publish-mac.sh                # Auto-detect host architecture (Apple Silicon -> arm64, Intel -> x64)
#   ./publish-mac.sh arm64          # Force an osx-arm64 build
#   ./publish-mac.sh x64            # Force an osx-x64 build
# ==========================================

set -e

# Parse the architecture argument
ARCH_ARG="${1:-auto}"
if [ "$ARCH_ARG" = "auto" ]; then
    HOST_ARCH=$(uname -m)
    case "$HOST_ARCH" in
        arm64|aarch64) RID="osx-arm64" ;;
        x86_64)        RID="osx-x64" ;;
        *)             RID="osx-arm64" ;;
    esac
elif [ "$ARCH_ARG" = "arm64" ] || [ "$ARCH_ARG" = "aarch64" ]; then
    RID="osx-arm64"
elif [ "$ARCH_ARG" = "x64" ] || [ "$ARCH_ARG" = "x86_64" ]; then
    RID="osx-x64"
else
    echo "❌ Unrecognized architecture argument: $ARCH_ARG (valid values: auto | arm64 | x64)"
    exit 1
fi

PublishDir="publish-mac-${RID#osx-}"

echo "🚀 Target platform: $RID"
echo "📂 Output directory: $PublishDir"

if [ -d "$PublishDir" ]; then
    echo "🧹 Cleaning the old publish directory..."
    rm -rf "$PublishDir"
fi

# Check dotnet
if ! command -v dotnet &> /dev/null; then
    echo "❌ dotnet not detected. Install the .NET 10 SDK first: https://dotnet.microsoft.com/download"
    exit 1
fi

echo ""
echo "🚀 Publishing SolTTS.Web (sol-server)..."
dotnet publish SolTTS.Web/SolTTS.Web.csproj \
    -c Release -r "$RID" --self-contained true \
    -o "$PublishDir" \
    /p:PublishSingleFile=false \
    /p:AllowMissingPrunePackageData=true

echo ""
echo "🚀 Publishing SolTTS.CLI (sol-cli)..."
dotnet publish SolTTS.CLI/SolTTS.CLI.csproj \
    -c Release -r "$RID" --self-contained true \
    -o "$PublishDir" \
    /p:PublishSingleFile=false \
    /p:AllowMissingPrunePackageData=true

# Remove the old config.json (if left over)
rm -f "$PublishDir/config.json"

# Copy required resources
echo ""
echo "📦 Copying configuration files..."
if [ -f "config.template.yaml" ]; then
    cp "config.template.yaml" "$PublishDir/config.template.yaml"
    cp "config.template.yaml" "$PublishDir/config.yaml"
fi

if [ -d "resources-minimal" ]; then
    echo "📦 Copying resources (minimal)..."
    cp -r resources-minimal "$PublishDir/resources"
fi

if [ -d "tools" ]; then
    echo "📦 Copying tools directory (converter)..."
    mkdir -p "$PublishDir/tools/converter"
    cp tools/converter/v1_converter.py "$PublishDir/tools/converter/v1_converter.py"
    if [ -d "tools/converter/templates" ]; then
        cp -r tools/converter/templates "$PublishDir/tools/converter/templates"
    fi
    # macOS does not need the Windows-only Python runtime
    if [ -d "$PublishDir/tools/converter/runtime" ]; then
        rm -rf "$PublishDir/tools/converter/runtime"
    fi
fi

# Copy the macOS initialization script
if [ -f "init-env-mac.sh" ]; then
    cp init-env-mac.sh "$PublishDir/init-env-mac.sh"
    chmod +x "$PublishDir/init-env-mac.sh"
fi

# Set executable permissions
chmod +x "$PublishDir/sol-server" 2>/dev/null || true
chmod +x "$PublishDir/sol-cli" 2>/dev/null || true

# Gatekeeper quarantine attributes may block execution on macOS; strip them for local use
if command -v xattr &> /dev/null; then
    xattr -dr com.apple.quarantine "$PublishDir" 2>/dev/null || true
fi

echo ""
echo "✅ Publish complete! Output directory: $(cd "$PublishDir" && pwd)"
echo "Run hints:"
echo "  - Start the Web service: cd $PublishDir && ./sol-server"
echo "  - Start the CLI: cd $PublishDir && ./sol-cli"
echo ""
echo "💡 If Gatekeeper blocks the first run, execute:"
echo "   xattr -dr com.apple.quarantine $PublishDir"
