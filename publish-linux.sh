#!/bin/bash

# SolTTS unified publish script (Linux)

PublishDir="publish-linux"
Runtime="linux-x64"

if [ -d "$PublishDir" ]; then
    echo "Cleaning the old publish directory..."
    rm -rf "$PublishDir"
fi

echo "🚀 Publishing SolTTS.Web (sol-server)..."
dotnet publish SolTTS.Web/SolTTS.Web.csproj -c Release -r $Runtime --self-contained true -o "$PublishDir" /p:PublishSingleFile=false /p:AllowMissingPrunePackageData=true

echo "🚀 Publishing SolTTS.CLI (sol-cli)..."
dotnet publish SolTTS.CLI/SolTTS.CLI.csproj -c Release -r $Runtime --self-contained true -o "$PublishDir" /p:PublishSingleFile=false /p:AllowMissingPrunePackageData=true

# Remove the old config.json (if left over)
rm -f "$PublishDir/config.json"

# Copy required resources
echo "📦 Copying resource files..."
if [ -f "config.template.yaml" ]; then
    cp "config.template.yaml" "$PublishDir/config.template.yaml"
    cp "config.template.yaml" "$PublishDir/config.yaml"
fi

if [ -d "resources-minimal" ]; then
    echo "📦 Copying resources (minimal)..."
    cp -r resources-minimal "$PublishDir/resources"
fi

if [ -d "tools" ]; then
    echo "📦 Copying tools directory..."
    mkdir -p "$PublishDir/tools/converter"
    cp -r tools/converter/v1_converter.py "$PublishDir/tools/converter/v1_converter.py"
    cp -r tools/converter/templates "$PublishDir/tools/converter/templates"
    # Remove the Windows-only Python runtime (to keep the Linux package small)
    if [ -d "$PublishDir/tools/converter/runtime" ]; then
        echo "🧹 Cleaning up the Windows Python runtime..."
        rm -rf "$PublishDir/tools/converter/runtime"
    fi
fi

echo "✅ Publish complete! Output directory: $PublishDir"
chmod +x "$PublishDir/sol-server"
chmod +x "$PublishDir/sol-cli"
