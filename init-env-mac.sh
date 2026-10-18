#!/bin/bash

# ==========================================
# SolTTS standalone environment setup (macOS)
# Used to:
#   1. Check/install the audio player required for TTS (ffplay recommended)
#   2. Deploy the Python environment required by the model converter
# ==========================================

set -e

echo "🚀 Initializing the SolTTS macOS runtime environment..."
echo ""

# 1. Check for Homebrew (recommended installation method)
HAS_BREW=0
if command -v brew &> /dev/null; then
    HAS_BREW=1
    echo "✅ Found Homebrew: $(brew --version | head -n1)"
else
    echo "⚠️  Homebrew not detected (https://brew.sh). Installing it is recommended for automatic ffmpeg installation."
fi

# 2. Recommend installing ffmpeg (provides ffplay for streaming playback)
if ! command -v ffplay &> /dev/null; then
    echo ""
    echo "ℹ️  ffplay not detected. Installing ffmpeg is recommended for low-latency streaming playback."
    if [ $HAS_BREW -eq 1 ]; then
        echo "📦 Installing ffmpeg via Homebrew..."
        brew install ffmpeg || echo "⚠️  ffmpeg installation failed; you can still use the built-in afplay (non-streaming)."
    else
        echo "💡 You can install it manually: brew install ffmpeg"
        echo "   Without ffplay, CLI streaming mode automatically falls back to afplay."
    fi
else
    echo "✅ Found ffplay: $(which ffplay)"
fi

# 3. Check python3 (the macOS-bundled python3 may be a stub; brew install python is recommended)
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed. Run: brew install python"
    exit 1
fi
echo "✅ Found python3: $(python3 --version)"

# 4. Create and activate the virtual environment (at tools/converter/.venv)
VENV_DIR="tools/converter/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating Python virtual environment ($VENV_DIR)..."
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
echo "✅ Virtual environment activated"

# 5. Install Python dependencies (macOS uses the official index by default; you can switch to a local mirror)
echo "📦 Upgrading pip and installing packages (onnx, numpy, onnxsim, onnxruntime)..."
pip install --upgrade pip
pip install onnx numpy onnxsim onnxruntime

echo "📦 Installing core dependency (CPU-only PyTorch)..."
pip install torch

echo ""
echo "🎉 Setup complete! The model converter is ready."
echo "💡 To activate the converter virtual environment later: source tools/converter/.venv/bin/activate"
