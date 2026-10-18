#!/bin/bash

# ==========================================
# SolTTS standalone environment setup (Linux)
# Deploys the Python environment required by the model converter
# ==========================================

echo "🚀 Initializing the SolTTS model converter Python environment..."

# 1. Check python3 and its modules
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed. Run 'sudo apt-get install python3 python3-pip python3-venv' (Ubuntu/Debian) or the equivalent package manager command."
    exit 1
fi

echo "✅ Found python3: $(python3 --version)"

# 2. Create and activate the virtual environment (at tools/converter/.venv)
VENV_DIR="tools/converter/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating Python virtual environment ($VENV_DIR)..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ Error: failed to create the virtual environment. You may need to install the 'python3-venv' package."
        exit 1
    fi
else
    echo "✅ Virtual environment already exists: $VENV_DIR"
fi

# 3. Activate the virtual environment
source "$VENV_DIR/bin/activate"
echo "✅ Virtual environment activated"

# 4. Upgrade pip and install dependencies (Tsinghua/Aliyun mirrors for speed)
echo "📦 Installing dependencies (onnx, numpy, onnxsim, onnxruntime)..."
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install onnx numpy onnxsim onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "📦 Installing core dependency (CPU-only PyTorch) to reduce size..."
pip install torch -f https://mirrors.aliyun.com/pytorch-wheels/cpu

echo ""
echo "🎉 Setup complete! The model converter is ready."
echo "💡 To run the converter separately in the background, first run: source tools/converter/.venv/bin/activate"
