#!/bin/bash

# RTX 3060 RF-GS Benchmark Setup Script
echo "🚀 Setting up RTX 3060 RF Gaussian Splatting Benchmark Environment"
echo ""

# Check for NVIDIA GPU
if ! nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

echo "✅ GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits

# Create virtual environment
echo ""
echo "📦 Creating Python environment..."
python3 -m venv rf_gs_env
source rf_gs_env/bin/activate

# Upgrade pip
pip install --upgrade pip

echo "🔧 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "📊 Installing additional dependencies..."
pip install matplotlib rich numpy scipy

echo "🔍 Testing CUDA availability..."
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    print('Memory:', f'{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB')
else:
    print('❌ CUDA not available - check your PyTorch installation')
"

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment:"
echo "  source rf_gs_env/bin/activate"
echo ""
echo "To run the benchmark:"
echo "  python code/experiment_rtx3060_rf_gs.py --sweep"
echo ""
echo "Available benchmark options:"
echo "  --sweep          Run multiple configurations"
echo "  --motion         Test with dynamic motion (if neural-correspondence available)"
echo "  --num_gaussians  Single test with specified Gaussian count"
echo "  --width/--height Single test with specified resolution"
echo ""
echo "Expected RTX 3060 12GB results:"
echo "  • 20K Gaussians @ 512x512: ~45-65 FPS"
echo "  • 40K Gaussians @ 512x512: ~25-35 FPS"  
echo "  • 40K Gaussians @ 768x432: ~20-30 FPS"