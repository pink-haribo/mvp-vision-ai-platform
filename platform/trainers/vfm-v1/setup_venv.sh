#!/bin/bash
# VFM-v1 Virtual Environment Setup Script
# Usage: ./setup_venv.sh [cuda_version]
# Example: ./setup_venv.sh cu118  (default)
#          ./setup_venv.sh cu121

set -e

CUDA_VERSION=${1:-cu118}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "VFM-v1 Environment Setup"
echo "CUDA Version: ${CUDA_VERSION}"
echo "========================================"

# Step 1: Create venv
echo ""
echo "[Step 1/6] Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "  .venv already exists. Remove it? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf .venv
        python3.11 -m venv .venv
    fi
else
    python3.11 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip setuptools wheel -q

# Step 2: Install PyTorch
echo ""
echo "[Step 2/6] Installing PyTorch (${CUDA_VERSION})..."
if [ "$CUDA_VERSION" = "cu118" ]; then
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu118 -q
elif [ "$CUDA_VERSION" = "cu121" ]; then
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121 -q
else
    echo "Unsupported CUDA version: ${CUDA_VERSION}"
    echo "Supported: cu118, cu121"
    exit 1
fi

# Verify torch installation
python -c "import torch; print(f'  PyTorch {torch.__version__} installed (CUDA: {torch.version.cuda})')"

# Step 3: Install mmcv from wheel
echo ""
echo "[Step 3/6] Installing mmcv==2.1.0..."
if [ "$CUDA_VERSION" = "cu118" ]; then
    pip install mmcv==2.1.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html -q
elif [ "$CUDA_VERSION" = "cu121" ]; then
    pip install mmcv==2.1.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html -q
fi

# Verify mmcv installation
python -c "import mmcv; print(f'  mmcv {mmcv.__version__} installed')"

# Step 4: Install remaining packages
echo ""
echo "[Step 4/6] Installing dependencies from requirements-version.txt..."
pip install -r "${SCRIPT_DIR}/requirements-version.txt" -q

# Step 5: Install mmyolo (editable)
echo ""
echo "[Step 5/6] Installing mmyolo (editable)..."
pip install --no-build-isolation -e "${SCRIPT_DIR}/third_party/mmyolo" -q

# Verify mmyolo installation
python -c "import mmyolo; print(f'  mmyolo {mmyolo.__version__} installed')"

# Step 6: Create activation script with env vars
echo ""
echo "[Step 6/6] Creating activation helper..."
cat > "${SCRIPT_DIR}/activate_vfm.sh" << 'EOF'
#!/bin/bash
# Activate VFM-v1 environment with required env vars
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.venv/bin/activate"
export MMENGINE_REGISTRY_SKIP_EXISTING=1
echo "VFM-v1 environment activated"
echo "  MMENGINE_REGISTRY_SKIP_EXISTING=1"
EOF
chmod +x "${SCRIPT_DIR}/activate_vfm.sh"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  source activate_vfm.sh"
echo ""
echo "To run training:"
echo "  python train.py --job-id 1 --model-name vfm_v1_l \\"
echo "      --dataset-s3-uri s3://bucket/datasets/xxx/ \\"
echo "      --callback-url http://localhost:8000/api/v1 \\"
echo "      --config '{\"epochs\": 100, \"batch_size\": 4}'"
echo ""

# Verify all imports
echo "Verifying installation..."
python -c "
import torch
import mmcv
import mmengine
import mmdet
import mmyolo
import transformers
import open_clip

print('All packages imported successfully!')
print(f'  torch: {torch.__version__}')
print(f'  mmcv: {mmcv.__version__}')
print(f'  mmengine: {mmengine.__version__}')
print(f'  mmdet: {mmdet.__version__}')
print(f'  mmyolo: {mmyolo.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  open_clip: {open_clip.__version__}')
"
