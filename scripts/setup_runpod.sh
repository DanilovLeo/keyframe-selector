#!/usr/bin/env bash
# setup_runpod.sh
# Run once after cloning the repo on a RunPod pod.
# Usage:  bash scripts/setup_runpod.sh
set -euo pipefail

LEROBOT_CACHE=/workspace/lerobot_cache
CLIP_CACHE=/workspace/clip_embeds
RESULTS_DIR=/workspace/results

echo "=== Installing dependencies ==="
pip install --quiet timm open-clip-torch scipy matplotlib pyyaml

echo "=== Installing lerobot ==="
pip install --quiet lerobot || {
    echo "Standard install failed, trying no-deps fallback..."
    pip install --quiet lerobot --no-deps
    pip install --quiet datasets huggingface_hub
}

echo "=== Verifying imports ==="
python - <<'EOF'
import torch, timm, open_clip, lerobot
print(f"  torch      {torch.__version__}  CUDA={torch.cuda.is_available()}")
print(f"  timm       {timm.__version__}")
print(f"  open_clip  {open_clip.__version__}")
print(f"  lerobot    {lerobot.__version__}")
EOF

echo "=== Creating workspace directories ==="
mkdir -p "$LEROBOT_CACHE" "$CLIP_CACHE" "$RESULTS_DIR"

echo ""
echo "Setup complete. Next step:"
echo ""
echo "  python scripts/preflight_check.py \\"
echo "      --root $LEROBOT_CACHE \\"
echo "      --embed_cache $CLIP_CACHE"
