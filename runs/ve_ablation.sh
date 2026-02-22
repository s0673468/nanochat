#!/bin/bash

# Value Embedding Ablation Study: per_layer vs v1_reuse (classic ResFormer)
#
# Compares two value embedding strategies at d10:
# - per_layer: independent nn.Embedding table per VE layer (current default, 196M params)
# - v1_reuse: compute V from layer 0's c_v, reuse at subsequent VE layers (91M params)
# Both have identical FLOPs — the VE tables are just lookups (zero compute).
#
# Expected time: ~10-15 min per run on RTX 5090 Mobile, ~50-60 min on MacBook MPS.
#
# Run as:
# bash runs/ve_ablation.sh

set -e

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Ensure tokenizer + data are ready (skip if already done)
python -m nanochat.dataset -n 8
if [ ! -f "$NANOCHAT_BASE_DIR/tok32768/tokenizer.json" ]; then
    python -m scripts.tok_train --max-chars=2000000000
fi

# Detect GPU vs CPU/MPS and set appropriate flags
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "CUDA detected — using GPU settings"
    DEVICE_BATCH=32
    HEAD_DIM=128
    SEQ_LEN=1024
else
    echo "No CUDA — using CPU/MPS settings"
    DEVICE_BATCH=16
    HEAD_DIM=64
    SEQ_LEN=512
fi

# Common training args — identical for both runs
DEPTH=10
COMMON_ARGS="--depth=$DEPTH \
    --head-dim=$HEAD_DIM \
    --window-pattern=L \
    --max-seq-len=$SEQ_LEN \
    --device-batch-size=$DEVICE_BATCH \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --num-iterations=5000 \
    --run=$WANDB_RUN"

echo ""
echo "=== VE Ablation Study ==="
echo "Depth: $DEPTH | Iterations: 5000 | Batch: 16384 tokens | Seq: $SEQ_LEN | Head dim: $HEAD_DIM"
echo ""

# Run 1: per_layer (current nanochat default)
echo "[1/2] Training ve_mode=per_layer..."
python -m scripts.base_train \
    $COMMON_ARGS \
    --ve-mode=per_layer \
    --model-tag=ve_ablation_per_layer \
    2>&1 | tee "$NANOCHAT_BASE_DIR/ve_ablation_per_layer.log"

echo ""

# Run 2: v1_reuse (classic ResFormer — reuse layer 0's V)
echo "[2/2] Training ve_mode=v1_reuse..."
python -m scripts.base_train \
    $COMMON_ARGS \
    --ve-mode=v1_reuse \
    --model-tag=ve_ablation_v1_reuse \
    2>&1 | tee "$NANOCHAT_BASE_DIR/ve_ablation_v1_reuse.log"

# Compare results
echo ""
echo "=== Results ==="
echo ""
echo "--- per_layer (independent VE tables) ---"
grep -E "Number of parameters|val_bpb" "$NANOCHAT_BASE_DIR/ve_ablation_per_layer.log" | tail -3
echo ""
echo "--- v1_reuse (reuse layer 0's V) ---"
grep -E "Number of parameters|val_bpb" "$NANOCHAT_BASE_DIR/ve_ablation_v1_reuse.log" | tail -3
echo ""
echo "Done! Full logs in $NANOCHAT_BASE_DIR/ve_ablation_*.log"
