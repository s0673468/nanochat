# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What is nanochat

A minimal, hackable full-stack harness for training LLMs from scratch, by Andrej Karpathy. Covers tokenization, pretraining, SFT, RL, evaluation, inference, and a chat UI. One complexity dial (`--depth`) auto-scales all hyperparameters. Trains GPT-2 capability (~505M params, d26) in ~3 hours on 8xH100 for ~$72.

## Commands

```bash
# Setup
uv venv && uv sync --extra gpu    # GPU (CUDA 12.8)
uv sync --extra cpu                # CPU/MPS
source .venv/bin/activate

# Tests
pytest                             # all tests
pytest -m "not slow"               # skip slow tests

# Full GPT-2 pipeline (~3 hours, 8xH100)
bash runs/speedrun.sh

# Quick research iteration (~5 min, 8xH100)
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 --run=d12 --model-tag=d12 --core-metric-every=999999 --sample-every=-1 --save-every=-1

# CPU/MPS demo (~30 min MacBook)
bash runs/runcpu.sh

# Miniseries (sweep depths 12-26)
bash runs/miniseries.sh

# Single-GPU: omit torchrun, use python -m directly (gradient accumulation automatic)
python -m scripts.base_train -- --depth=12

# Evaluation
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# SFT
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16

# Inference
python -m scripts.chat_cli -p "Why is the sky blue?"   # CLI (or omit -p for interactive)
python -m scripts.chat_web                               # Web UI at :8000
```

All scripts are run as modules (`python -m scripts.base_train`, not `python scripts/base_train.py`).

## Architecture

**Pipeline stages:** Tokenization → Pretraining → SFT → (optional RL) → Inference

**The single dial:** `--depth` (number of transformer layers) determines everything:
- `model_dim = depth * aspect_ratio` (default 64), rounded up to `head_dim` (128) multiple
- Optimal batch size, training horizon, learning rates, weight decay all auto-computed
- Reference model is d12; hyperparameters transfer to other depths via muP-style scaling
- GPT-2 capability ≈ d24-d26

**Model (`nanochat/gpt.py`):** Modern GPT transformer with RoPE, QK norm, ReLU² MLP, untied embeddings, no bias, no learnable RMSNorm params. Notable features: sliding window attention (pattern string like `"SSSL"`), value embeddings on alternating layers (ResFormer-style), per-layer learnable residual/x0 scalars, logit softcapping.

**Optimizer (`nanochat/optim.py`):** Split optimizer — Muon (with polar express orthogonalization) for matrix params (attention, MLP), AdamW for embeddings/scalars. Distributed variants for multi-GPU.

**FP8 (`nanochat/fp8.py`):** Custom 272-line drop-in replacement for torchao's Float8Linear. Tensorwise dynamic scaling via `torch._scaled_mm`. ~2x speedup on H100+.

**Dataloader (`nanochat/dataloader.py`):** BOS-aligned best-fit packing (100% utilization, ~35% cropped tokens). Distributed sharding, resumable.

**Evaluation (`nanochat/core_eval.py`):** DCLM CORE score — 22-task ensemble. GPT-2 target: 0.256525. Also tracks val_bpb (bits per byte, vocab-invariant).

**Attention (`nanochat/flash_attention.py`):** Auto-selects FA3 (Hopper+) or PyTorch SDPA fallback. Sliding window + KV cache support.

**Tasks (`tasks/`):** MMLU, GSM8K, ARC, HumanEval (simple Python), SmolTalk, SpellingBee, custom JSONL. `TaskMixture` and `TaskSequence` in `tasks/common.py`.

## Key Environment Variables

- `NANOCHAT_BASE_DIR` — artifact storage (default `~/.cache/nanochat`)
- `WANDB_RUN` — wandb run name (`"dummy"` = no logging)
- `OMP_NUM_THREADS=1` — required for distributed training

## Metrics to Watch

- `val_bpb` vs step/time/FLOPs — primary pretraining signal
- `core_metric` — DCLM CORE score (GPT-2 target: 0.2565)
- `train/mfu` — model FLOPs utilization (bf16 MFU)
- `train/tok_per_sec` — throughput

## Contributing Context

Changes must work across all `--depth` settings (not just d26). The codebase deliberately avoids configuration objects, model factories, and if-else monsters — keep it minimal and readable. AI contributions must be disclosed in PRs.
