# Keyframe Selection for Demonstration Compression in Vision-Language-Action Models

**Variant C — the visual branch of demonstration encoding, studied in isolation.**

This project compares **keyframe selection strategies** for compressing robot
manipulation demonstrations. Selection operates on **pixels only** (computer-vision
methods over the video frames), and the compressed demonstrations are evaluated with
an **intrinsic, retrieval-based protocol** — not by training or fine-tuning any
policy.

> Scope is deliberately narrow. There is **no** policy fine-tuning, **no** simulator
> rollouts, **no** task-success-rate evaluation, and **no** robot-state signals
> (velocity, gripper, joint angles) used for selection. See `docs/decisions.md`
> and the **Scope guardrails** section below for the full scope contract and the
> reasoning behind it.

## Methods

Every extractor implements the `KeyframeExtractor` interface in
`src/extractors/base.py` — pixels in, sorted frame indices out, always including
frame 0 and frame T-1.

| Method | Selection signal | Backbone |
|---|---|---|
| **Uniform** | evenly spaced indices | — (baseline) |
| **Random** | random indices, ≥3 seeds (matched-N control) | — (baseline) |
| **Optical flow** | RAFT motion magnitude | `raft_small` (torchvision) |
| **Attention saliency** | ViT attention maps | DINOv2 ViT-S/14 (timm) |
| **Frame difference** | mean absolute pixel change | pure NumPy |

## Evaluation

Intrinsic and retrieval-based — no downstream policy. Three metrics, always reported
together across a compression sweep:

1. **Task retrieval accuracy** (top-1, top-5) — retrieve a compressed query demo's
   task label from a gallery of compressed demos by embedding similarity.
2. **CLIP similarity** — cosine similarity between a compressed demo's pooled
   embedding and the CLIP text embedding of the task instruction.
3. **Frame compression ratio** (K / T) — reported alongside the above, never alone.

Embeddings use **CLIP ViT-L-14-quickgelu (openai)**, pinned in
`configs/models.yaml` (the `-quickgelu` variant matches the activation the OpenAI
weights were trained with — loading them into a plain-GELU `ViT-L-14` is a silent
mismatch; see the config comment). The sweep covers **K ∈ {4, 8, 16, 32}** with
random seeds **{42, 123, 456}**, set in `scripts/run_retrieval_eval.py`.

## Dataset

**BridgeData V2**, streamed one episode at a time from the IPEC LeRobot mirror
(`IPEC-COMMUNITY/bridge_orig_lerobot`). The loader fetches ~12 MB of metadata once,
then downloads and decodes each episode's single camera view
(`observation.images.image_0`, an AV1 MP4, ~0.36 MB) on demand — there is no bulk
download. The data-source choice and the AV1-decode decision are logged in
`docs/decisions.md`.

## Setup

```bash
pip install -r requirements.txt
# AV1 video decoder — required by the loader (prebuilt wheel bundles libdav1d):
pip install --only-binary :all: av==13.1.0
```

The selection and embedding steps (CLIP / RAFT / DINOv2 inference) need a GPU; the
diagnostic suite in `docs/methods.md` §5 is CPU-only and runs from a cached
embedding bundle (see **Reproduce** below). GPU host setup is environment-specific
and not committed.

## Quick start

```bash
# 1. Invariant check on a synthetic episode (no download, no GPU):
python smoke_test.py --synthetic

# 2. Preflight against real data (streams a handful of episodes):
python scripts/preflight_check.py \
    --root ~/.cache/lerobot --embed_cache ~/.cache/clip_embeds

# 3. Full retrieval sweep:
python scripts/run_retrieval_eval.py \
    --root ~/.cache/lerobot --embed_cache ~/.cache/clip_embeds --output_dir results
```

## Reproduce

The headline retrieval grid and the §5 diagnostics reproduce in two phases. Phase 1
needs a GPU; phase 2 is pure NumPy/pandas and runs anywhere.

```bash
# --- Phase 1 (GPU): retrieval grid, consistency stats, cached embeddings ---
python scripts/run_retrieval_eval.py \
    --root ~/.cache/lerobot --embed_cache ~/.cache/clip_embeds --output_dir results
python scripts/run_consistency.py --root ~/.cache/lerobot

# Export the frozen per-frame embedding bundle the diagnostics read:
python scripts/export_eval_bundle.py \
    --root ~/.cache/lerobot --embed_cache ~/.cache/clip_embeds --out_dir results/bundle

# --- Phase 2 (CPU only): the §5 diagnostic suite, off the cached bundle ---
for d in similarity_distributions extra_baselines stats pooling_sensitivity coverage_error; do
    python scripts/diagnostics/$d.py --bundle results/bundle --out_dir results
done

# Retrieval grid tables + figures (the diagnostics write their own tables):
python scripts/make_tables.py --results_dir results
python scripts/plot_results.py
```

The exported bundle (`results/bundle/`, ~150–250 MB) is gitignored and re-created
by the export step; the diagnostic output tables under `results/tables/` are
committed.

## Project layout

```
src/
  extractors/   KeyframeExtractor interface + uniform, random, optical_flow,
                attention (DINOv2), frame_diff
  data/         BridgeData V2 loader (per-episode streaming) + Demo types
  evaluation/   retrieval accuracy, CLIP similarity, pooled embeddings
scripts/        preflight_check, run_retrieval_eval, run_consistency,
                export_eval_bundle, make_tables, plot_results
  diagnostics/  CPU-only suite reading the exported bundle (methods.md §5)
configs/        models.yaml (pinned CLIP / DINOv2 / RAFT identifiers)
docs/           decisions.md (pinned decisions), methods.md (methods & results)
results/        JSON metrics, tables, and plots (committed)
archive/        quarantined drift-era code and docs — read-only, do not import
```

## Scope guardrails

This repository was previously drifted into VLA fine-tuning territory and pulled
back. To prevent re-drift, the following are out of scope and must not be added
without explicit supervisor approval: policy fine-tuning / LoRA / training loops,
simulator rollouts and task-success evaluation, robot-state selection signals, and
new datasets or heavyweight dependencies. The pinned decisions and scope reminders
that govern this repository live in `docs/decisions.md`.
