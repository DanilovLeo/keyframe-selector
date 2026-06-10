# Decision log

Append-only record of pinned decisions for this project. Newest entries at the
bottom. One dated entry per decision: Context / Decision / Rationale /
Consequences.

---

## 2026-06-08 — Data source: IPEC-COMMUNITY/bridge_orig_lerobot (per-episode lazy fetch)

**Context.** The loader (`src/data/bridge_loader.py`) targeted
`lerobot/bridgedata_v2`, which returns HTTP 401 / RepositoryNotFoundError — the
repo does not exist. An earlier candidate mirror (`TorridFish/bridge_dataset`)
was rejected because its V2 split carries no pixels (only `lang.txt`). We needed
a source that (a) actually exists, (b) ships BridgeData V2 pixels, and (c)
supports random access to a single episode without downloading the whole
dataset ("verdict C" — the full-download tax).

**Decision.** Switch to `IPEC-COMMUNITY/bridge_orig_lerobot` (LeRobot v2.0
packaging of BridgeData V2). The loader now does a metadata-only `__init__`
(fetches `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl`) and a
per-episode `hf_hub_download` of the single camera view
`observation.images.image_0` MP4 inside `load_episode`. Episode→file paths come
from the dataset's own `video_path` template and `chunks_size`, not hardcoded
guesses. The public contract of `BridgeDataLoader` is unchanged.

**Rationale.** The IPEC mirror exists, is public/ungated, and stores one MP4 per
episode per view — ideal for the loader's random-access `load_episode(idx)`
contract. Per-episode fetch cost measured at ~0.36 MB/episode (vs. downloading
the full dataset), and HuggingFace's content-addressed cache means repeated
access across the K-sweep and the 3 random seeds re-reads from disk with zero
re-download (verified: re-loading the same episode added 0 bytes to the cache).

**Consequences.**
- New runtime dependency on `huggingface_hub` per-file download + a video
  decoder (see the AV1 entry below). `datasets.load_dataset` is no longer used
  by the loader.
- First `__init__` downloads ~12 MB of metadata (all 53,192 episodes' index
  rows, no pixels).
- The original RAIL-style loader is archived at
  `archive/bridge_loader_rail.py` (not deleted).
- Only `observation.images.image_0` is fetched; the other three camera views are
  intentionally not pulled (see scope reminder below).

---

## 2026-06-08 — Accept AV1-lossy decode as the pixel source

**Context.** On the IPEC mirror, BridgeData V2 pixels are stored as AV1-encoded
MP4 (one file per episode per view). Raw/lossless RGB frames are not available
from this source. AV1 is a lossy codec, so decoded frames are not bit-identical
to the original camera RGB.

**Decision.** Decode each episode's `observation.images.image_0` MP4 with PyAV
(`frame.to_ndarray(format="rgb24")`) to `(T, 256, 256, 3)` uint8 RGB and accept
the AV1 compression as-is. No attempt to source raw frames.

**Rationale.** Every keyframe-selection strategy in this study (uniform, optical
flow, attention saliency, random) decodes the *same* AV1 frames, so any AV1
compression artifact is a constant applied identically across all four
extractors and across both query and gallery demos in retrieval. A constant
shared by all conditions cancels out of a *comparative* evaluation — it cannot
advantage or disadvantage one extractor relative to another. Raw RGB is simply
not on offer from this mirror, so the alternative is no pixels at all.

**Consequences.**
- Decoded frames differ from the original raw camera frames by AV1 quantisation;
  absolute (non-comparative) pixel statistics should be read with that caveat.
- Decode path requires an AV1-capable decoder; we pin `av==13.1.0`, whose
  prebuilt wheel bundles `libdav1d`. On RunPod, install with
  `pip install --only-binary :all: av` to avoid a source build needing
  ffmpeg + pkg-config. (See `requirements.txt`.)
- `to_ndarray(format="rgb24")` yields RGB directly, so the loader never emits
  BGR — downstream extractors and the retrieval embedder receive the channel
  order they assume.

---

## 2026-06-10 — PROPOSED (pending supervisor sign-off): embedding-space coverage error as a selection-sensitive intrinsic metric

**Status.** PROPOSED, not yet adopted. Implemented as a diagnostic and run for
*provisional* numbers, but **not** promoted to a headline metric in
`methods.md` until the supervisor confirms the framing below. Sign-off requested
2026-06-10.

**Context.** The pinned 3-metric protocol (task retrieval, CLIP similarity,
compression ratio) is selection-invariant on scene-dominated BridgeData v2: the
Tier-1 diagnostics show intra-episode frame similarity (median ~0.917) is
essentially equal to inter-episode-same-task similarity (~0.914), so mean-pooled
retrieval cannot resolve *which* frames a method keeps (0/17 method pairs
significant; the Tier-3 pooling sweep shows max- and best-match aggregation do
not separate them either). The external audit's central recommendation is a
metric that is selection-sensitive *by construction*, so the negative retrieval
result can be contrasted against a metric that actually distinguishes the
methods.

**Decision (proposed).** Add **embedding-space coverage error** as a fourth,
*reported-alongside* intrinsic metric — never standalone, always paired with
retrieval. Definition: given an episode's per-frame CLIP embeddings
`e_1..e_T` (already cached, L2-normalised) and a selector's keyframe set `S`,
for every non-selected frame `t` compute its cosine distance to the nearest
selected frame,
`d(t) = min_{s in S} (1 - cos(e_t, e_s))`,
and report the episode mean (typical distortion) and max (worst uncovered
frame), averaged over episodes. This is vector-quantisation distortion with the
keyframe set as the codebook and the frames as the data.

**Why this is in scope (the framing the supervisor must confirm).** It is an
**evaluation metric, not a training loss**: it is computed post-hoc on frozen,
already-cached embeddings; nothing is optimised, no model is fit, no parameter
is learned to minimise it. It is pixels-in (via the frozen CLIP encoder),
indices-in, scalar-out — no policy, no robot state, no rollout, no new dataset,
no new dependency. The only gray-zone risk the audit flags is that "coverage
error" *reads* like a reconstruction/autoencoding objective; it is not — there
is no reconstruction, only nearest-anchor distance in a fixed embedding space.
This entry exists so that reading is on the record before the metric appears in
the report.

**Rationale.** Coverage error is selection-sensitive by construction: the
consecutive-block control (all `K` frames from the episode start) leaves the
back of the episode uncovered and must score far worse, whereas it is invisible
to mean-pooled retrieval (it scored ~0.80–0.83, indistinguishable from every
other method). The metric rewards exactly what keyframe selection is for —
spanning the episode's visual trajectory with few anchors. The expected outcome
is honest either way: because the metric rewards even spreading, uniform
sampling is near-optimal for coverage almost by definition, so the likely
finding is consecutive-block ≫ adaptive ≳ uniform (higher = worse) — i.e. the
methods *do* separate, and the result is that adaptive selection trades episode
coverage for saliency. That is a stronger, more defensible contribution than a
marginal retrieval win.

**Consequences.**
- New diagnostic `scripts/diagnostics/coverage_error.py`; output
  `results/tables/coverage_error.{md,csv}`. Reuses the cached bundle; no GPU.
- `methods.md` gains a sub-section clearly marked *provisional, pending
  sign-off*.
- If the supervisor rejects the framing, the metric is dropped from the report
  and this entry is annotated REJECTED; no code in the pinned pipeline depends on
  it.
- The audit's second selection-sensitive option (held-out frame identification)
  is **not** adopted; coverage error alone is sufficient and lower-risk.

---

## Scope reminder

This project compresses a **single camera view** (`observation.images.image_0`)
of **BridgeData V2 only**, evaluated with a **CV-only intrinsic retrieval**
protocol. BridgeData **V1 was NOT adopted** — pulling V1 in would require
supervisor approval and is out of scope until then.
