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
flow, attention saliency, frame difference, random) decodes the *same* AV1
frames, so any AV1 compression artifact is a constant applied identically across
all extractors and across both query and gallery demos in retrieval. A constant
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

## 2026-06-10 — ADOPTED: embedding-space coverage error as a selection-sensitive intrinsic metric

**Status.** ADOPTED as a fourth intrinsic metric, **reported alongside** the
original three (it does not replace them). It is an evaluation metric computed
post-hoc on frozen cached embeddings — not a training loss, no reconstruction,
no optimisation — so it stays inside the pixels-only, CV-only Variant C scope.
Recorded here so the addition to the pinned 3-metric protocol is on the record.

**Context.** The pinned 3-metric protocol (task retrieval, CLIP similarity,
compression ratio) is selection-invariant on scene-dominated BridgeData v2: the
Tier-1 diagnostics show intra-episode frame similarity (median ~0.917) is
essentially equal to inter-episode-same-task similarity (~0.914), so mean-pooled
retrieval cannot resolve *which* frames a method keeps (0/40 method pairs
significant; the Tier-3 pooling sweep shows max- and best-match aggregation do
not separate them either). The external audit's central recommendation is a
metric that is selection-sensitive *by construction*, so the negative retrieval
result can be contrasted against a metric that actually distinguishes the
methods.

**Decision.** Add **embedding-space coverage error** as a fourth,
*reported-alongside* intrinsic metric — never standalone, always paired with
retrieval. Definition: given an episode's per-frame CLIP embeddings
`e_1..e_T` (already cached, L2-normalised) and a selector's keyframe set `S`,
for every non-selected frame `t` compute its cosine distance to the nearest
selected frame,
`d(t) = min_{s in S} (1 - cos(e_t, e_s))`,
and report the episode mean (typical distortion) and max (worst uncovered
frame), averaged over episodes. This is vector-quantisation distortion with the
keyframe set as the codebook and the frames as the data.

**Why this is in scope.** It is an
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
- `methods.md` gains a sub-section (§5.6) reporting it alongside retrieval.
- The metric is reported-alongside only; no code in the pinned retrieval pipeline
  depends on it, so it can be cut from the report without touching the pipeline.
- The audit's second selection-sensitive option (held-out frame identification)
  is **not** adopted; coverage error alone is sufficient and lower-risk.

---

## 2026-06-12 — PRE-REGISTERED: residual-embedding de-saturation gate (Stage 1)

**Status.** PRE-REGISTERED gate, written *before* the diagnostic runs. This entry
fixes the PASS/FAIL rule and the expected outcome **both ways** so the decision
cannot be rationalised after seeing the numbers. Stage 1 is an internal go/no-go
diagnostic; promoting any *positive* residual result into the report body (a
`methods.md` §5.7) is held pending supervisor sign-off (see
`docs/supervisor_signoff_request.md`, item 4). A *negative* result is recorded as
a one-paragraph caveat in `methods.md`.

**Context.** §5.1 established that BridgeData v2 is scene-dominated: within-episode
frame cosines (median ~0.917) are as high as inter-episode same-task means
(~0.914) and well above the inter-task gap (~0.820). Because every frame
embedding is dominated by the static scene, mean-pooling any subset lands in the
same tight cone, so retrieval is selection-invariant (0/40 method pairs
significant). The audit raised the natural question: does subtracting a
per-episode scene anchor expose the residual *motion / change* signal that
selection could then act on?

**Decision.** Before any retrieval work, run a Stage-1 diagnostic
(`scripts/diagnostics/residual_similarity.py`) that repeats the §5.1
three-distribution analysis in residual space, for two anchor choices:
- **Variant A** `r_t = normalize(e_t − e_0)` — anchor is the first frame.
- **Variant B** `r_t = normalize(e_t − mean_t e_t)` — anchor is the episode mean.
For each variant compute (1) intra-episode pairwise residual cosines, and (2)/(3)
same-task / inter-task cosines between *episode-mean residuals*. Frame 0 in A
yields a zero residual and is excluded; any residual with norm < 1e-6 is set to
the zero vector and excluded from pairwise statistics. **Variant B's episode-mean
residual is identically zero by construction** (the mean of `e_t − mean(e)` is 0),
so B's inter-episode / inter-task panels are *undefined* and reported as empty
with a note; B contributes only its intra-episode distribution.

**Pre-registered decision rule (both ways).**
- **PASS → run Stage 2** if, for the gate-bearing variant (A): intra-episode
  residual similarity drops *clearly* below the inter-task level (the
  within-episode cone widens past the task gap, so selection now has leverage)
  **AND** inter-task structure survives (same-task episode-mean-residual pairs
  remain more similar than cross-task pairs). Operationalised:
  `median(intra_residual) < median(inter_task_residual)` by a margin ≥ 0.05, **and**
  `median(same_task_residual) − median(inter_task_residual) ≥ 0.02`.
- **FAIL → STOP, do not run Stage 2** if residuals are noise: intra-episode
  residual cosines collapse toward 0 with no surviving task structure
  (`same_task ≈ inter_task`, `|Δmedian| < 0.02`). On FAIL, record a one-paragraph
  negative note in `methods.md` (residual de-saturation tried, did not recover
  signal) and close the thread.

**Why this is in scope.** Residuals are *arithmetic on frozen, already-cached
CLIP embeddings* — `e_t − e_0` or `e_t − mean(e)`, then renormalise. No model is
fit, nothing is trained, no parameter is learned, no reconstruction is performed.
It is pixels-in (via the frozen encoder), indices/scalars-out — no policy, no
robot state, no rollout, no new dataset, no new dependency, single view
`image_0` only. Identical scope posture to the 2026-06-10 coverage-error metric.
The only governance nuance: per the pending supervisor email, a *reported
positive* residual result waits for sign-off, hence Stage 1 is gated and Stage 2
is sequenced late.

**Consequences.**
- New diagnostic `scripts/diagnostics/residual_similarity.py`; outputs
  `results/tables/residual_similarity.{md,csv}` and
  `results/plots/fig_residual_similarity.{pdf,png}`. Reuses the cached bundle;
  numpy-only; no GPU.
- Does not touch the pinned retrieval pipeline or any committed result table.
- On PASS, Stage 2 (`residual_retrieval.py`, `methods.md` §5.7) is queued behind
  the other CPU tasks per the mandated order; on FAIL, only the negative note is
  added.

---

## 2026-06-12 — PRE-REGISTERED: velocity-placement explanation for the K=32 coverage crossover

**Status.** PRE-REGISTERED analysis, written *before* the diagnostic runs. It
explains an *already-reported* observation (§5.6: `attention`/`frame_diff` have
worse coverage error than `uniform` at K≤16 but *better* at K=32); it adds no new
metric and changes no reported number. The expected outcome is fixed **both
ways** so the mechanism paragraph cannot be back-fitted to the data.

**Context.** §5.6 found a crossover: even spacing (`uniform`) minimises coverage
error at tight budgets, but content-adaptive anchors (`attention`, `frame_diff`)
overtake it at K=32. The report currently states *that* this happens without
explaining *why*. The candidate mechanism: adaptive methods place their interior
anchors in high embedding-**velocity** regions (`v_t = ||e_t − e_{t−1}||`), so
once the budget is generous they cover the fast-changing parts of the trajectory
— where nearest-anchor distance concentrates — better than velocity-agnostic even
spacing; at tight budgets this is outweighed by the larger temporal holes their
clustering leaves.

**Decision.** Run `scripts/diagnostics/crossover_analysis.py` (numpy-only, reads
the bundle's exported keyframe indices) computing, per method × K, episode means
of: (1) `velocity_ratio` — mean velocity at interior anchors ÷ mean velocity over
all interior frames (>1 ⇒ anchors prefer fast regions; `uniform` ≈ 1 by
construction); (2) `max_gap_ratio` — largest inter-anchor temporal gap ÷ mean gap
(the "hole" the clustering leaves; `uniform` ≈ 1); (3) a coverage decomposition
`mean_cov_highvel` / `mean_cov_lowvel` splitting non-selected frames at the
episode-median velocity. `mean_cov` is recomputed via the §5.6
`episode_coverage` for an exact consistency cross-check.

**Pre-registered decision rule (both ways).**
- **SUPPORTS → add a mechanism paragraph to methods.md §5.6.1** if, at K=32,
  `velocity_ratio` for *both* `attention` and `frame_diff` exceeds `uniform`'s by
  a clear margin (≥ 0.10 and > 1.0), **and** their K=32 coverage advantage over
  `uniform` localises to the high-velocity region (uniform − method coverage is
  larger, i.e. more positive, in `mean_cov_highvel` than in `mean_cov_lowvel`).
- **INCONCLUSIVE → record the null in this log, leave methods.md unchanged** if
  adaptive `velocity_ratio` ≈ `uniform` (no preferential placement) or the K=32
  advantage does not concentrate in high-velocity frames. No mechanism claim is
  added to the report.

**Why this is in scope.** Pure arithmetic on the frozen cached embeddings and the
already-exported keyframe indices — `||e_t − e_{t−1}||`, anchor positions, gaps.
No model, no training, no reconstruction, no robot state, no rollout, no new data,
single view `image_0`. Same posture as the §5.6 coverage metric it explains.

**Consequences.**
- New diagnostic `scripts/diagnostics/crossover_analysis.py`; outputs
  `results/tables/crossover_analysis.{md,csv}` and
  `results/plots/fig_crossover_velocity.{pdf,png}`. Reuses the cached bundle;
  numpy-only; no GPU.
- methods.md is touched *only* on SUPPORTS (a §5.6.1 paragraph); on INCONCLUSIVE
  the outcome lives here and the report is unchanged.

**Result (2026-06-12) — INCONCLUSIVE; methods.md left unchanged.**
`results/tables/crossover_analysis.md`,
`results/plots/fig_crossover_velocity.{pdf,png}`. The pre-registered
velocity-density condition **fails**: at K=32 `velocity_ratio` ≈ 1 for *every*
method (uniform 0.998, attention 1.009, frame_diff 1.010, optical_flow 0.996,
random 1.000). With ~32 anchors the interior is sampled so densely that no method
retains a velocity preference; the adaptive margin over uniform is +0.011/+0.012,
an order of magnitude below the pre-registered ≥ 0.10 — a robust miss, not a
borderline one. Note the preference *does* exist at low K (attention/frame_diff
velocity_ratio ≈ 1.11 at K=4/8 vs uniform ≈ 1.02–1.07) and decays to ~1 by K=32.
The *second* condition holds — the K=32 advantage localises to high-velocity
frames (attention hi-vel adv +0.0032 vs lo-vel −0.0025; frame_diff +0.0045 vs
−0.0005: adaptive covers fast frames better, slow frames slightly worse, net
better) — so the **effect** is real but its proposed **cause** (higher mean
anchor velocity) is not. A subtler explanation (anchoring at velocity *peaks* /
transition frames without raising mean anchor velocity) is plausible but not
cleanly demonstrated by this analysis. Because the pre-registered rule requires
**both** conditions, the mechanism is not confirmed and no paragraph is added to
methods.md §5.6; the artifacts are kept for the record.

---

## 2026-06-12 — PRE-REGISTERED: instance-level retrieval diagnostic (within-episode half-split)

**Status.** PRE-REGISTERED, written *before* the diagnostic runs. A new diagnostic
that re-frames the *pinned* retrieval metric with **episode identity** as the
label (instance ID) instead of the task label. Reported-alongside, like coverage
error; it changes no existing reported number.

**Context.** §4–§5 showed *task* retrieval is scene-saturated and
selection-invariant (0/40 method pairs significant). Open question: at the harder
*instance* level — identify the specific episode, not its task — does selection
matter? If even instance ID is selection-invariant, saturation is deeper still;
if methods separate, there is a selection-sensitive retrieval regime worth
reporting.

**Decision (the fork, pre-registered).**
- **Frame split.** Each episode's frames are split at the temporal midpoint
  `mid = T // 2`: `H1 = [0, mid)`, `H2 = [mid, T)`. *Justification for temporal,
  not random:* a temporal split tests whether the early portion identifies the
  late portion — a stricter instance-consistency test; a random split would put
  temporally-adjacent near-duplicate frames in both halves and inflate the
  within-episode match.
- **Vectors.** Query = pooled selected keyframes (from the full-episode exported
  indices) that fall in `H1`; each episode's gallery entry = its selected
  keyframes that fall in `H2`. Endpoints 0 and T−1 are forced, so both halves are
  always non-empty.
- **Protocol.** Gallery = all 863 episodes' `H2` pools; queries = all 863
  episodes' `H1` pools; a query is correct iff its nearest gallery vector is the
  **same** episode. Chance = 1/863 ≈ 0.0012. Instance ID ignores the task
  gallery/query split (every episode is its own class). Metric: Top-1 (and Top-5)
  same-episode ID per method × K; paired sign-flip permutation test of per-query
  correctness vs `uniform` (random averaged over its seeds). Query = first half is
  fixed (not symmetrised) for a single, pre-stated direction.

**Expected outcome (both ways).**
- **Methods separate** (≥ 1 method beats `uniform` at p < 0.05): instance
  retrieval is a *selection-sensitive* regime — §5.8 reports it as such.
- **Methods tie** (0 significant, as in task retrieval): saturation extends to the
  instance level — §5.8 reports *that*.
Either way §5.8 reports the grid + significance honestly; no headline depends on
which outcome lands.

**Why this is in scope.** Re-uses the frozen cached embeddings and the already
exported keyframe indices; pixels-in (via the frozen encoder), indices-in,
scalar-out. No model, no training, no reconstruction, no robot state, no rollout,
no new dataset, single view `image_0`. It is a *re-labelling* of the approved
intrinsic-retrieval metric (episode identity vs task identity), not a new data
source or model. (Scope note flagged to the supervisor: instance ID is not on the
current sign-off list; it sits inside the approved retrieval protocol, but its
addition is surfaced rather than assumed.)

**Consequences.**
- New diagnostic `scripts/diagnostics/instance_retrieval.py`; outputs
  `results/tables/instance_retrieval.{md,csv}` and
  `results/tables/instance_significance.{md,csv}`. methods.md gains §5.8.
  numpy-only; no GPU.

---

## 2026-06-12 — PRE-REGISTERED: TOST equivalence bound on the retrieval grid (δ = 0.02)

**Status.** PRE-REGISTERED. Complements §5.4 (which reports no *difference*
detected) with an *equivalence* test. A non-significant permutation result is
absence of evidence, not evidence of absence; TOST (two one-sided tests) turns it
into a positive claim — "the methods are equivalent to within ±δ" — when the
(1−2α) CI lies inside (−δ, +δ). Part of Task 2; the 20-task half runs now on the
existing bundle, the 100-task half waits for the GPU bundle.

**Context.** §5.4 found 0/40 method-pair Top-1 differences significant. With only
178 queries that could reflect low power rather than true equivalence. Fixing a
margin δ and running TOST quantifies which it is.

**Decision.** For every method pair at every K (the same C(5,2)×4 = 40 cells as
§5.4), on paired per-query Top-1 correctness (random averaged over seeds),
compute the paired mean difference Δ, its **90% CI** (Student-t, df = n−1), and
`p_TOST = max(p_lower, p_upper)` against ±δ with **δ = 0.02**. Equivalence is
declared at α = 0.05 iff the 90% CI ⊂ (−0.02, +0.02). Output
`results/tables/equivalence_tost.{md,csv}` per bundle.

**Pre-registered expectation (both ways).**
- If the 90% CIs fit within ±0.02 → certify the methods equivalent within ±0.02
  (strengthens §4.3's "no method beats uniform").
- If CIs exceed ±0.02 (likely at n = 178, where the paired binary 90% half-width
  is ≈ ±0.03) → report that the 20-task sample *bounds* differences to its
  achievable CI but is underpowered to certify the tighter ±0.02, which
  quantitatively motivates the 100-task scale-up (~5× queries shrink the CI ~2.2×
  to ≈ ±0.013). Either way the numbers are reported honestly; δ is fixed in
  advance.

**Why this is in scope.** Paired arithmetic on the existing per-query correctness
derived from the frozen cached embeddings; no model, no training, no robot state,
no rollout, no new dataset, single view.

**Consequences.**
- New diagnostic `scripts/diagnostics/equivalence.py`; output
  `results/tables/equivalence_tost.{md,csv}`. methods.md §5.4 gains an
  equivalence paragraph (§5.4.1). The 100-task variant awaits the GPU bundle
  (Task 2); the diagnostic takes `--bundle`, so it re-runs unchanged on it.

**Result (2026-06-12) — UNDERPOWERED at ±0.02 (second branch fired, as
anticipated).** On the 20-task bundle (n = 178 queries) only **7 / 40** pairs
have their 90% CI fully inside (−0.02, +0.02); the rest are inconclusive (CI
straddles a boundary), and *none* fall outside ±δ on both sides — i.e. no pair is
certified *different* either, consistent with §5.4's 0/40. The 90% CI half-width
is **median 0.0235, max 0.0406**, so the sample as-is certifies equivalence only
to ≈ ±0.04, not the pre-set ±0.02. The 7 equivalent pairs cluster at K = 32 (6 of
7), where Top-1 has saturated and per-query differences are tiny. This is the
predicted outcome and quantitatively motivates the 100-task scale-up (~5× queries
→ CI ≈ ±0.013 < δ). No goalposts moved: δ stayed 0.02; the honest read is "bounded
to ±0.04, underpowered for ±0.02." methods.md §5.4.1 records this.

---

## 2026-06-12 — PRE-REGISTERED: DINOv2 retrieval backbone (cross-encoder check on the saturation finding)

**Status.** PRE-REGISTERED, GPU-blocked (re-embedding needs the GPU; the
diagnostic/analysis path stays numpy-only). Part of Task 4. Pinned identifier
already in `configs/models.yaml` (`dinov2.timm_model = vit_small_patch14_dinov2`,
ViT-S/14, 384-dim); timm is already a project dependency and DINOv2 is already
stamped into `bundle_meta.json` as provenance — **no new dependency, no new
dataset, no new view.**

**Context.** §5.1–§5.5 show task retrieval is saturated and selection-invariant
under **CLIP ViT-L/14** embeddings: intra-episode frame cosine (0.917) ≈
inter-episode same-task (0.914) ≫ inter-task (0.820), so scene similarity
dominates and no selection method separates. A live alternative explanation is
that this is a **CLIP-specific artifact** — CLIP is caption/scene-biased by its
contrastive image–text pretraining, which could inflate same-scene similarity. A
self-supervised, vision-only backbone (DINOv2) is the natural cross-encoder
control: if saturation persists under DINOv2, it is a property of the *data*
(near-identical frames within a BridgeData episode), not of CLIP.

**Decision.** Re-embed the **same 20-task set, same `image_0` view** with DINOv2
under a **separate embedding-cache key** (so CLIP caches are untouched), export a
parallel bundle (`results/bundle_dinov2/`), and re-run the saturation + retrieval
diagnostics: similarity distributions and the Top-1/Top-5 method×K grid with
bootstrap CIs and the paired permutation grid. Outputs
`results/tables/dinov2_similarity.{md,csv}` and
`results/tables/dinov2_retrieval.{md,csv}`. **No CLIP↔text similarity metric**
(DINOv2 has no text tower; the CLIP-text metric stays CLIP-only and is unchanged).
CLIP remains the pinned primary retrieval backbone (CLAUDE.md); DINOv2 is an
additional analysis backbone, not a replacement.

**Pre-registered expectation (both ways).**
- **If DINOv2 also saturates** (intra ≈ inter-task gap, between-method Top-1
  spread inside the binomial band, 0/40-style null) → confirms the saturation and
  selection-invariance are **data-intrinsic**, robust across a contrastive and a
  self-supervised encoder. Strengthens §4.3/§5.1 and closes the "CLIP artifact"
  threat. Record as a new methods subsection (cross-backbone replication).
- **If DINOv2 de-saturates** (inter-task gap widens, a selection method separates
  beyond CIs, ≥1/40 significant) → the saturation is **CLIP-specific** and
  selection signal is recoverable under a vision-only encoder. This would be a
  genuinely positive result: it reframes §4–§5 as backbone-dependent and warrants
  promoting DINOv2 to a co-primary backbone (with supervisor notice). Either way
  the numbers are reported honestly; the decision rule is fixed in advance.

**Why this is in scope.** Pixels-only CV image encoder (no text, no robot state,
no rollout, no fine-tuning); same dataset, same single view; the model and its
dependency are already pinned. The only methodological addition is using an
**already-pinned** encoder as a second retrieval backbone for a robustness check —
documented here, sanctioned by the Task 4 brief. Flagged for the supervisor
sign-off list as "second retrieval backbone (DINOv2), same data/view, no new
dependency."

**Consequences.**
- **Code (landed, CPU-validated):** `export_eval_bundle.py` gains a
  `--backbone {clip,dinov2}` flag (default `clip`, byte-identical to before);
  `dinov2` swaps in `src/evaluation/retrieval.py::{load_dinov2,
  embed_all_frames_dinov2}` (same timm recipe as the attention extractor) and a
  distinct embedding cache (`~/.cache/kf_eval/dinov2_embeds`). The bundle meta now
  stamps `retrieval_backbone` / `embedding_model`. New thin driver
  `scripts/diagnostics/dinov2_retrieval.py` re-uses the CLIP-side
  `similarity_distributions` + `stats` machinery and writes
  `dinov2_similarity.*`, `dinov2_retrieval.*` (boot CIs), `dinov2_permutation.*`,
  plus the PERSISTS/BREAKS verdict. Validated on the CLIP bundle: it reproduces
  §5.1 medians and the `retrieval_cis` grid exactly.
- **GPU-blocked (the only remaining step):** re-embed with
  `python scripts/export_eval_bundle.py --backbone dinov2 --out_dir
  results/bundle_dinov2 --allow_embed`, then
  `python scripts/diagnostics/dinov2_retrieval.py --bundle results/bundle_dinov2
  --out_dir results_dinov2`. Selection extractors are backbone-independent, so the
  DINOv2 bundle's `keyframes.jsonl` is identical to CLIP's; only
  `frame_embeddings.npz` (384-dim) differs. CLIP tables and all previously
  reported numbers are untouched.

**Result (2026-06-13) — PERSISTS after multiple-comparison correction; recorded as
methods §5.9 (cross-backbone replication).** Re-embedded all 863 episodes (same
20 tasks, same `image_0`) with DINOv2 ViT-S/14 on a fresh GPU pod and ran the
driver (`results_dinov2/tables/dinov2_{similarity,retrieval,permutation}.{md,csv}`).
- **Saturation reproduces.** Intra-episode frame-pair cosine median **0.840 ≈
  same-task 0.823** (Δ = 0.017, tight-gate PASS), both ≫ inter-task **0.461**.
  Retrieval grid: all 20 method × K Top-1 cells in **0.798–0.837**, every bootstrap
  95% CI overlapping. The §5.1/§5.4 structure holds under a vision-only backbone.
- **One real difference — the task gap is ~4× wider** (same − inter = 0.361 vs
  CLIP's 0.094): DINOv2 separates tasks more (no shared-caption pull), yet
  selection still does not move retrieval. This matches the pre-registered
  "DINOv2 also saturates" branch, not the "de-saturates / promote to co-primary"
  branch — so **CLIP stays the pinned primary**; DINOv2 remains a robustness control.
- **The driver printed `BREAKS`, and this is reported honestly as a rule
  limitation, not a finding.** The pre-registered verdict triggers on **≥ 1/40
  uncorrected** permutation rejections; exactly **1/40** fired — uniform vs random
  at K=8, Δ = 0.032, p = 0.0093. That single hit is a multiple-comparisons false
  positive: the chance-expected false-positive count across 40 tests at α = 0.05 is
  **2** (so 1/40 is below chance), the next-smallest p is 0.106 (a 10× gap), and it
  survives **no** correction — Bonferroni / Holm / Benjamini–Hochberg all set the
  rank-1 threshold at ≤ 0.00125. It is also two *non-CV baselines* at a *single* K
  (a coverage-uniformity blip, cf. §5.6), not a CV method consistently winning.
  **After correcting for the 40 comparisons, 0/40 pairs differ → corrected verdict
  PERSISTS.** The pre-registration omitted a multiple-comparison correction; that
  omission — the sole reason the mechanical verdict read `BREAKS` — is the deviation
  recorded here. We report both the mechanical verdict and the corrected conclusion.
- **Reproducible offline.** The DINOv2 bundle (`results/bundle_dinov2/`, untracked
  per CLAUDE.md) was copied back to CPU; re-running the driver locally regenerates
  the retrieval and permutation tables **byte-identically** and the similarity
  percentiles to ~1e-6 (cross-numpy float rounding). The pod's run is authoritative
  and committed; no GPU is needed to re-derive §5.9.

---

## Scope reminder

This project compresses a **single camera view** (`observation.images.image_0`)
of **BridgeData V2 only**, evaluated with a **CV-only intrinsic retrieval**
protocol. BridgeData **V1 was NOT adopted** — pulling V1 in would require
supervisor approval and is out of scope until then.
