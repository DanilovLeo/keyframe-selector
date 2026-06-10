# Methods & Results

Working draft of the methods/results section. Scope is **Variant C**: the visual
branch of demonstration encoding, studied in isolation with **CV-only keyframe
selection on pixels** and an **intrinsic, retrieval-based** evaluation. No policy
training, no rollouts, no success-rate metrics. See `docs/brief.md` for the
approved brief and `docs/decisions.md` for pinned decisions.

---

## 1. Dataset

BridgeData v2, via the LeRobot mirror `IPEC-COMMUNITY/bridge_orig_lerobot`
(pinned in `docs/decisions.md`). The loader fetches metadata once, then lazily
downloads one MP4 per episode for the single camera view
`observation.images.image_0`, decoded with PyAV to `(T, 256, 256, 3)` uint8 RGB.
AV1 decode is lossy, but the same decoded frames feed every extractor and both
sides of every retrieval pair, so the artifact is a constant that cancels in a
comparative evaluation.

This run used **20 tasks** with ≥ 20 episodes each (`min_demos = 20`), up to 50
episodes per task — **863 episodes total** (685 gallery + 178 query).

## 2. Keyframe extractors

All extractors implement `src/extractors/base.py::KeyframeExtractor.extract(demo)
-> list[int]`, returning sorted indices that always include frame 0 and T-1.

| Extractor | Signal | Notes |
|---|---|---|
| **uniform** | evenly spaced indices | baseline |
| **random** | uniform random indices | matched-N control; ≥ 3 seeds (42, 123, 456), reported mean ± std; endpoints forced (slight upward bias, intentional) |
| **optical_flow** | RAFT-Small flow magnitude per frame-pair | local minima of motion → "settled" frames |
| **attention** | DINOv2 ViT-S/14 CLS-token attention dispersion | salient-frame peaks |
| **frame_diff** | pixel mean-absolute-difference | lightweight pixel-change baseline |

Two K regimes are used:

- **Fixed-K (retrieval sweep):** every extractor selects exactly K ∈ {4, 8, 16,
  32}, so at a given K all methods share the same compression ratio — an
  apples-to-apples comparison of *which* frames, not *how many*.
- **Natural-K (consistency check):** the three CV extractors choose their own
  count; uniform/random are pinned at K = 10 as the fixed-K reference.

## 3. Embedding & evaluation protocol

- **Frame embeddings:** CLIP **ViT-L-14-quickgelu / openai** (768-dim), pinned in
  `configs/models.yaml`. The `-quickgelu` variant matches the activation the
  OpenAI weights were trained with; the plain `ViT-L-14` would silently pair
  standard-GELU layers with QuickGELU-trained weights and degrade every
  embedding. All T frames are embedded once per episode and disk-cached.
- **Demo embedding:** L2-normalised per-frame embeddings indexed by the keyframe
  set → mean-pool → L2-normalise → one (768,) vector per (demo, extractor).
  This pooling is **order-invariant by construction**: it maps any keyframe set
  to an unordered bag, so it cannot, even in principle, reward motion-aware
  *ordering* — a design property (not a tunable risk) revisited in §5.
- **Metrics** (`src/evaluation/`):
  1. **Task retrieval** — 80/20 gallery/query split per task (seed 42), pooled
     into one multi-class problem; rank gallery by cosine similarity; Top-1 and
     Top-5.
  2. **CLIP similarity** — cosine between the pooled demo embedding and the CLIP
     text embedding of the task instruction.
  3. **Compression ratio** — K / T, reported alongside, never alone.
- **Sweep:** K ∈ {4, 8, 16, 32}; metrics reported across the sweep, never at a
  single K.

Reproduction commands are in `RUNPOD.md` (§5–§7). Figures: `scripts/plot_results.py`;
tables: `scripts/make_tables.py`.

---

## 4. Results

### 4.1 Task retrieval is insensitive to selection strategy

Top-1 accuracy by method × K (`results/tables/retrieval_top1_pivot.md`):

| method | K=4 | K=8 | K=16 | K=32 |
|---|---|---|---|---|
| uniform | 0.809 | 0.815 | 0.826 | 0.837 |
| random | 0.826 | 0.839 | 0.818 | 0.835 |
| optical_flow | 0.803 | 0.826 | 0.815 | 0.826 |
| attention | 0.820 | 0.809 | 0.831 | 0.837 |
| frame_diff | 0.837 | 0.831 | 0.831 | 0.831 |

Every method, at every K, lands in a tight **0.80–0.84** band (Top-5: 0.94–0.97;
full grid in `results/tables/retrieval_summary.md`, curves in
`results/plots/fig1_accuracy_vs_cr.{pdf,png}`). The spread between methods at a
fixed K (≤ ~0.03) is no larger than — and sometimes smaller than — the random
baseline's own seed-to-seed std, and the single best cell is `random_k8`
(0.839), i.e. noise. With only 178 queries the binomial 95% CI on a Top-1 of
~0.82 is ≈ ±0.056 — *wider than the entire between-method spread* — so no method
is statistically distinguishable from any other (proper bootstrap CIs and a
paired permutation test confirm this in §5.4 — 0/17 method pairs are significant
at p < 0.05). CLIP
image–text similarity is flat at **~0.209–0.210** across all methods and all K;
absolute CLIP-sim values are compressed and incomparable, so it carries no
discriminative signal here and is relegated to an appendix
(`fig2_clipsim_vs_cr`) — its flatness is itself further evidence of saturation,
not a usable metric.

Two facts hold simultaneously and are both important:

- Retrieval **works** — ~0.80 Top-1 and ~0.95 Top-5 are far above the 1/20 =
  0.05 chance line.
- Retrieval is **blind to which frames are kept** — motion-aware, saliency-aware,
  and random selection are statistically indistinguishable.

**Mechanism.** Task identity in BridgeData v2 is *scene-dominated*: the objects,
background, and camera are near-static within an episode, so almost every frame
already reveals the task. Mean-pooling L2-normalised CLIP embeddings over any
subset — 4 frames or 32, motion-picked or random — yields nearly the same demo
vector. The retrieval metric therefore measures *scene recognition*, which is
easy and selection-invariant, rather than *motion content*, which is where the
CV extractors differ.

### 4.2 Where the methods actually diverge: keyframe-count behaviour

Aggregate over 20 tasks (`results/tables/consistency_aggregate.md`,
`results/plots/fig4_consistency.{pdf,png}` and the per-demo distribution in
`fig3_kf_distribution`):

| extractor | mean_KF | cv_kf | mean_CR |
|---|---|---|---|
| uniform_10 | 9.99 | 0.005 | 0.488 |
| random_10 | 9.99 | 0.005 | 0.488 |
| optical_flow | 2.99 | 0.219 | 0.136 |
| attention_dino | 3.57 | 0.187 | 0.163 |
| frame_diff | 3.56 | 0.175 | 0.163 |

Read this in two parts:

- **Compression.** The content-adaptive methods self-select ~3 frames
  (CR ≈ 0.14–0.16) *without being told a budget*. The often-quoted ~3× gap
  against the fixed-10 references (CR ~0.49) is an artifact of the **arbitrary
  K=10 reference**: uniform at K=4 already reaches CR 0.19 at no retrieval cost
  (§4.1), so non-adaptive selection compresses just as hard. The genuine
  differentiators are therefore (i) *unsupervised* budget selection — the
  adaptive methods find ~3 frames with no K chosen by hand — and (ii)
  **episode-length-stable CR**: adaptive CR stays in 0.10–0.23 across tasks,
  whereas uniform-10's CR swings 0.30 → 0.74 purely with episode length (half
  the frames of a short reaching demo, a third of a long closing demo, with no
  content awareness). CR *stability*, not raw compression, is the real result.
- **Count consistency (cv_kf).** Do **not** compare adaptive vs fixed-K here:
  uniform/random sit at cv ≈ 0.005 *by construction* (they ignore content), not
  by merit. Among the three adaptive methods, the *observed* ordering of count
  variability is `frame_diff` (0.175) ≲ `attention` (0.187) < `optical_flow`
  (0.219) — frame_diff/attention pick a more predictable number of keyframes per
  task — but the task-to-task spreads overlap (fig4), so this describes the
  means, not a tested ranking. A paired comparison of cv_kf across the 20 tasks
  would test the ordering; it is not part of the §5 suite (which targets the
  retrieval metric) and remains future work.

### 4.3 Synthesis

Combining the two: retrieval is **flat from K=4 to K=32** (and, by §4.1's CI,
statistically flat *across methods* at every K), so pushing the keyframe budget
down costs almost nothing on this metric; and the CV methods *naturally* operate
at that low-budget end (~3 frames). The defensible claim is therefore:

> **Intrinsic retrieval accuracy is preserved down to K ≈ 4 regardless of
> selection strategy; content-adaptive methods converge to that low budget
> *unsupervised* and with markedly more episode-length-stable compression ratios
> (CR 0.10–0.23) than fixed-K references (0.30–0.74).** They do **not** beat
> uniform or random at a matched budget, because the retrieval metric is
> scene-saturated and — being built on an order-invariant mean-pool — is
> structurally unable to resolve *which* frames are kept.

This is a legitimate negative/neutral result for an intrinsic-evaluation study,
not a failure of the CV methods: the limitation lives in the *metric*, and the
diagnostic suite in §5 now proves that quantitatively rather than narrating it.
It also sharpens the claim. The result is **not** that the data lacks signal: an
oracle that selects a single frame *with* the label retrieves the correct task
for **95.5%** of queries (§5.3), so a discriminative frame exists in almost every
episode. What fails is recovery — *unsupervised* CV selection feeding an
*order-invariant mean-pool* dilutes that frame among redundant ones. The
finding is therefore "the signal exists but is unrecoverable by unsupervised CV
under a mean-pooled metric," not "the benchmark is dead."

---

## 5. Diagnostic suite (saturation, measured)

§4 argued that the retrieval metric is scene-saturated and selection-invariant.
This section proves it from the cached frame embeddings — no GPU, no re-run, no
policy/rollout/robot-state/new-dataset. Every script reads only the exported
analysis bundle (`scripts/export_eval_bundle.py`); the shared numpy-only loader
is `scripts/diagnostics/bundle.py`.

### 5.1 Mechanism: intra-episode similarity ≈ the inter-task gap

Pairwise CLIP cosine-similarity distributions
(`results/tables/similarity_distributions.md`,
`results/plots/fig5_similarity_distributions.{pdf,png}`):

| comparison | median | mean ± std | n |
|---|---|---|---|
| intra-episode (frame pairs) | 0.917 | 0.907 ± 0.052 | 277,809 |
| inter-episode, same task (means) | 0.914 | 0.906 ± 0.056 | 18,912 |
| inter-task (means) | 0.820 | 0.819 ± 0.056 | 353,041 |

Frames within one episode are as similar to each other (0.917) as two
*different* same-task episodes are (0.914); the only real separation the
embedding carries is the inter-task gap (0.820). So any subset mean of an
episode's frames lands in a tiny cone shared by every other subset — retrieval
**cannot** depend on which frames are chosen. The scene-dominance mechanism is
now quantitative, not narrative.

### 5.2 One frame suffices, and the metric is blind to coverage

Single-frame and degenerate-block controls
(`results/tables/extra_baselines.md`; these intentionally break the
"include 0 and T-1" convention — they probe the metric, they are not proposed
extractors):

| control | Top-1 | mean CR |
|---|---|---|
| frame 0 only (K=1) | 0.764 | — |
| middle frame only (K=1) | 0.770 | — |
| consecutive block from start, K=4 | 0.798 | 0.190 |
| consecutive block from start, K=8 | 0.826 | 0.380 |

A **single** frame retrieves the task ~77% of the time — within ~5–7 points of
the full K-sweep (~0.82): one frame suffices for scene recognition. And a
maximally pathological selection (a consecutive block from the episode start,
worst-case temporal coverage) scores **0.798–0.826**, indistinguishable from
every spread-out method. The metric is insensitive even to degenerate selection
— a stronger statement than random-baseline parity.

### 5.3 Oracle upper bound: the signal exists but is unrecoverable

Label-aware **selection** (never deployable; an upper bound on what any selector
could extract under this metric):

| oracle | Top-1 |
|---|---|
| any single frame retrieves correctly (exact K=1 ceiling) | 0.955 |
| greedy K=4 margin maximisation | 0.927 |

For **95.5%** of queries some single frame retrieves the correct task, and a
greedy oracle K=4 reaches 0.927 — **~10–13 points above** the ~0.82 plateau of
every deployable method. The full-episode mean also sits at ~0.82. So the
discriminative frame-level signal is present in almost every episode; what
destroys it is the combination of (i) *unsupervised* selection, which cannot
know which frame is discriminative, and (ii) the *order-invariant mean-pool*,
which averages that frame together with many redundant ones. This converts the
result from "nothing is findable" to "the signal exists and is unrecoverable by
unsupervised CV under a mean-pooled metric" — a measured ceiling, not a guess.

### 5.4 Statistical significance: no method pair is distinguishable

Bootstrap 95% CIs (10k resamples of the 178 queries) and paired permutation
tests (per-query correctness, sign-flip null; random averaged over its 3 seeds)
on the Top-1 grid (`results/tables/retrieval_cis.md`,
`results/tables/retrieval_permutation.md`):

- Every method × K Top-1 CI overlaps every other; point estimates span only
  0.803 (optical_flow K=4) to 0.839 (random K=8).
- **0 of 17** comparisons reach p < 0.05. The smallest p-value is 0.056
  (random K=8 vs uniform K=8); the single largest gap anywhere in the grid
  (optical_flow vs frame_diff at K=4, Δ = 0.034) gives p = 0.11.

Every "X beats Y" reading in the grid — including the `frame_diff_k4 = 0.837`
and `random_k8 = 0.839` peaks — is inside the noise.

### 5.5 Pooling sensitivity: saturation is not a mean-pool artifact

Re-running the full method × K grid under two aggregators that do **not** average
the keyframe set into one point (`results/tables/pooling_sensitivity.md`):
**max-pool** (coordinate-wise extremum) and **best-match** (no pooling; demo =
set of keyframe embeddings, similarity = max cosine over all query×gallery frame
pairs). This is a sensitivity analysis on the pinned protocol, not a protocol
change. The between-method Top-1 spread at each K stays in **0.006–0.039** under
all three aggregators — inside the ±0.056 binomial band. Max-pool lifts the
floor ~1–2 points but separates nothing. The invariance is therefore a property
of the **data** (near-identical frames), not of the averaging step.

### 5.6 A selection-sensitive contrast: coverage error

> Added as a fourth intrinsic metric, **reported alongside** the original three
> (`docs/decisions.md`, 2026-06-10, ADOPTED). It is an evaluation metric computed
> post-hoc on frozen cached embeddings — no training, no reconstruction, no
> optimisation — so it stays inside the pixels-only, CV-only Variant C scope.

Embedding-space **coverage error** — for each non-selected frame, cosine
distance to its nearest selected frame; episode mean/max averaged over episodes
(VQ distortion with the keyframe set as codebook, lower = better,
`results/tables/coverage_error.md`). Unlike retrieval, it **separates** the
methods:

- The **consecutive-block** control is far worst at every K (e.g. K=4 mean 0.077
  vs ~0.046–0.050) — the metric sees the pathological coverage that retrieval
  scored as identical (§5.2).
- **uniform** is best or near-best at every low-to-mid K (even spreading
  minimises coverage error almost by construction).
- **optical_flow** is the worst non-degenerate method at every K (it clusters
  anchors on "settled" frames, leaving trajectory gaps); attention/frame_diff
  sit just above uniform and match it by K=32.

The honest reading is that adaptive selection **trades episode coverage for
saliency**: a metric sensitive to *which* frames are kept does distinguish the
methods, and it favours even spreading over content-adaptive clustering. This is
the selection-sensitive companion the retrieval metric structurally cannot be.

---

## 6. Threats to validity

- **Metric saturation.** Top-5 ≈ 0.95 and the 20-task gallery includes
  near-duplicate instructions: three labels ("Close the drawer" / "close the
  drawer" / "closed the drawer") are the *same* task, and further variants (the
  four `*box flap(s)` labels, "close fridge" / "close low fridge") are
  near-duplicates, so the **effective class count is ≤ 18, likely lower** —
  Top-5 is partly duplicate labels absorbing errors. A de-duplicated gallery is
  **not** the fix: it would lift apparent Top-1 but attacks label confusion, not
  the real mechanism (intra-episode embedding redundancy), so the methods would
  stay indistinguishable. The mechanism is addressed by the diagnostic suite
  (§5), not by cleaning the gallery.
- **Pooling is order-invariant by construction**, not merely "selection-robust":
  it maps any keyframe set to an unordered bag and averages, so optical-flow's
  central hypothesis — settled frames carry task structure, a property of frame
  *sequence* — cannot be tested by this aggregator even in principle. This is a
  design property, not a tunable risk. A max-style pooling could in principle be
  more sensitive; the Tier-3 sensitivity analysis (§5.5) shows it is **not** here
  — max-pool and best-match leave the methods equally indistinguishable, so
  order-invariance is not the binding constraint; scene-dominance (§5.1) is.
- **Single dataset / single view / lossy decode.** BridgeData v2 only,
  `image_0` only, AV1-decoded (constant across conditions; see
  `docs/decisions.md`).
- **Natural-K is hyperparameter-dependent.** The ~3-frame budgets reflect the
  current smoothing/threshold settings of each extractor; the ordering of cv_kf
  is stable across tasks but the absolute counts are not a fixed property.
- **AWE not included.** Trajectory-geometry selection (AWE) uses robot state and
  is outside the pure-CV scope unless explicitly approved (`docs/decisions.md`).
- **Saturation is proven, not just argued (§5).** The scene-dominance mechanism
  is now quantitative: intra-episode similarity ≈ the inter-task gap (§5.1); one
  frame suffices and a degenerate block is indistinguishable (§5.2); the oracle
  ceiling sits 10–13 points above every deployable method (§5.3); 0/17 method
  pairs are significant (§5.4); and the invariance survives max- and best-match
  pooling (§5.5). All reuse the cached frame embeddings and add no policy,
  rollout, robot-state signal, or new dataset — in scope for Variant C.

## 7. Artifacts

```
results/eval_retrieval.json            aggregated Top-1/Top-5/CLIP-sim/CR per (method,K)
results/eval_per_demo.jsonl            one record per (episode × extractor)  [24,164 rows]
results/consistency_check_bridge.json  per-task mean_kf / cv_kf / mean_cr
results/plots/fig1_accuracy_vs_cr.*    retrieval accuracy vs CR  (the headline)
results/plots/fig2_clipsim_vs_cr.*     CLIP similarity vs CR
results/plots/fig3_kf_distribution.*   per-demo keyframe-count box plots
results/plots/fig4_consistency.*       cv_kf per extractor across tasks
results/plots/fig5_similarity_distributions.*  intra/inter cosine distributions (§5.1)
results/tables/retrieval_summary.*     full method × K grid (md + csv)
results/tables/retrieval_top1_pivot.*  compact Top-1 grid (md + csv)
results/tables/consistency_aggregate.* per-extractor aggregate (md + csv)
results/tables/similarity_distributions.*  intra/inter similarity medians (§5.1)
results/tables/extra_baselines.*       K=1 / consecutive-block / oracle (§5.2–5.3)
results/tables/retrieval_cis.*         bootstrap 95% CIs on the grid (§5.4)
results/tables/retrieval_permutation.* paired permutation tests (§5.4)
results/tables/pooling_sensitivity.*   mean/max/best-match grid (§5.5)
results/tables/coverage_error.*        coverage error (§5.6)

Diagnostics (numpy-only, read the exported bundle; no GPU):
scripts/diagnostics/bundle.py                  shared loader
scripts/diagnostics/similarity_distributions.py  §5.1 + fig5
scripts/diagnostics/extra_baselines.py         §5.2–5.3
scripts/diagnostics/stats.py                   §5.4
scripts/diagnostics/pooling_sensitivity.py     §5.5
scripts/diagnostics/coverage_error.py          §5.6
```
