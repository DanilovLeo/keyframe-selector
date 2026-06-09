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
(0.839), i.e. noise. CLIP similarity is essentially constant at **~0.209–0.210**
across all methods and all K (`fig2_clipsim_vs_cr`).

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
  (CR ≈ 0.14–0.16) versus the fixed-10 references' ~49%, i.e. **~3× harder
  compression**. Their CR is also *stable across tasks* (0.10–0.23), whereas
  uniform-10's CR swings 0.30 → 0.74 purely with episode length — it keeps half
  the frames of a short reaching demo but a third of a long closing demo, with
  no content awareness.
- **Count consistency (cv_kf).** Do **not** compare adaptive vs fixed-K here:
  uniform/random sit at cv ≈ 0.005 *by construction* (they ignore content), not
  by merit. Among the three adaptive methods, `frame_diff` (0.175) ≈ `attention`
  (0.187) < `optical_flow` (0.219); frame_diff/attention pick a more predictable
  number of keyframes per task, though the task-to-task spread overlaps (see the
  error bars in fig4).

### 4.3 Synthesis

Combining the two: retrieval is **flat from K=4 to K=32**, so pushing the
keyframe budget down costs almost nothing on this metric; and the CV methods
*naturally* operate at that low-budget end (~3 frames). The defensible claim is
therefore:

> **Content-adaptive selection compresses demonstrations ~3× more aggressively
> than fixed-K uniform/random while intrinsic retrieval accuracy is preserved**
> — but it does **not** beat uniform or random at a matched budget, because the
> retrieval metric is scene-saturated and cannot resolve selection quality.

This is a legitimate negative/neutral result for an intrinsic-evaluation study,
not a failure of the CV methods.

---

## 5. Threats to validity

- **Metric saturation.** Top-5 ≈ 0.95 and the 20-task gallery include
  near-duplicate instructions ("Close the drawer" / "close the drawer" /
  "closed the drawer" / "close drawer of box"), making retrieval easy. A larger,
  finer-grained gallery might separate methods; this run cannot rule that in.
- **Pooling choice.** Mean-pooling is selection-robust by design and may be
  washing out differences. A sequence-aware or max-style pooling could be more
  sensitive. The pooling is the pinned protocol (`CLAUDE.md`) — changing it is a
  deliberate decision, not a silent default, and is **not** done here.
- **Single dataset / single view / lossy decode.** BridgeData v2 only,
  `image_0` only, AV1-decoded (constant across conditions; see
  `docs/decisions.md`).
- **Natural-K is hyperparameter-dependent.** The ~3-frame budgets reflect the
  current smoothing/threshold settings of each extractor; the ordering of cv_kf
  is stable across tasks but the absolute counts are not a fixed property.
- **AWE not included.** Trajectory-geometry selection (AWE) uses robot state and
  is outside the pure-CV scope unless explicitly approved (`docs/decisions.md`).

## 6. Artifacts

```
results/eval_retrieval.json            aggregated Top-1/Top-5/CLIP-sim/CR per (method,K)
results/eval_per_demo.jsonl            one record per (episode × extractor)  [24,164 rows]
results/consistency_check_bridge.json  per-task mean_kf / cv_kf / mean_cr
results/plots/fig1_accuracy_vs_cr.*    retrieval accuracy vs CR  (the headline)
results/plots/fig2_clipsim_vs_cr.*     CLIP similarity vs CR
results/plots/fig3_kf_distribution.*   per-demo keyframe-count box plots
results/plots/fig4_consistency.*       cv_kf per extractor across tasks
results/tables/retrieval_summary.*     full method × K grid (md + csv)
results/tables/retrieval_top1_pivot.*  compact Top-1 grid (md + csv)
results/tables/consistency_aggregate.* per-extractor aggregate (md + csv)
```
