# Report restructure outline (narrative-first)

**Status.** Outline only (Task 6). This reorders the *already-measured* content of
`docs/methods.md` (§1–§7) and the pre-registered diagnostics in
`docs/decisions.md` into a claim-driven arc, instead of the current
method-by-method / metric-by-metric exposition. It re-derives **no numbers**:
every figure and table referenced already exists in `results/` (the one
exception, DINOv2, is pre-registered and GPU-blocked and is flagged as pending).
A section-mapping table at the end shows nothing is dropped.

**Arc.** oracle → plateau → mechanism → coverage → (residual / instance / DINOv2)
→ implication upward. The point of the reorder: open with the *puzzle* (the
signal provably exists), then show retrieval can't see it, explain why, then show
the one place selection *is* visible, stress-test that boundary, and close with
the practical compute/compression recommendation — including the frame_diff cost
argument.

---

## 0. Abstract / one-paragraph thesis

- Claim in one breath: *a discriminative frame exists in almost every episode
  (oracle 95.5%), yet unsupervised CV keyframe selection cannot beat uniform or
  random on intrinsic task retrieval, because CLIP frame embeddings are
  scene-saturated and the pooled metric is order-invariant; selection only
  becomes resolvable under a coverage objective and at instance granularity,
  where it rewards even spreading, not content-adaptivity — so on the intrinsic
  metric the rational choice is the cheapest selector.*
- Sets up every section below as evidence for one clause of this sentence.

## 1. Introduction & scope

- Variant C: visual branch only, CV-on-pixels selection, **intrinsic
  retrieval** evaluation. (CLAUDE.md / `docs/brief.md`)
- Explicit out-of-scope fence: no policy training, no rollouts, no task-success,
  no robot-state signals, single dataset (BridgeData V2), single view
  (`image_0`). (carry from methods §1–§3, `docs/decisions.md` Scope reminder)
- Three metrics: Top-1/Top-5 task retrieval, CLIP-text similarity, compression
  ratio K/T — always swept over K ∈ {4,8,16,32}.

## 2. The signal exists — oracle upper bound  *(was §5.3)*

- **Claim:** episodes are not information-poor; a single *label-aware* oracle
  frame retrieves the correct task **95.5%** of the time, greedy K=4 = 0.927 —
  **10–13 pts above** the deployable plateau.
- This is the hook: it converts the rest of the report from "methods tie" into
  "why is recoverable signal lost?"
- Backing: `results/tables/extra_baselines.*` (oracle rows). Cross-ref §5.2
  single-frame controls (one frame already gets ~77%).

## 3. The plateau — retrieval is flat and selection-blind  *(was §4.1, §4.2, §4.3, §5.4, §5.4.1)*

- **Claim:** Top-1 is flat K=4→32 (~0.82) **and** statistically flat across
  methods at every K; no deployable method beats uniform/random.
- Evidence ladder (strongest framing = converging tests, not one p-value):
  - point spread 0.803–0.839, all method×K CIs overlap — `retrieval_cis.*`;
  - **0/40** permutation pairs significant — `retrieval_permutation.*`;
  - TOST equivalence (§5.4.1): differences **bounded to ±0.04**, 7/40 certified
    within ±0.02 (6 at K=32) — `equivalence_tost.*`. State honestly: this is the
    *underpowered* branch; it certifies "indistinguishable, not identical," and
    motivates the 100-task scale-up (pending GPU).
- Headline figure: `fig1_accuracy_vs_cr.*`. Tables: `retrieval_summary.*`,
  `retrieval_top1_pivot.*`.
- Bridge sentence to §4: pushing the budget down to K≈4 costs ~nothing on this
  metric, and the CV methods already operate at that low-budget end.

## 4. The mechanism — why retrieval can't see selection  *(was §5.1, §5.2, §5.5)*

- **Claim:** saturation is a property of the **data + the pool**, not a failure
  of the CV methods.
- Three nested proofs:
  - **Scene dominance** (§5.1): intra-episode cosine **0.917** ≈ inter-episode
    same-task **0.914** ≫ inter-task **0.820** — any subset mean lands in the
    same tiny cone. `similarity_distributions.*`, `fig5_similarity_distributions.*`.
  - **One frame suffices / coverage-blind** (§5.2): single frame ~77%, a
    degenerate consecutive block is indistinguishable — the metric doesn't reward
    coverage. `extra_baselines.*`.
  - **Not a pooling artifact** (§5.5): max-pool and best-match leave methods
    equally tied → order-invariance is *not* the binding constraint;
    scene-dominance is. `pooling_sensitivity.*`.
- Resolves the §2 puzzle: oracle signal is real but diluted by redundancy under
  an order-invariant mean-pool.

## 5. Where selection *does* matter — coverage error  *(was §5.6 + crossover analysis)*

- **Claim:** swap the objective from task-retrieval to embedding-space
  **coverage** and selection becomes resolvable: uniform is best at K≤16, but
  content-adaptive anchors (`attention`, `frame_diff`) **overtake at K=32** (the
  crossover).
- Backing: `coverage_error.*`, `coverage_significance.*`,
  `fig_crossover_velocity.*` (left panel).
- **Mechanism probe (honest null):** the pre-registered velocity-placement
  explanation for the crossover is **INCONCLUSIVE** — at K=32 anchor velocity
  ratio ≈ 1 for all methods (margin +0.011 vs required ≥0.10); one of two
  conditions held (localization), the rule required both. Report as recorded in
  `decisions.md` (2026-06-12); `crossover_analysis.*`. Do **not** over-claim the
  cause.

## 6. Probing the boundary — three pre-registered stress tests

Group the three "is the saturation finding robust?" probes. Each was
pre-registered both-ways in `decisions.md`.

- **6a Residual embeddings — de-saturation FAIL**  *(was §5.7)*
  - **Claim:** subtracting the scene anchor (variant A: e_t − e_0) does **not**
    recover discriminability; residual space is *worse*-separated than raw, so
    saturation is not a removable additive scene offset. Pre-registered negative.
  - `residual_similarity.*`, `fig_residual_similarity.*`.
- **6b Instance-level retrieval — selection-sensitive, favours even spread**  *(was §5.8)*
  - **Claim:** re-label by *episode identity* (within-episode half-split) and
    selection finally matters — **6/16** method-vs-uniform Top-1 comparisons
    significant — but the winner is **uniform / even spreading**; adaptive and
    random are *worse* at low K and converge by K=32. Extends (does not
    contradict) the coverage story: harder task surfaces selection, and it
    rewards spread, not content-adaptivity.
  - `instance_retrieval.*`, `instance_significance.*`.
- **6c DINOv2 cross-backbone — PENDING (GPU-blocked), pre-registered**  *(Task 4)*
  - **Claim (to be filled):** does saturation survive a self-supervised,
    vision-only encoder? Decision rule fixed both ways in `decisions.md`
    (2026-06-12): persists ⇒ data-intrinsic, robust across backbones; breaks ⇒
    CLIP-specific caption/scene artifact, promote DINOv2 to co-primary.
  - Artifacts when run: `dinov2_similarity.*`, `dinov2_retrieval.*`. Flag clearly
    as not-yet-run.

## 7. Implication — what to actually compress with  *(synthesis up from §4.3)*

- **Claim:** on the intrinsic retrieval metric, keyframe selection is a
  **compute / compression-ratio tradeoff, not an accuracy lever** — so pick the
  cheapest selector that meets the budget.
- **The frame_diff compute argument (centerpiece):**
  - Cost asymmetry: `frame_diff` is a near-free **NumPy pixel mean-abs-difference**
    pass — no model, no GPU. `optical_flow` needs a **RAFT-Small** forward pass per
    frame-pair; `attention` needs a **ViT-S/14** forward pass per frame. Two of
    the three CV methods carry a GPU cost the metric never repays.
  - Yet `frame_diff` is (a) statistically **indistinguishable** from all methods
    on retrieval (§3) **and** (b) one of the two methods that **overtakes** uniform
    on coverage error at K=32 (§5). So: if only retrieval matters → **uniform at
    K≈4** is the rational default (zero adaptive cost); if a coverage objective is
    ever wanted → **frame_diff** delivers it at ~zero marginal compute. The
    expensive encoders are dominated on a compute-adjusted basis.
- The genuine, measured advantage of content-adaptive methods is **unsupervised
  low-budget convergence** with episode-length-stable CR (**0.10–0.23** vs fixed-K
  **0.30–0.74**) — a compression-ratio result, not an accuracy result. Keep this
  claim; drop any "smarter selection ⇒ better retrieval" framing.
- **Upward implication for VLA demonstration encoding:** under intrinsic
  retrieval the visual branch's keyframe choice doesn't move accuracy; the burden
  of proof for "smarter selection helps" shifts to **ordered / downstream**
  metrics this protocol cannot see — explicitly out of scope here, named as
  future work, not smuggled in.

## 8. Threats to validity  *(carry §6 intact)*

- Metric saturation & **effective class count ≤ 18** (near-duplicate labels);
  de-dup is not the fix (attacks label confusion, not intra-episode redundancy).
- Order-invariant pooling is a *design property*, not a tunable risk; §5.5 shows
  it isn't the binding constraint anyway.
- Single dataset / single view / lossy AV1 decode (constant across conditions).
- Natural-K is hyperparameter-dependent (ordering stable, absolute counts not).
- AWE excluded (robot-state, out of pure-CV scope without approval).

## 9. Artifacts & reproducibility  *(carry §7)*

- Point to `results/tables/*` (md+csv), `results/plots/*` (pdf+png), and the
  numpy-only `scripts/diagnostics/*` suite that reads the exported bundle
  (`bundle.py`); no-GPU on the diagnostic path; grid read from `bundle_meta.json`.
- Note the 100-task scale-up bundle and the DINOv2 bundle as the two
  GPU-dependent extensions, both pre-registered.

---

## Old → new section map (nothing dropped)

| Current methods.md / decisions.md | New section |
|---|---|
| §1 Dataset, §2 Extractors, §3 Protocol | 1. Introduction & scope |
| §5.3 Oracle upper bound | 2. The signal exists (oracle) |
| §4.1 / §4.2 / §4.3 retrieval grid + synthesis | 3. The plateau |
| §5.4 significance, §5.4.1 TOST equivalence | 3. The plateau (evidence ladder) |
| §5.1 similarity, §5.2 single-frame, §5.5 pooling | 4. The mechanism |
| §5.6 coverage error + crossover_analysis (INCONCLUSIVE) | 5. Where selection matters |
| §5.7 residual gate (FAIL) | 6a Residual probe |
| §5.8 instance retrieval | 6b Instance probe |
| Task 4 DINOv2 pre-registration (decisions.md) | 6c DINOv2 (pending) |
| §4.3 synthesis (forward-looking half) | 7. Implication + frame_diff cost argument |
| §6 Threats to validity | 8. Threats to validity |
| §7 Artifacts | 9. Artifacts & reproducibility |

**Note.** This is a structural plan, not a rewrite. `docs/methods.md` stays the
working draft of record; converting it to this order is a separate writing pass
to be approved before any prose moves.
