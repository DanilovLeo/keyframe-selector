# SMOKE_RUNBOOK.md

Read-only analysis of the BridgeData v2 retrieval eval, plus a minimal smoke-run
plan. **Branch detected: (C) DOWNLOAD-THEN-MATERIALIZE.** This runbook covers only
that branch.

> Nothing in this document has been executed. No pod was started, no dataset
> downloaded, no eval run, no source file edited.

---

## 0. Verdict (the linchpin)

**(C)** — `src/data/bridge_loader.py` uses a **non-streaming** HuggingFace
`load_dataset(...)`. The entire `lerobot/bridgedata_v2` train split is downloaded
and materialized, and a full ~9M-row index scan runs, **before episode 0 is
readable**.

Deciding lines (`src/data/bridge_loader.py`):

```python
51  from datasets import load_dataset
55  kwargs: dict = {"split": "train"}
56  if self._root is not None:
57      kwargs["cache_dir"] = str(self._root)
60  self._ds = load_dataset(dataset_name, **kwargs)   # ← NO streaming=True
```

- There is **no** `streaming=True` and **no** per-episode lazy fetch → not (A).
- It is HF `datasets` parquet materialized as one Arrow table, accessed via
  `self._ds.select(range(start, end))` (line 93) on the already-downloaded table —
  not independently readable tfrecord/tfds shards → not (B).
- The full split must land and the full index must build first → **(C)**.

The index build forces a whole-dataset scan:

```python
62  self._task_to_episodes, self._episode_slices = _build_indexes(self._ds, self._root)
136 ep_col   = ds["episode_index"]   # whole column, one per frame
174 def _read_task_column(ds): ... ds["task"] / ds["task_index"]  # whole column
93  episode_ds = self._ds.select(range(start, end))   # operates on the full table
```

> Mechanism nuance: this is **not** the literal `demos_8_17.zip` / `gsutil cp` /
> `bridgedata_raw_to_numpy` / `*.npy` pipeline named in category C. It is a
> non-streaming `load_dataset`. The **consequence is identical**: no episode is
> accessible until the whole split is downloaded and indexed.

---

## 1. Can a smoke run read ~20 episodes WITHOUT the full dataset?

**No.**

1. `load_dataset("lerobot/bridgedata_v2", split="train")` without `streaming=True`
   downloads **all** parquet shards of the train split before returning
   (`bridge_loader.py:60`).
2. `_build_indexes` then reads the **entire** `episode_index` and `task` columns
   across ~9M frame rows (`bridge_loader.py:136, 174`). `RUNPOD.md:164` confirms:
   *"It's scanning ~9M parquet rows — wait 10 min."*
3. Only after (1) and (2) can `load_episode` `.select()` episode 0
   (`bridge_loader.py:83, 92-93`).

**Minimum bytes before episode 0 is accessible:** the **entire `lerobot/bridgedata_v2`
train split** (all parquet shards). There is no documented per-shard or per-episode
subset path in this loader.

> ⚠️ Size discrepancy to resolve before provisioning. `RUNPOD.md:24` claims
> *"BridgeData v2 metadata ~3 GB"* and provisions a 50 GB volume. But a
> **non-streaming** `load_dataset` pulls the image-bearing parquet, i.e. the full
> split, not 3 GB of metadata. Upstream BridgeData v2 is ~150 GB+ raw; the lerobot
> parquet repacking may be tens of GB. **Verify the actual `lerobot/bridgedata_v2`
> size on the HF Hub page before sizing the volume — 50 GB may be too small.**

**Consequence for "smoke":** restricting tasks/episodes only reduces CLIP/extraction
compute. It does **not** reduce the data download. The smoke run still pays the full
download + full index scan up front.

---

## 2. Run knobs (exact file:line)

### Set in `scripts/run_retrieval_eval.py`

| Knob | Where set | Where applied | Value |
|---|---|---|---|
| tasks | `--max_tasks` default 20 — `:241` | `:254` `list_tasks(...)[:args.max_tasks]` | 20 |
| episodes/task | `--max_episodes` default 50 — `:242` | `eval_task` `:125` `list_episodes(task)[:max_episodes]` | 50 |
| min demos/task | `--min_demos` default 20 — `:240` | `:254` `list_tasks(min_demos=...)` | 20 |
| K-sweep | **hardcoded** `:85` `K_SWEEP = [4, 8, 16, 32]` | `:92, :94, :212, :305-306` | **[4, 8, 16, 32]** ✅ |
| random seeds | **hardcoded** `:86` `RANDOM_SEEDS = [42, 123, 456]` | `:95, :213` | **[42, 123, 456]** ✅ |

> ⚠️ `CLAUDE.md` states the K sweep and seeds live in `configs/experiment.yaml`.
> **That file does not exist** — `configs/` contains only `models.yaml`. K and seeds
> are hardcoded in `run_retrieval_eval.py:85-86` (and **duplicated** in
> `scripts/plot_results.py:64` and `:143`). If you change one, change all three.

### Restrict to a smoke run (~2–3 tasks × ~20 episodes)

**Existing flags — no code change needed:**

```
--max_tasks 3  --max_episodes 20  --min_demos 20
```

`--max_tasks` slices at `:254`; `--max_episodes` slices at `eval_task:125`.

### Reported-but-do-not-fix items

- **`list_tasks()` ordering** — `bridge_loader.py:70-74`. **Already wrapped in
  `sorted()` at `:72`.** Its output is deterministic (alphabetical) in this revision,
  so the "needs `sorted()`" concern **does not apply here**. No fix required.
  (Episode order from `list_episodes` `:78` is scan-order, then shuffled
  deterministically by `gallery_query_split(seed=42)` in `retrieval.py:117-131`.)
- **Hardcoded relative config path** — `run_retrieval_eval.py:73`
  `_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "models.yaml"`.
  Same pattern in `preflight_check.py:177` and `run_consistency.py:158`. These are
  anchored to `__file__`, so they are **cwd-robust** (work from any launch dir), but
  the filename `models.yaml` is fixed — you cannot point at an alternate config
  without editing. Reported only; not changed.

---

## 3. Smoke-run plan (Branch C)

### 3.1 Volume decision — PROVISION A PERSISTENT NETWORK VOLUME FIRST

Because the **full split download is unavoidable even for a smoke run** (§1), it must
**not** live on ephemeral container disk (lost on stop/terminate → re-download).

- Mount a **persistent RunPod network volume at `/workspace`**.
- Point both caches at it (already the case in `setup_runpod.sh:7-8`):
  - `--root /workspace/lerobot_cache`  (HF dataset cache)
  - `--embed_cache /workspace/clip_embeds`  (CLIP frame-embedding cache)
- **Size:** full dataset + CLIP embed cache. `RUNPOD.md:22` says 50 GB; **verify the
  real `lerobot/bridgedata_v2` size first** and grow the volume if needed (§1 warning).
- **Storage cost:** RunPod network volume ≈ **$0.05–0.10 / GB / month**, billed while
  the volume exists (even when the pod is stopped). A 100 GB volume ≈ $5–10/mo.

### 3.2 Pod spec

| Setting | Value |
|---|---|
| GPU | A100 40 GB (smoke does not need 80 GB) |
| Template | RunPod **PyTorch 2.1** (CUDA 12.1, Python 3.10, Ubuntu 22.04) |
| Container disk | 20 GB |
| Network volume | persistent, mounted at `/workspace` (sized per §3.1) |

### 3.3 Commands, in order

```bash
# 0. Local: push commits so the pod can clone (RUNPOD.md:8)
git push origin main

# 1. On the pod: clone (or pull) into the persistent volume
cd /workspace
git clone https://github.com/DanilovLeo/keyframe-selector.git
cd keyframe-selector
tmux new -s smoke        # survive SSH drops

# 2. Dependencies
bash scripts/setup_runpod.sh
#   ⚠️ setup_runpod.sh:15,23 installs and imports `lerobot`. The loader does NOT
#   need lerobot (bridge_loader uses `datasets` directly). If the lerobot import at
#   setup_runpod.sh:23 fails but `datasets`/`open_clip`/`timm` imported, it is safe
#   to proceed. (RUNPOD.md:60 uses a lighter install without lerobot.)

# 3. Preflight — THIS triggers the full download + ~9M-row index build
#    (BridgeDataLoader(root=...) at preflight_check.py:195). Expect the long wait here.
python scripts/preflight_check.py \
    --root /workspace/lerobot_cache \
    --embed_cache /workspace/clip_embeds \
    --min_demos 5 \
    --n_check_eps 5
#    Must end with "All checks passed". Also read the natural-K table for
#    optical_flow / attention_dino; if max/min > 5× it warns (RUNPOD.md:91-93).

# 4. Smoke eval — 3 tasks × 20 episodes (index + dataset already cached from step 3)
python scripts/run_retrieval_eval.py \
    --root /workspace/lerobot_cache \
    --embed_cache /workspace/clip_embeds \
    --min_demos 20 \
    --max_tasks 3 \
    --max_episodes 20 \
    --output_dir /workspace/results
```

### 3.4 Outputs

`run_retrieval_eval.py:328, 346` write to `--output_dir`:

- `/workspace/results/eval_retrieval.json` — aggregated metrics per extractor.
- `/workspace/results/eval_per_demo.jsonl` — one line per (episode × extractor).

Model weights pulled on first run (separate from the dataset, into the HF/torch
caches): CLIP ViT-L/14 `openai` (open_clip), DINOv2 ViT-S/14 (timm,
`configs/models.yaml:10`), RAFT-small (torchvision, used by the optical-flow
extractor).

### 3.5 Estimated wall-clock and cost

The smoke run is **dominated by the one-time full download + index build**, not by
the 3×20 compute:

| Phase | Estimate |
|---|---|
| Full split download | network-bound; minutes to a few hours depending on real dataset size (§1) |
| ~9M-row index build | ~10 min (`RUNPOD.md:164`), cached afterward |
| CLIP embed + extract, 3 tasks × 20 eps | ~5–10 min on A100 (~6% of the 20×50 full run, which is 45–75 min per `RUNPOD.md:109`) |

- **Wall-clock:** ~20–40 min if the real download is small; **hours** if the split is
  the full ~150 GB-class dataset. Resolve §1 size first to tighten this.
- **Dollar cost:** A100 ≈ **$1.5–2.5/hr** on RunPod → smoke ≈ **$2–6** (plus the
  persistent volume's monthly storage), **dominated by download time**.

---

## 4. GO / STOP gate

Inspect `/workspace/results/eval_retrieval.json` (+ `eval_per_demo.jsonl`) before
committing to the full 20×50 run. **GO only if all hold:**

1. **Ceiling sanity (densest proxy).** ⚠️ The grid has **no "all-frames" extractor** —
   `build_extractor_grid` (`run_retrieval_eval.py:89-101`) emits only
   uniform/random/optical_flow/attention. Use the **densest proxy `uniform_k32`**:
   its `top_1` must be **strictly between chance and 1.0**. With 3 balanced tasks,
   chance ≈ 1/3 ≈ 0.33, so expect `0.33 < top_1 < 1.0`. (A true all-frames ceiling
   would require adding an extractor — a source edit, out of scope for this smoke.)
2. **Distinguishable methods.** uniform / random / optical_flow / attention_dino must
   **not** produce identical `top_1` / `clip_sim` — the curves should separate.
3. **Frame-count consistency at each K.** For each K ∈ {4,8,16,32}, `uniform_kK` and
   every `random_kK_s{42,123,456}` must report the **same `mean_n_kf` (= K)**.
   - Caveat: for short episodes where `T ≤ K`, both extractors return all `T` frames
     by design (`uniform.py:37-38`, `random_extractor.py:45-46`), so `mean_n_kf` can
     dip below K — acceptable if it matches between uniform and random.
   - `optical_flow` / `attention_dino` use **natural (variable) K** and are **not**
     tied to the sweep — this check is uniform-vs-random per K, not across CV methods.
4. **No NaNs** anywhere in `eval_retrieval.json` (top_1, top_5, clip_sim, mean_cr,
   mean_n_kf, and the random `*_std` fields).
5. **All cells populated.** `results` must contain `uniform_k{4,8,16,32}`,
   aggregated `random_k{4,8,16,32}` **and** per-seed `random_k{K}_s{42,123,456}`,
   plus `optical_flow` and `attention_dino`.

If any fail → **STOP**, diagnose on the cheap 3-task run, do not pay for the full
sweep. Note a 3-task smoke yields only ~12 query items, so Recall@1 is noisy — treat
this gate as a plumbing/sanity check, not a result.

---

## Verdict (restated)

**(C) DOWNLOAD-THEN-MATERIALIZE** — non-streaming `load_dataset` requires the full
`lerobot/bridgedata_v2` split on disk before episode 0.

**Cheapest path to a real `eval_retrieval.json`:** provision a persistent
`/workspace` volume, pay the one-time full download + index build during
`preflight_check.py`, then run `run_retrieval_eval.py --max_tasks 3 --max_episodes 20
--min_demos 20` — the download dominates cost, so the only real lever is reusing that
cached volume rather than shrinking the eval.
```
