# Running experiments on RunPod (A100)

## 0. Before you leave your local machine

Push the latest commits so RunPod can clone them:

```bash
git push origin main
```

---

## 1. Create the pod

In the RunPod web UI:

| Setting | Value |
|---|---|
| GPU | A100 40 GB or 80 GB |
| Template | **RunPod PyTorch 2.1** (CUDA 12.1, Python 3.10, Ubuntu 22.04) |
| Container disk | 20 GB (code + pip cache) |
| Volume disk | 50 GB — mount at `/workspace` (lerobot data + embedding cache) |

> Why 50 GB volume: BridgeData v2 metadata ~3 GB, episode image cache ~15 GB
> (50 eps × 20 tasks), CLIP embedding cache ~500 MB. The volume persists
> between pod restarts; the container disk does not.

Start the pod and copy the SSH command from the **Connect** button.

---

## 2. SSH in and start tmux

```bash
ssh <your-runpod-ssh-command>   # paste from RunPod UI
tmux new -s exp                 # keeps job alive if connection drops
```

> All subsequent commands run inside the tmux session.
> To re-attach after disconnect: `tmux attach -t exp`

---

## 3. Clone the repo

```bash
cd /workspace
git clone https://github.com/DanilovLeo/keyframe-selector.git
cd keyframe-selector
```

---

## 4. Install dependencies

```bash
# PyTorch and torchvision are pre-installed in the RunPod PyTorch template.
# Install the remaining project dependencies:

pip install timm open-clip-torch scipy matplotlib pyyaml

# lerobot — dataset loader for BridgeData v2
pip install lerobot

# Verify key imports
python -c "import timm, open_clip, lerobot; print('imports OK')"
```

If `pip install lerobot` fails with a dependency conflict, try:

```bash
pip install "lerobot[dev]" --no-deps
pip install datasets huggingface_hub
```

---

## 5. Run the preflight check

This must pass before you launch the full sweep.
It will download BridgeData v2 metadata on first run (~3 GB, takes ~5–10 min).

```bash
python scripts/preflight_check.py \
    --root /workspace/lerobot_cache \
    --embed_cache /workspace/clip_embeds \
    --min_demos 5 \
    --n_check_eps 5
```

Expected output ends with:
```
  [PASS]  cuda
  [PASS]  model_cfg
  [PASS]  cache
  [PASS]  split
  [PASS]  real_eps
  All checks passed — safe to launch run_retrieval_eval.py
```

**Read the natural-K table printed for `optical_flow` and `attention_dino`.**
If either shows `max/min > 5×`, the variable-K methods span too wide a range
for a clean comparison — come back and discuss before launching the full sweep.

---

## 6. Run the experiments

```bash
python scripts/run_retrieval_eval.py \
    --root /workspace/lerobot_cache \
    --embed_cache /workspace/clip_embeds \
    --min_demos 20 \
    --max_tasks 20 \
    --max_episodes 50 \
    --output_dir /workspace/results
```

Expected wall time on A100: **45–75 minutes**.

Progress is printed per-task. If the run is interrupted, re-running
**skips CLIP inference** for any already-cached episodes and resumes quickly.

Also run the consistency check for keyframe count stats:

```bash
python scripts/run_consistency.py \
    --root /workspace/lerobot_cache \
    --min_demos 20 \
    --max_tasks 20 \
    --max_episodes 50 \
    --output results/consistency_check_bridge.json
```

---

## 7. Download results

Results are in `/workspace/results/`:

```
eval_retrieval.json      — aggregated metrics (Top-1, Top-5, CLIP-sim, CR)
eval_per_demo.jsonl      — one line per episode × extractor (for analysis)
consistency_check_bridge.json  — mean_kf, cv_kf, mean_cr per extractor
```

Download with `scp` (replace port and host with your RunPod SSH details):

```bash
# Run this on your LOCAL machine (new terminal, not inside tmux)
scp -P <port> root@<host>:/workspace/results/eval_retrieval.json .
scp -P <port> root@<host>:/workspace/results/eval_per_demo.jsonl .
scp -P <port> root@<host>:/workspace/results/consistency_check_bridge.json .
```

Or use the RunPod web file browser (pod page → Files tab).

---

## 8. Stop the pod

Once results are downloaded, stop (not terminate) the pod to pause billing.
The `/workspace` volume persists — if you need to re-run, restart the same pod
and the CLIP embedding cache will already be warm.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: lerobot` | `pip install lerobot` |
| `RuntimeError: CUDA out of memory` | Reduce `--max_episodes` to 20 or batch_size in extractors |
| Task index build hangs | It's scanning ~9M parquet rows — wait 10 min, it will finish and cache |
| `preflight_check.py` reports K range > 5× | See §5 note — discuss before full sweep |
| SSH disconnects mid-run | Re-attach with `tmux attach -t exp` — job is still running |
