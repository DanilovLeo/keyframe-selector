# Keyframe Selection for VLA Fine-Tuning

This project benchmarks keyframe extraction methods for robot manipulation
demonstrations, targeting Vision-Language-Action (VLA) model fine-tuning.
The core hypothesis — supported by NoTVLA, AWE, and SPHINX — is that sparse
keyframes better match VLM pre-training granularity than dense trajectories
and can improve generalisation while reducing compute. We compare four
extraction strategies (uniform subsampling, near-zero velocity, gripper-state
transitions, and AWE linear approximation) across LIBERO task suites,
sweeping compression ratios and measuring downstream task success rate.

## Setup

```bash
conda create -n keyframe python=3.10
conda activate keyframe
pip install -r requirements.txt
```

Link your LIBERO demo data:

```bash
ln -s /path/to/libero/datasets data
```

## Project layout

```
src/
  extractors/   keyframe extraction methods
  eval/         metrics (compression ratio, coverage, phase coverage)
  utils/        HDF5 loader, trajectory visualisation
notebooks/
  01_explore_demo.ipynb   — run all extractors on one demo, compare visually
results/                  — saved plots (gitignored)
data/                     — symlink to LIBERO demos (gitignored)
```

## Quick start

```bash
conda activate keyframe
jupyter notebook notebooks/01_explore_demo.ipynb
```

Update `DEMO_PATH` in the notebook to point at a LIBERO `.hdf5` file.
