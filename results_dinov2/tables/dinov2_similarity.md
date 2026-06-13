> DINOv2 saturation distributions (intra/same-task/inter-task cosine). Pre-registered cross-encoder check; see docs/decisions.md (2026-06-12).

| distribution | n | mean | std | p05 | p25 | p50 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| intra_episode_frame_pairs | 277809 | 0.8107 | 0.1369 | 0.5420 | 0.7304 | 0.8400 | 0.9193 | 0.9784 |
| inter_episode_same_task_means | 18912 | 0.8026 | 0.1448 | 0.5507 | 0.6874 | 0.8228 | 0.9397 | 0.9757 |
| inter_task_means | 353041 | 0.4809 | 0.1808 | 0.2267 | 0.3420 | 0.4614 | 0.5913 | 0.8553 |

```
DINOv2 retrieval pass (docs/decisions.md 2026-06-12), backbone=dinov2, dim=384, n_queries=178:
  intra=0.840  same-task=0.823  inter-task=0.461  (intra~same within 0.05: True; task gap same-inter = +0.361)
  method-pair Top-1 permutation: 1 / 40 significant at p<0.05
  VERDICT: saturation BREAKS (backbone-specific)
```
