| variant | distribution | n | mean | std | p05 | p25 | p50 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | intra_episode_frame_pairs | 1476528 | 0.918 | 0.046 | 0.831 | 0.893 | 0.927 | 0.953 | 0.977 |
| raw | inter_episode_same_task_means | 96909 | 0.917 | 0.054 | 0.812 | 0.888 | 0.931 | 0.957 | 0.982 |
| raw | inter_task_means | 9536946 | 0.788 | 0.069 | 0.664 | 0.744 | 0.796 | 0.837 | 0.889 |
| A | intra_episode_frame_pairs | 1370694 | 0.501 | 0.234 | 0.083 | 0.347 | 0.524 | 0.679 | 0.843 |
| A | inter_episode_same_task_means | 96909 | 0.145 | 0.229 | -0.231 | -0.012 | 0.142 | 0.302 | 0.532 |
| A | inter_task_means | 9536946 | 0.052 | 0.159 | -0.208 | -0.054 | 0.050 | 0.157 | 0.318 |
| B | intra_episode_frame_pairs | 1476528 | -0.032 | 0.339 | -0.521 | -0.292 | -0.073 | 0.195 | 0.587 |
| B | inter_episode_same_task_means | 0 | nan | nan | nan | nan | nan | nan | nan |
| B | inter_task_means | 0 | nan | nan | nan | nan | nan | nan | nan |

```
Pre-registered gate (variant A; docs/decisions.md 2026-06-12):
  median(intra_A)      = 0.524
  median(same_task_A)  = 0.142
  median(inter_task_A) = 0.050
  cond1  intra <= inter_task - 0.05:  0.524 <= 0.000  -> False
  cond2  same - inter_task >= 0.02:    +0.092 >= 0.02  -> True
  VERDICT: FAIL
```
