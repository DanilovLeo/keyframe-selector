| variant | distribution | n | mean | std | p05 | p25 | p50 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | intra_episode_frame_pairs | 277809 | 0.907 | 0.052 | 0.808 | 0.877 | 0.917 | 0.946 | 0.974 |
| raw | inter_episode_same_task_means | 18912 | 0.906 | 0.056 | 0.801 | 0.873 | 0.914 | 0.952 | 0.979 |
| raw | inter_task_means | 353041 | 0.819 | 0.056 | 0.727 | 0.784 | 0.820 | 0.855 | 0.914 |
| A | intra_episode_frame_pairs | 257795 | 0.502 | 0.238 | 0.077 | 0.343 | 0.525 | 0.683 | 0.850 |
| A | inter_episode_same_task_means | 18912 | 0.165 | 0.217 | -0.197 | 0.021 | 0.164 | 0.312 | 0.526 |
| A | inter_task_means | 353041 | 0.070 | 0.166 | -0.200 | -0.041 | 0.069 | 0.181 | 0.345 |
| B | intra_episode_frame_pairs | 277809 | -0.033 | 0.352 | -0.538 | -0.307 | -0.077 | 0.207 | 0.611 |
| B | inter_episode_same_task_means | 0 | nan | nan | nan | nan | nan | nan | nan |
| B | inter_task_means | 0 | nan | nan | nan | nan | nan | nan | nan |

```
Pre-registered gate (variant A; docs/decisions.md 2026-06-12):
  median(intra_A)      = 0.525
  median(same_task_A)  = 0.164
  median(inter_task_A) = 0.069
  cond1  intra <= inter_task - 0.05:  0.525 <= 0.019  -> False
  cond2  same - inter_task >= 0.02:    +0.095 >= 0.02  -> True
  VERDICT: FAIL
```
