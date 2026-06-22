> Paired TOST equivalence on Top-1 (diff = method_a - method_b). Equivalent = 90% CI within +/-delta. See docs/decisions.md (2026-06-12).

| K | method_a | method_b | diff_top1 | se | ci90_lo | ci90_hi | p_tost | equivalent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | uniform | random | 0.0131 | 0.0131 | -0.0086 | 0.0348 | 0.2999 | no |
| 4 | uniform | optical_flow | 0.0337 | 0.0176 | 0.0045 | 0.0629 | 0.7810 | no |
| 4 | uniform | attention | 0.0056 | 0.0203 | -0.0280 | 0.0392 | 0.2399 | no |
| 4 | uniform | frame_diff | 0.0056 | 0.0149 | -0.0190 | 0.0303 | 0.1679 | no |
| 4 | random | optical_flow | 0.0206 | 0.0131 | -0.0010 | 0.0422 | 0.5183 | no |
| 4 | random | attention | -0.0075 | 0.0176 | -0.0366 | 0.0216 | 0.2392 | no |
| 4 | random | frame_diff | -0.0075 | 0.0143 | -0.0311 | 0.0161 | 0.1913 | no |
| 4 | optical_flow | attention | -0.0281 | 0.0202 | -0.0615 | 0.0053 | 0.6553 | no |
| 4 | optical_flow | frame_diff | -0.0281 | 0.0186 | -0.0588 | 0.0026 | 0.6682 | no |
| 4 | attention | frame_diff | 0.0000 | 0.0138 | -0.0228 | 0.0228 | 0.0745 | no |
| 8 | uniform | random | 0.0318 | 0.0118 | 0.0123 | 0.0513 | 0.8417 | no |
| 8 | uniform | optical_flow | 0.0112 | 0.0178 | -0.0182 | 0.0407 | 0.3115 | no |
| 8 | uniform | attention | 0.0225 | 0.0177 | -0.0069 | 0.0518 | 0.5553 | no |
| 8 | uniform | frame_diff | 0.0169 | 0.0186 | -0.0140 | 0.0477 | 0.4331 | no |
| 8 | random | optical_flow | -0.0206 | 0.0151 | -0.0455 | 0.0043 | 0.5158 | no |
| 8 | random | attention | -0.0094 | 0.0158 | -0.0355 | 0.0168 | 0.2510 | no |
| 8 | random | frame_diff | -0.0150 | 0.0180 | -0.0447 | 0.0147 | 0.3902 | no |
| 8 | optical_flow | attention | 0.0112 | 0.0178 | -0.0182 | 0.0407 | 0.3115 | no |
| 8 | optical_flow | frame_diff | 0.0056 | 0.0218 | -0.0305 | 0.0417 | 0.2553 | no |
| 8 | attention | frame_diff | -0.0056 | 0.0218 | -0.0417 | 0.0305 | 0.2553 | no |
| 16 | uniform | random | 0.0000 | 0.0070 | -0.0116 | 0.0116 | 0.0025 | yes |
| 16 | uniform | optical_flow | 0.0112 | 0.0112 | -0.0073 | 0.0298 | 0.2182 | no |
| 16 | uniform | attention | 0.0000 | 0.0138 | -0.0228 | 0.0228 | 0.0745 | no |
| 16 | uniform | frame_diff | 0.0000 | 0.0113 | -0.0186 | 0.0186 | 0.0388 | yes |
| 16 | random | optical_flow | 0.0112 | 0.0124 | -0.0093 | 0.0318 | 0.2408 | no |
| 16 | random | attention | -0.0000 | 0.0141 | -0.0232 | 0.0232 | 0.0782 | no |
| 16 | random | frame_diff | -0.0000 | 0.0116 | -0.0191 | 0.0191 | 0.0429 | yes |
| 16 | optical_flow | attention | -0.0112 | 0.0112 | -0.0298 | 0.0073 | 0.2182 | no |
| 16 | optical_flow | frame_diff | -0.0112 | 0.0138 | -0.0340 | 0.0115 | 0.2627 | no |
| 16 | attention | frame_diff | 0.0000 | 0.0113 | -0.0186 | 0.0186 | 0.0388 | yes |
| 32 | uniform | random | 0.0056 | 0.0042 | -0.0013 | 0.0125 | 0.0004 | yes |
| 32 | uniform | optical_flow | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | yes |
| 32 | uniform | attention | 0.0056 | 0.0056 | -0.0037 | 0.0149 | 0.0057 | yes |
| 32 | uniform | frame_diff | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | yes |
| 32 | random | optical_flow | -0.0056 | 0.0042 | -0.0125 | 0.0013 | 0.0004 | yes |
| 32 | random | attention | -0.0000 | 0.0027 | -0.0044 | 0.0044 | 0.0000 | yes |
| 32 | random | frame_diff | -0.0056 | 0.0042 | -0.0125 | 0.0013 | 0.0004 | yes |
| 32 | optical_flow | attention | 0.0056 | 0.0056 | -0.0037 | 0.0149 | 0.0057 | yes |
| 32 | optical_flow | frame_diff | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | yes |
| 32 | attention | frame_diff | -0.0056 | 0.0056 | -0.0149 | 0.0037 | 0.0057 | yes |

```
TOST equivalence at delta=0.02, 90% CI, n_queries=178 (docs/decisions.md 2026-06-12):
  pairs equivalent within +/-0.02: 14 / 40
  90% CI half-width: median 0.0222, max 0.0361
  => UNDERPOWERED for +/-0.02: achievable bound ~+/-0.036 (needs more queries)
```
