> Paired TOST equivalence on Top-1 (diff = method_a - method_b). Equivalent = 90% CI within +/-delta. See docs/decisions.md (2026-06-12).

| K | method_a | method_b | diff_top1 | se | ci90_lo | ci90_hi | p_tost | equivalent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | uniform | random | -0.0169 | 0.0153 | -0.0422 | 0.0085 | 0.4188 | no |
| 4 | uniform | optical_flow | 0.0056 | 0.0149 | -0.0190 | 0.0303 | 0.1679 | no |
| 4 | uniform | attention | -0.0112 | 0.0138 | -0.0340 | 0.0115 | 0.2627 | no |
| 4 | uniform | frame_diff | -0.0281 | 0.0186 | -0.0588 | 0.0026 | 0.6682 | no |
| 4 | random | optical_flow | 0.0225 | 0.0163 | -0.0045 | 0.0494 | 0.5602 | no |
| 4 | random | attention | 0.0056 | 0.0147 | -0.0186 | 0.0299 | 0.1640 | no |
| 4 | random | frame_diff | -0.0112 | 0.0176 | -0.0403 | 0.0179 | 0.3095 | no |
| 4 | optical_flow | attention | -0.0169 | 0.0149 | -0.0414 | 0.0077 | 0.4162 | no |
| 4 | optical_flow | frame_diff | -0.0337 | 0.0176 | -0.0629 | -0.0045 | 0.7810 | no |
| 4 | attention | frame_diff | -0.0169 | 0.0203 | -0.0504 | 0.0167 | 0.4384 | no |
| 8 | uniform | random | -0.0243 | 0.0119 | -0.0440 | -0.0047 | 0.6424 | no |
| 8 | uniform | optical_flow | -0.0112 | 0.0195 | -0.0435 | 0.0210 | 0.3268 | no |
| 8 | uniform | attention | 0.0056 | 0.0187 | -0.0253 | 0.0365 | 0.2212 | no |
| 8 | uniform | frame_diff | -0.0169 | 0.0169 | -0.0447 | 0.0110 | 0.4261 | no |
| 8 | random | optical_flow | 0.0131 | 0.0171 | -0.0151 | 0.0414 | 0.3435 | no |
| 8 | random | attention | 0.0300 | 0.0179 | 0.0004 | 0.0595 | 0.7110 | no |
| 8 | random | frame_diff | 0.0075 | 0.0155 | -0.0181 | 0.0331 | 0.2100 | no |
| 8 | optical_flow | attention | 0.0169 | 0.0232 | -0.0215 | 0.0552 | 0.4461 | no |
| 8 | optical_flow | frame_diff | -0.0056 | 0.0246 | -0.0462 | 0.0350 | 0.2794 | no |
| 8 | attention | frame_diff | -0.0225 | 0.0177 | -0.0518 | 0.0069 | 0.5553 | no |
| 16 | uniform | random | 0.0075 | 0.0079 | -0.0057 | 0.0206 | 0.0586 | no |
| 16 | uniform | optical_flow | 0.0112 | 0.0159 | -0.0151 | 0.0375 | 0.2912 | no |
| 16 | uniform | attention | -0.0056 | 0.0126 | -0.0264 | 0.0152 | 0.1274 | no |
| 16 | uniform | frame_diff | -0.0056 | 0.0149 | -0.0303 | 0.0190 | 0.1679 | no |
| 16 | random | optical_flow | 0.0037 | 0.0122 | -0.0164 | 0.0239 | 0.0916 | no |
| 16 | random | attention | -0.0131 | 0.0097 | -0.0292 | 0.0029 | 0.2394 | no |
| 16 | random | frame_diff | -0.0131 | 0.0126 | -0.0339 | 0.0077 | 0.2919 | no |
| 16 | optical_flow | attention | -0.0169 | 0.0125 | -0.0376 | 0.0039 | 0.4010 | no |
| 16 | optical_flow | frame_diff | -0.0169 | 0.0125 | -0.0376 | 0.0039 | 0.4010 | no |
| 16 | attention | frame_diff | 0.0000 | 0.0113 | -0.0186 | 0.0186 | 0.0388 | yes |
| 32 | uniform | random | 0.0019 | 0.0019 | -0.0012 | 0.0050 | 0.0000 | yes |
| 32 | uniform | optical_flow | 0.0112 | 0.0079 | -0.0019 | 0.0243 | 0.1351 | no |
| 32 | uniform | attention | 0.0000 | 0.0080 | -0.0132 | 0.0132 | 0.0065 | yes |
| 32 | uniform | frame_diff | 0.0056 | 0.0056 | -0.0037 | 0.0149 | 0.0057 | yes |
| 32 | random | optical_flow | 0.0094 | 0.0082 | -0.0041 | 0.0228 | 0.0969 | no |
| 32 | random | attention | -0.0019 | 0.0082 | -0.0154 | 0.0117 | 0.0140 | yes |
| 32 | random | frame_diff | 0.0037 | 0.0059 | -0.0061 | 0.0136 | 0.0034 | yes |
| 32 | optical_flow | attention | -0.0112 | 0.0112 | -0.0298 | 0.0073 | 0.2182 | no |
| 32 | optical_flow | frame_diff | -0.0056 | 0.0097 | -0.0217 | 0.0105 | 0.0710 | no |
| 32 | attention | frame_diff | 0.0056 | 0.0056 | -0.0037 | 0.0149 | 0.0057 | yes |

```
TOST equivalence at delta=0.02, 90% CI, n_queries=178 (docs/decisions.md 2026-06-12):
  pairs equivalent within +/-0.02: 7 / 40
  90% CI half-width: median 0.0235, max 0.0406
  => UNDERPOWERED for +/-0.02: achievable bound ~+/-0.041 (needs more queries)
```
