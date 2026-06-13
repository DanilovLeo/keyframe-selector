> Paired TOST equivalence on Top-1 (diff = method_a - method_b). Equivalent = 90% CI within +/-delta. See docs/decisions.md (2026-06-12).

| K | method_a | method_b | diff_top1 | se | ci90_lo | ci90_hi | p_tost | equivalent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | uniform | random | 0.0085 | 0.0065 | -0.0021 | 0.0192 | 0.0386 | yes |
| 4 | uniform | optical_flow | 0.0011 | 0.0070 | -0.0103 | 0.0126 | 0.0034 | yes |
| 4 | uniform | attention | -0.0033 | 0.0081 | -0.0167 | 0.0100 | 0.0201 | yes |
| 4 | uniform | frame_diff | 0.0022 | 0.0077 | -0.0105 | 0.0149 | 0.0108 | yes |
| 4 | random | optical_flow | -0.0074 | 0.0064 | -0.0180 | 0.0032 | 0.0257 | yes |
| 4 | random | attention | -0.0119 | 0.0071 | -0.0235 | -0.0003 | 0.1250 | no |
| 4 | random | frame_diff | -0.0063 | 0.0065 | -0.0171 | 0.0045 | 0.0184 | yes |
| 4 | optical_flow | attention | -0.0045 | 0.0083 | -0.0182 | 0.0093 | 0.0313 | yes |
| 4 | optical_flow | frame_diff | 0.0011 | 0.0080 | -0.0120 | 0.0142 | 0.0089 | yes |
| 4 | attention | frame_diff | 0.0056 | 0.0084 | -0.0083 | 0.0194 | 0.0432 | yes |
| 8 | uniform | random | -0.0011 | 0.0055 | -0.0101 | 0.0079 | 0.0003 | yes |
| 8 | uniform | optical_flow | -0.0011 | 0.0075 | -0.0134 | 0.0112 | 0.0058 | yes |
| 8 | uniform | attention | -0.0067 | 0.0072 | -0.0186 | 0.0052 | 0.0327 | yes |
| 8 | uniform | frame_diff | 0.0011 | 0.0062 | -0.0091 | 0.0113 | 0.0012 | yes |
| 8 | random | optical_flow | -0.0000 | 0.0063 | -0.0103 | 0.0103 | 0.0008 | yes |
| 8 | random | attention | -0.0056 | 0.0070 | -0.0171 | 0.0060 | 0.0197 | yes |
| 8 | random | frame_diff | 0.0022 | 0.0057 | -0.0071 | 0.0115 | 0.0009 | yes |
| 8 | optical_flow | attention | -0.0056 | 0.0084 | -0.0194 | 0.0083 | 0.0432 | yes |
| 8 | optical_flow | frame_diff | 0.0022 | 0.0070 | -0.0094 | 0.0138 | 0.0059 | yes |
| 8 | attention | frame_diff | 0.0078 | 0.0066 | -0.0031 | 0.0186 | 0.0321 | yes |
| 16 | uniform | random | -0.0056 | 0.0041 | -0.0123 | 0.0012 | 0.0002 | yes |
| 16 | uniform | optical_flow | 0.0022 | 0.0063 | -0.0082 | 0.0126 | 0.0025 | yes |
| 16 | uniform | attention | -0.0078 | 0.0058 | -0.0173 | 0.0017 | 0.0176 | yes |
| 16 | uniform | frame_diff | -0.0022 | 0.0067 | -0.0132 | 0.0088 | 0.0040 | yes |
| 16 | random | optical_flow | 0.0078 | 0.0054 | -0.0011 | 0.0167 | 0.0122 | yes |
| 16 | random | attention | -0.0022 | 0.0054 | -0.0111 | 0.0066 | 0.0005 | yes |
| 16 | random | frame_diff | 0.0033 | 0.0055 | -0.0057 | 0.0124 | 0.0013 | yes |
| 16 | optical_flow | attention | -0.0100 | 0.0066 | -0.0209 | 0.0008 | 0.0650 | no |
| 16 | optical_flow | frame_diff | -0.0045 | 0.0070 | -0.0161 | 0.0071 | 0.0138 | yes |
| 16 | attention | frame_diff | 0.0056 | 0.0058 | -0.0040 | 0.0151 | 0.0064 | yes |
| 32 | uniform | random | 0.0048 | 0.0022 | 0.0012 | 0.0084 | 0.0000 | yes |
| 32 | uniform | optical_flow | 0.0022 | 0.0022 | -0.0014 | 0.0059 | 0.0000 | yes |
| 32 | uniform | attention | 0.0011 | 0.0025 | -0.0030 | 0.0052 | 0.0000 | yes |
| 32 | uniform | frame_diff | 0.0045 | 0.0022 | 0.0008 | 0.0081 | 0.0000 | yes |
| 32 | random | optical_flow | -0.0026 | 0.0027 | -0.0070 | 0.0019 | 0.0000 | yes |
| 32 | random | attention | -0.0037 | 0.0023 | -0.0075 | 0.0001 | 0.0000 | yes |
| 32 | random | frame_diff | -0.0004 | 0.0018 | -0.0033 | 0.0026 | 0.0000 | yes |
| 32 | optical_flow | attention | -0.0011 | 0.0033 | -0.0066 | 0.0044 | 0.0000 | yes |
| 32 | optical_flow | frame_diff | 0.0022 | 0.0032 | -0.0030 | 0.0074 | 0.0000 | yes |
| 32 | attention | frame_diff | 0.0033 | 0.0019 | 0.0002 | 0.0065 | 0.0000 | yes |

```
TOST equivalence at delta=0.02, 90% CI, n_queries=898 (docs/decisions.md 2026-06-12):
  pairs equivalent within +/-0.02: 38 / 40
  90% CI half-width: median 0.0104, max 0.0138
  => UNDERPOWERED for +/-0.02: achievable bound ~+/-0.014 (needs more queries)
```
