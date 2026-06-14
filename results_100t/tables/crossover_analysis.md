> Mechanism analysis for the §5.6 K=32 coverage crossover (velocity placement of anchors). See docs/decisions.md (2026-06-12).

| method | K | mean_cov | mean_cov_highvel | mean_cov_lowvel | velocity_ratio | max_gap_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 4 | 0.0416 | 0.0485 | 0.0341 | 1.0193 | 1.0454 |
| random | 4 | 0.0445 | 0.0506 | 0.0379 | 1.0177 | 1.7255 |
| optical_flow | 4 | 0.0430 | 0.0495 | 0.0356 | 0.9213 | 1.4868 |
| attention | 4 | 0.0444 | 0.0491 | 0.0395 | 1.1116 | 1.4985 |
| frame_diff | 4 | 0.0445 | 0.0491 | 0.0398 | 1.1104 | 1.4932 |
| uniform | 8 | 0.0308 | 0.0387 | 0.0233 | 1.0281 | 1.1319 |
| random | 8 | 0.0344 | 0.0408 | 0.0275 | 1.0004 | 2.3664 |
| optical_flow | 8 | 0.0368 | 0.0423 | 0.0299 | 0.9217 | 2.1646 |
| attention | 8 | 0.0337 | 0.0395 | 0.0288 | 1.1109 | 2.1482 |
| frame_diff | 8 | 0.0335 | 0.0391 | 0.0287 | 1.1106 | 2.1417 |
| uniform | 16 | 0.0221 | 0.0354 | 0.0177 | 1.0104 | 1.3085 |
| random | 16 | 0.0236 | 0.0340 | 0.0209 | 1.0002 | 2.2283 |
| optical_flow | 16 | 0.0308 | 0.0404 | 0.0299 | 0.9564 | 3.0388 |
| attention | 16 | 0.0230 | 0.0337 | 0.0238 | 1.0741 | 2.9747 |
| frame_diff | 16 | 0.0225 | 0.0332 | 0.0233 | 1.0767 | 2.9348 |
| uniform | 32 | 0.0058 | 0.0322 | 0.0171 | 0.9981 | 1.1627 |
| random | 32 | 0.0062 | 0.0352 | 0.0197 | 0.9997 | 1.2464 |
| optical_flow | 32 | 0.0087 | 0.0425 | 0.0308 | 0.9960 | 1.6104 |
| attention | 32 | 0.0045 | 0.0277 | 0.0183 | 1.0090 | 1.5837 |
| frame_diff | 32 | 0.0042 | 0.0265 | 0.0173 | 1.0095 | 1.5559 |

```
Pre-registered mechanism test at K=32 (docs/decisions.md 2026-06-12):
  uniform  velocity_ratio = 0.998
  attention   velocity_ratio = 1.009 (margin +0.011; >1 & >=0.10 -> False)
  attention   hi-vel adv = +0.0045  lo-vel adv = -0.0012 (hi>lo -> True)
  frame_diff  velocity_ratio = 1.009 (margin +0.011; >1 & >=0.10 -> False)
  frame_diff  hi-vel adv = +0.0057  lo-vel adv = -0.0002 (hi>lo -> True)
  VERDICT: INCONCLUSIVE
```
