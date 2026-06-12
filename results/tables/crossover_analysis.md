> Mechanism analysis for the §5.6 K=32 coverage crossover (velocity placement of anchors). See docs/decisions.md (2026-06-12).

| method | K | mean_cov | mean_cov_highvel | mean_cov_lowvel | velocity_ratio | max_gap_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 4 | 0.0459 | 0.0538 | 0.0370 | 1.0236 | 1.0365 |
| random | 4 | 0.0496 | 0.0569 | 0.0412 | 1.0270 | 1.7398 |
| optical_flow | 4 | 0.0485 | 0.0557 | 0.0400 | 0.9307 | 1.4708 |
| attention | 4 | 0.0485 | 0.0547 | 0.0417 | 1.1170 | 1.4612 |
| frame_diff | 4 | 0.0486 | 0.0550 | 0.0416 | 1.1130 | 1.4285 |
| uniform | 8 | 0.0342 | 0.0441 | 0.0252 | 1.0653 | 1.1789 |
| random | 8 | 0.0389 | 0.0462 | 0.0306 | 0.9970 | 2.3909 |
| optical_flow | 8 | 0.0420 | 0.0479 | 0.0343 | 0.9157 | 2.2079 |
| attention | 8 | 0.0376 | 0.0447 | 0.0317 | 1.1137 | 2.2098 |
| frame_diff | 8 | 0.0372 | 0.0448 | 0.0310 | 1.1077 | 2.1701 |
| uniform | 16 | 0.0207 | 0.0409 | 0.0208 | 1.0106 | 1.2376 |
| random | 16 | 0.0228 | 0.0408 | 0.0256 | 1.0000 | 1.9860 |
| optical_flow | 16 | 0.0285 | 0.0471 | 0.0350 | 0.9670 | 2.5851 |
| attention | 16 | 0.0215 | 0.0403 | 0.0279 | 1.0631 | 2.5448 |
| frame_diff | 16 | 0.0215 | 0.0410 | 0.0273 | 1.0610 | 2.5078 |
| uniform | 32 | 0.0060 | 0.0312 | 0.0157 | 0.9976 | 1.1760 |
| random | 32 | 0.0069 | 0.0356 | 0.0199 | 0.9997 | 1.2602 |
| optical_flow | 32 | 0.0091 | 0.0422 | 0.0299 | 0.9962 | 1.6510 |
| attention | 32 | 0.0048 | 0.0280 | 0.0182 | 1.0087 | 1.6269 |
| frame_diff | 32 | 0.0043 | 0.0266 | 0.0162 | 1.0097 | 1.5799 |

```
Pre-registered mechanism test at K=32 (docs/decisions.md 2026-06-12):
  uniform  velocity_ratio = 0.998
  attention   velocity_ratio = 1.009 (margin +0.011; >1 & >=0.10 -> False)
  attention   hi-vel adv = +0.0032  lo-vel adv = -0.0025 (hi>lo -> True)
  frame_diff  velocity_ratio = 1.010 (margin +0.012; >1 & >=0.10 -> False)
  frame_diff  hi-vel adv = +0.0045  lo-vel adv = -0.0005 (hi>lo -> True)
  VERDICT: INCONCLUSIVE
```
