> Paired sign-flip permutation tests on per-query Top-1 instance correctness (diff = method - uniform). 863 paired queries.

| method | vs | K | diff_top1 | p_value | sig_0.05 |
| --- | --- | --- | --- | --- | --- |
| random | uniform | 4 | -0.0603 | 0.0000 | True |
| optical_flow | uniform | 4 | -0.0440 | 0.0034 | True |
| attention | uniform | 4 | -0.0406 | 0.0102 | True |
| frame_diff | uniform | 4 | -0.0232 | 0.1726 | False |
| random | uniform | 8 | -0.0379 | 0.0017 | True |
| optical_flow | uniform | 8 | -0.0660 | 0.0000 | True |
| attention | uniform | 8 | -0.0035 | 0.8836 | False |
| frame_diff | uniform | 8 | -0.0023 | 0.9449 | False |
| random | uniform | 16 | -0.0143 | 0.0702 | False |
| optical_flow | uniform | 16 | -0.0301 | 0.0134 | True |
| attention | uniform | 16 | 0.0058 | 0.6794 | False |
| frame_diff | uniform | 16 | 0.0162 | 0.1527 | False |
| random | uniform | 32 | 0.0027 | 0.5677 | False |
| optical_flow | uniform | 32 | 0.0023 | 0.8341 | False |
| attention | uniform | 32 | 0.0012 | 1.0000 | False |
| frame_diff | uniform | 32 | 0.0046 | 0.3894 | False |
