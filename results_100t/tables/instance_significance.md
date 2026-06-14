> Paired sign-flip permutation tests on per-query Top-1 instance correctness (diff = method - uniform). 863 paired queries.

| method | vs | K | diff_top1 | p_value | sig_0.05 |
| --- | --- | --- | --- | --- | --- |
| random | uniform | 4 | -0.0652 | 0.0000 | True |
| optical_flow | uniform | 4 | -0.0533 | 0.0000 | True |
| attention | uniform | 4 | -0.0367 | 0.0000 | True |
| frame_diff | uniform | 4 | -0.0239 | 0.0010 | True |
| random | uniform | 8 | -0.0553 | 0.0000 | True |
| optical_flow | uniform | 8 | -0.0975 | 0.0000 | True |
| attention | uniform | 8 | -0.0554 | 0.0000 | True |
| frame_diff | uniform | 8 | -0.0355 | 0.0000 | True |
| random | uniform | 16 | -0.0194 | 0.0000 | True |
| optical_flow | uniform | 16 | -0.0483 | 0.0000 | True |
| attention | uniform | 16 | -0.0018 | 0.7879 | False |
| frame_diff | uniform | 16 | -0.0005 | 0.9666 | False |
| random | uniform | 32 | -0.0021 | 0.2188 | False |
| optical_flow | uniform | 32 | -0.0050 | 0.0469 | True |
| attention | uniform | 32 | -0.0002 | 1.0000 | False |
| frame_diff | uniform | 32 | 0.0005 | 0.9090 | False |
