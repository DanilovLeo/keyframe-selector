## Pooling sensitivity — per (pooling, method, K)

| pooling | method | K | top_1 | top_5 |
| --- | --- | --- | --- | --- |
| mean | uniform | 4 | 0.809 | 0.961 |
| mean | random | 4 | 0.826 | 0.957 |
| mean | optical_flow | 4 | 0.803 | 0.944 |
| mean | attention | 4 | 0.820 | 0.972 |
| mean | frame_diff | 4 | 0.837 | 0.966 |
| mean | uniform | 8 | 0.815 | 0.949 |
| mean | random | 8 | 0.839 | 0.955 |
| mean | optical_flow | 8 | 0.826 | 0.961 |
| mean | attention | 8 | 0.809 | 0.955 |
| mean | frame_diff | 8 | 0.831 | 0.961 |
| mean | uniform | 16 | 0.826 | 0.955 |
| mean | random | 16 | 0.818 | 0.955 |
| mean | optical_flow | 16 | 0.815 | 0.949 |
| mean | attention | 16 | 0.831 | 0.955 |
| mean | frame_diff | 16 | 0.831 | 0.955 |
| mean | uniform | 32 | 0.837 | 0.961 |
| mean | random | 32 | 0.835 | 0.955 |
| mean | optical_flow | 32 | 0.826 | 0.961 |
| mean | attention | 32 | 0.837 | 0.961 |
| mean | frame_diff | 32 | 0.831 | 0.955 |
| max | uniform | 4 | 0.848 | 0.966 |
| max | random | 4 | 0.828 | 0.963 |
| max | optical_flow | 4 | 0.837 | 0.949 |
| max | attention | 4 | 0.820 | 0.955 |
| max | frame_diff | 4 | 0.843 | 0.955 |
| max | uniform | 8 | 0.826 | 0.949 |
| max | random | 8 | 0.835 | 0.938 |
| max | optical_flow | 8 | 0.837 | 0.961 |
| max | attention | 8 | 0.837 | 0.955 |
| max | frame_diff | 8 | 0.848 | 0.944 |
| max | uniform | 16 | 0.848 | 0.955 |
| max | random | 16 | 0.841 | 0.953 |
| max | optical_flow | 16 | 0.837 | 0.955 |
| max | attention | 16 | 0.837 | 0.961 |
| max | frame_diff | 16 | 0.837 | 0.938 |
| max | uniform | 32 | 0.848 | 0.961 |
| max | random | 32 | 0.839 | 0.955 |
| max | optical_flow | 32 | 0.854 | 0.955 |
| max | attention | 32 | 0.843 | 0.949 |
| max | frame_diff | 32 | 0.837 | 0.955 |
| best_match | uniform | 4 | 0.803 | 0.938 |
| best_match | random | 4 | 0.820 | 0.957 |
| best_match | optical_flow | 4 | 0.781 | 0.944 |
| best_match | attention | 4 | 0.787 | 0.966 |
| best_match | frame_diff | 4 | 0.820 | 0.966 |
| best_match | uniform | 8 | 0.837 | 0.961 |
| best_match | random | 8 | 0.831 | 0.959 |
| best_match | optical_flow | 8 | 0.815 | 0.966 |
| best_match | attention | 8 | 0.815 | 0.949 |
| best_match | frame_diff | 8 | 0.826 | 0.961 |
| best_match | uniform | 16 | 0.831 | 0.949 |
| best_match | random | 16 | 0.824 | 0.953 |
| best_match | optical_flow | 16 | 0.815 | 0.938 |
| best_match | attention | 16 | 0.831 | 0.961 |
| best_match | frame_diff | 16 | 0.831 | 0.944 |
| best_match | uniform | 32 | 0.815 | 0.972 |
| best_match | random | 32 | 0.818 | 0.970 |
| best_match | optical_flow | 32 | 0.820 | 0.972 |
| best_match | attention | 32 | 0.820 | 0.972 |
| best_match | frame_diff | 32 | 0.820 | 0.966 |

## Between-method Top-1 spread per (pooling, K)

| pooling | K | top1_min | top1_max | top1_spread | argmax_method |
| --- | --- | --- | --- | --- | --- |
| mean | 4 | 0.803 | 0.837 | 0.034 | frame_diff |
| mean | 8 | 0.809 | 0.839 | 0.030 | random |
| mean | 16 | 0.815 | 0.831 | 0.017 | attention |
| mean | 32 | 0.826 | 0.837 | 0.011 | uniform |
| max | 4 | 0.820 | 0.848 | 0.028 | uniform |
| max | 8 | 0.826 | 0.848 | 0.022 | frame_diff |
| max | 16 | 0.837 | 0.848 | 0.011 | uniform |
| max | 32 | 0.837 | 0.854 | 0.017 | optical_flow |
| best_match | 4 | 0.781 | 0.820 | 0.039 | random |
| best_match | 8 | 0.815 | 0.837 | 0.022 | uniform |
| best_match | 16 | 0.815 | 0.831 | 0.017 | uniform |
| best_match | 32 | 0.815 | 0.820 | 0.006 | optical_flow |
