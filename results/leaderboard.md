Bradley-Terry ratings on the Elo scale, anchored at `random` = 0, with 95% bootstrap intervals from resampling deals.

| Rank | Agent | Elo | 95% CI | Score | Avg points | ms / move |
| ---: | --- | ---: | :---: | ---: | ---: | ---: |
| 1 | `ismcts` | +428 | [+395, +471] | 0.681 | 63.5 | 254.84 |
| 2 | `alphabeta` | +400 | [+367, +439] | 0.640 | 68.1 | 81.56 |
| 3 | `heuristic` | +357 | [+320, +395] | 0.574 | 63.9 | 0.01 |
| 4 | `ppo-v1` | +314 | [+281, +352] | 0.508 | 62.6 | 6.74 |
| 5 | `greedy` | +297 | [+263, +336] | 0.483 | 61.7 | 0.01 |
| 6 | `random` | +0 | [+0, +0] | 0.114 | 40.2 | 0.00 |

Head-to-head score of the row agent against the column agent (± half-width of the 95% Wilson interval):

| | `ismcts` | `alphabeta` | `heuristic` | `ppo-v1` | `greedy` | `random` |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ismcts` | - | 0.56 ± 0.07 | 0.56 ± 0.07 | 0.70 ± 0.06 | 0.66 ± 0.07 | 0.93 ± 0.04 |
| `alphabeta` | 0.44 ± 0.07 | - | 0.56 ± 0.07 | 0.64 ± 0.07 | 0.63 ± 0.07 | 0.92 ± 0.04 |
| `heuristic` | 0.44 ± 0.07 | 0.44 ± 0.07 | - | 0.54 ± 0.07 | 0.58 ± 0.07 | 0.88 ± 0.05 |
| `ppo-v1` | 0.30 ± 0.06 | 0.36 ± 0.07 | 0.46 ± 0.07 | - | 0.57 ± 0.07 | 0.85 ± 0.05 |
| `greedy` | 0.34 ± 0.07 | 0.37 ± 0.07 | 0.42 ± 0.07 | 0.43 ± 0.07 | - | 0.85 ± 0.05 |
| `random` | 0.07 ± 0.04 | 0.08 ± 0.04 | 0.12 ± 0.05 | 0.15 ± 0.05 | 0.15 ± 0.05 | - |
