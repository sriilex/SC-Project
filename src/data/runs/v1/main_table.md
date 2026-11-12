# Main table — test split, common support

All models are scored on the intersection of rows where every model emitted a prediction (the LSTM needs a warm-up window). Lower pinball is better.

| h | model | n | mean pinball | cover-80 | Winkler-80 |
|---|-------|---|--------------|----------|------------|
| 1 | lstm | 765 | 0.00715 | 0.8366 | 0.1164 |
| 1 | lstm_fuzzy | 765 | 0.00751 | 0.7451 | 0.1217 |
| 1 | cqr | 765 | 0.00716 | 0.8706 | 0.1173 |
| 1 | aci | 765 | 0.00707 | 0.8157 | 0.1141 |
| 7 | lstm | 895 | 0.02200 | 0.9084 | 0.3293 |
| 7 | lstm_fuzzy | 895 | 0.02451 | 0.8201 | 0.3781 |
| 7 | cqr | 895 | 0.02179 | 0.8860 | 0.3229 |
| 7 | aci | 895 | 0.02136 | 0.8078 | 0.3100 |
| 30 | lstm | 505 | 0.04833 | 0.9386 | 0.7487 |
| 30 | lstm_fuzzy | 505 | 0.04645 | 0.7960 | 0.6828 |
| 30 | cqr | 505 | 0.04834 | 0.8990 | 0.7129 |
| 30 | aci | 505 | 0.04451 | 0.8099 | 0.6343 |
| 90 | lstm | 571 | 0.06914 | 0.9912 | 1.1143 |
| 90 | lstm_fuzzy | 571 | 0.06080 | 0.7303 | 0.8089 |
| 90 | cqr | 571 | 0.05953 | 0.9054 | 0.8801 |
| 90 | aci | 571 | 0.05894 | 0.6935 | 0.8086 |

## h=90 per class

| class | model | n | mean pinball | cover-80 | Winkler-80 |
|-------|-------|---|--------------|----------|------------|
| Capesize | lstm | 157 | 0.11791 | 0.9873 | 1.8362 |
| Capesize | lstm_fuzzy | 157 | 0.10200 | 0.7261 | 1.3424 |
| Capesize | cqr | 157 | 0.10607 | 0.9618 | 1.5959 |
| Capesize | aci | 157 | 0.10158 | 0.7134 | 1.3465 |
| Panamax | lstm | 158 | 0.05586 | 0.9810 | 0.9371 |
| Panamax | lstm_fuzzy | 158 | 0.04112 | 0.8228 | 0.5943 |
| Panamax | cqr | 158 | 0.04711 | 0.8608 | 0.7279 |
| Panamax | aci | 158 | 0.04843 | 0.7405 | 0.7142 |
| Supramax | lstm | 158 | 0.05249 | 1.0000 | 0.8018 |
| Supramax | lstm_fuzzy | 158 | 0.04490 | 0.7658 | 0.5912 |
| Supramax | cqr | 158 | 0.04471 | 0.8797 | 0.5660 |
| Supramax | aci | 158 | 0.04530 | 0.6709 | 0.5860 |
| Handysize | lstm | 98 | 0.03924 | 1.0000 | 0.7475 |
| Handysize | lstm_fuzzy | 98 | 0.05220 | 0.5306 | 0.6513 |
| Handysize | cqr | 98 | 0.02888 | 0.9286 | 0.4852 |
| Handysize | aci | 98 | 0.02959 | 0.6224 | 0.4582 |
