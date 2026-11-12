# Fuzzy-on-CQR stack — h=90 per class (test)

The fuzzy stage does **not** inherit CQR's coverage guarantee.

| class | model | n | mean pinball | cover-80 | Winkler-80 |
|-------|-------|---|--------------|----------|------------|
| Capesize | cqr | 157 | 0.10607 | 0.9618 | 1.5959 |
| Capesize | cqr | 242 | 0.11246 | 0.9008 | 1.8853 |
| Capesize | fuz_cqr | 157 | 0.10066 | 0.7707 | 1.3708 |
| Capesize | fuz_cqr | 242 | 0.08798 | 0.8099 | 1.3518 |
| Handysize | cqr | 98 | 0.02888 | 0.9286 | 0.4852 |
| Handysize | cqr | 244 | 0.05167 | 0.7582 | 0.8195 |
| Handysize | fuz_cqr | 98 | 0.02735 | 0.8980 | 0.3777 |
| Handysize | fuz_cqr | 244 | 0.03969 | 0.7008 | 0.4957 |
| Panamax | cqr | 158 | 0.04711 | 0.8608 | 0.7279 |
| Panamax | cqr | 242 | 0.05534 | 0.8099 | 0.8204 |
| Panamax | fuz_cqr | 158 | 0.04404 | 0.8354 | 0.6917 |
| Panamax | fuz_cqr | 242 | 0.04554 | 0.7521 | 0.6493 |
| Supramax | cqr | 158 | 0.04471 | 0.8797 | 0.5660 |
| Supramax | cqr | 242 | 0.05610 | 0.7314 | 0.8356 |
| Supramax | fuz_cqr | 158 | 0.04773 | 0.7089 | 0.6335 |
| Supramax | fuz_cqr | 242 | 0.04318 | 0.7314 | 0.6188 |
