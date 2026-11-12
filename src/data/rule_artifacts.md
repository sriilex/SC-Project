# Learned Sugeno rule artifacts

Generated from the saved parameter JSON. Each row lists one input's
Gaussian membership functions as (centre, width) pairs, then the constant
rule consequents in row-major order.

| h | system | inputs | membership functions | consequents |
|---|---|---|---|---|
| 1 | width | cov_gap, vol_z | (-1.983, 0.441) (+0.551, 0.147) (+1.696, 0.104) (-1.260, 0.249) (+0.269, 0.084) (+2.583, 0.556) | -0.214, +0.109, -1.386, +0.082, +0.265, -1.967, -0.565, -0.021, +0.940 |
| 1 | loc | bias_z, vol_z | (-1.338, 0.357) (+0.869, 0.195) (+1.266, 0.234) (-0.666, 0.284) (-0.272, 0.131) (+0.889, 0.176) | +0.330, -0.181, -0.128, +0.005, +0.029, -0.156, -0.296, +0.072, -5.455 |
| 7 | width | cov_gap, vol_z | (-1.086, 0.202) (+0.845, 0.184) (+2.402, 0.624) (-0.546, 0.063) (+5.673, 0.912) (+0.336, 0.073) | -0.128, -2.511, +0.729, -3.540, -0.025, -0.364, +0.352, -2.481, -0.304 |
| 7 | loc | bias_z, vol_z | (-1.136, 0.986) (-0.830, 0.964) (+1.267, 0.892) (-1.244, 0.426) (-0.180, 0.114) (+2.066, 0.374) | +1.203, -0.500, +0.642, +0.404, +0.500, -1.178, -1.293, -0.043, +0.117 |
| 30 | width | vol_z | (-0.860, 0.115) (+4.205, 0.667) (-0.023, 0.030) | -0.849, -2.194, -0.289 |
| 30 | loc | bias_z, vol_z | (-1.024, 0.168) (+0.291, 0.494) (+0.301, 0.172) (-0.331, 0.143) (+1.353, 0.590) (+3.313, 0.994) | -0.253, +0.938, +2.159, +0.282, +0.726, -0.340, -0.254, -1.248, -0.983 |
| 90 | width | vol_z | (+1.037, 1.425) (+0.993, 0.146) (+0.991, 0.185) | -3.236, -1.063, -1.178 |
| 90 | loc | bias_z, vol_z | (-1.151, 0.191) (-0.352, 0.081) (+0.958, 0.166) (-0.263, 0.222) (+0.085, 2.784) (+3.643, 2.291) | -0.348, +0.663, -0.570, +0.353, +1.112, -1.446, -0.355, +0.743, -1.700 |

## Honest reading

Overlapping Gaussian memberships mean individual rules should not be read as
isolated if-then statements. At long horizons the width memberships can be
nearly identical; when they are, the layer is effectively a constant width
adjustment and the table says so.
