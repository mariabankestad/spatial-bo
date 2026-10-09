# Weight fields

One file per field and dimension, D = 2..6, loaded by
`spatialbo.load_field(kind, D)`.

| folder | field | principle | file | networks |
|---|---|---|---|---|
| `geo/` | g_geo | geometric equalization | `g_geo_D{D}.pt` | geometric mean of two networks trained from independent seeds |
| `eff/` | g_eff | efficiency-oriented exploration | `g_eff_D{D}.pt` | geometric mean of two networks trained from independent seeds |
| `uniform/` | g_uniform | uniform allocation | `g_uniform_D{D}.pt` | one network |

Trained range: N in [2D, 150] observations and lengthscale in [0.05, 3]
([0.02, 3] for g_geo and g_uniform in D = 2). Inputs outside the range are
clamped to it. g(center) = 1 holds exactly, and the fields are invariant to
reflections and permutations of the coordinates by construction.

Each file holds the networks as `members` (state dict, normalization
statistics and training metadata) and the trained range as `envelope`.
`training/pack_field.py` writes such a file from the artifacts of the
trainers in `training/`. The JSON files in `uniform/` are the fit and
certification reports of g_uniform (`training/gunif_fit.py`,
`training/gunif_certify.py`).
