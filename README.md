# MS-PDViT: Multi-Scale Patch-wise-Decoder Vision Transformer for radar-based lightning nowcasting

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22693613.svg)](https://doi.org/10.5281/zenodo.22693613)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reference implementation for the paper

> *A multi-scale patch-wise-decoder vision transformer for radar-only lightning
> nowcasting* (submitted to *Journal of Geophysical Research: Atmospheres*).

MS-PDViT predicts dense, pixel-level lightning probability fields for six
6-minute lead times (t+6 … t+36 min) from radar alone. It reads a 36-minute
history of three radar products at two patch scales, decodes **each spatial
patch's own** future lightning probability rather than reconstructing the whole
field from a single global token, and fuses the two scales with fixed uniform
weights.

## Data availability

**This repository contains code only. It contains no observational data.**

The radar imagery and lightning location observations used in the paper were
provided by the Shenzhen Meteorological Bureau and cannot be redistributed by
the authors under the Bureau's data-management regulations. Requests for access,
where permitted, must be made directly to the Bureau
(`webmaster@mail.weather.sz.gov.cn`) and are subject to its approval and to any
required data-use agreement. See the paper's Open Research section for the
authoritative statement.

The code therefore cannot be run end-to-end without separately obtaining
equivalent data. What it does provide is the complete model definition, the
training and evaluation procedure, and every architectural and optimisation
setting used to produce the reported results, so that the method can be
reimplemented and applied to other radar archives.

## What is here

```
c4dllightning/
├── features/                 data pipeline
│   ├── batch.py                windowing, chronological splits, dataset/loader
│   ├── image_loader.py         radar product reading, per-channel normalization
│   ├── lightning_data.py       flash records -> binary target fields
│   ├── static_data_loader.py   optional static predictors (unused in the paper)
│   ├── transform.py, utils.py, regions.py, optimized_batch.py
├── ml/models/
│   ├── pure_transformer.py     ** SpatioTemporalTransformer, MultiPathTransformer,
│   │                              the patch-wise decoder and the fusion modes **
│   ├── models.py               model construction, loss, optimiser, training loop
│   ├── enhanced_transformer.py hybrid CNN-transformer variant
│   ├── blocks.py, layers.py, optimizers.py, rnn.py
├── analysis/
│   ├── evaluation.py           CSI / POD / FAR / HSS / ETS, PR-AUC, Brier
│   ├── calibration.py, ensemble.py, lagrangian.py
└── visualization/plots.py

scripts/
└── test_pytorch_own_data2.py   single entry point: training and evaluation,
                                with every published configuration in MODEL_CONFIGS
```

`pure_transformer.py` is where the paper's contribution lives. The patch-wise
decoder is in `SpatioTemporalTransformer.forward`; the CLS-only ablation head is
selected by `legacy_prediction_head=True`; the three fusion strategies
(`uniform`, `learnable_fixed`, `dynamic`) are in `MultiPathTransformer`.

The convolutional-recurrent (CNN+GRU) baseline lives on a separate branch and is
not part of this release.

## Configurations

Every model in the paper is a named entry in `MODEL_CONFIGS` in
`scripts/test_pytorch_own_data2.py`, selected with the `MODEL_CONFIG`
environment variable:

| Paper model | Training config | Test-evaluation config |
|---|---|---|
| MS-PDViT (uniform fusion) | `pure_transformer` | `pure_transformer_testeval` |
| Learnable fixed fusion | `pure_transformer_learnable_fusion` | `pure_transformer_learnable_fusion_testeval` |
| Dynamic gating | `pure_transformer_dynamic_gating` | `pure_transformer_dynamic_gating_testeval` |
| Single-patch ViT (CLS-only) | `pure_transformer_singlepatch_cls_train` | `pure_transformer_singlepatch_testeval` |
| MultiPath ViT (CLS-only) | `pure_transformer_multipath_cls_train` | `pure_transformer_multipath_cls_testeval` |

Two architecture flags must match the checkpoint being loaded, because they
change which parameters exist:

- `legacy_prediction_head` — `True` selects the CLS-only decoder, `False` the
  patch-wise decoder.
- `cls_attends_patches` — `True` adds a single-query cross-attention that lets
  the CLS token read the patch tokens. It is `True` for the two CLS-only models
  and `False` for the patch-wise models; the asymmetry is deliberate and is
  explained in the paper's methodology.

## Running

```bash
pip install -r requirements.txt

# single GPU
MODEL_CONFIG=pure_transformer python scripts/test_pytorch_own_data2.py

# multi-GPU (DistributedDataParallel)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MODEL_CONFIG=pure_transformer \
  python -m torch.distributed.launch --nproc_per_node=8 --master_port=29537 \
  scripts/test_pytorch_own_data2.py
```

Checkpoints are read from and written to `<repo>/models`; override with
`MSPDVIT_MODELS_DIR`. The data root is expected at `~/Weather` (see
`build_batch_gen` / `RadarBatchGenerator` for the products and directory layout
the loader assumes).

Trained weights are archived separately; see the paper's Open Research section.

## A note on channel naming

The three radar predictors are base reflectivity (DBZ, dBZ), **echo-top height
(ETH, km)** and vertically integrated liquid water (VIL, kg m⁻²), in that channel
order. For historical reasons the code identifies the echo-top-height channel as
`dbzh` (and `frac_DBZH` in diagnostic output). It is echo-top height throughout,
not a polarimetric reflectivity field.

Normalization is a fixed global linear map per channel
(`image_loader._normalize_product`): DBZ is clipped to 0–75 dBZ and divided by
75 with values below 5 dBZ set to zero, ETH is clipped to 0–20 km and divided by
20, and VIL is clipped to 0–70 kg m⁻² and divided by 70.

## Citation

If you use this code, please cite the paper together with the archived release
(see `CITATION.cff`). Version 1.0.1 is archived at
[10.5281/zenodo.22693613](https://doi.org/10.5281/zenodo.22693613); that DOI is
fixed to this version. The concept DOI
[10.5281/zenodo.22693612](https://doi.org/10.5281/zenodo.22693612) always
resolves to the most recent version instead.

## License

MIT — see `LICENSE`.
