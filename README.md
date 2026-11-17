# Wildfire Detection with Sen2Fire + EfficientNetB0

A reproducible research project for patch-level Fire / No-Fire classification from multispectral Sentinel-2 data. The starter code is prepared for the public Sen2Fire dataset.

> **Scientific honesty:** no trained weights or performance scores are included. Metrics are generated only after training and evaluation on the supplied test split. A 94% accuracy / 0.92 Fire recall target is an evaluation goal, not a guaranteed result.

## Dataset

- Dataset: Sen2Fire (Xu, Berg & Haglund, 2024)
- Zenodo record: https://zenodo.org/records/10881058
- Download: https://zenodo.org/records/10881058/files/Sen2Fire.zip?download=1
- Archive size listed by Zenodo: approximately 6.3 GB.
- 2,466 patches, 512×512, 12 Sentinel-2 bands + Sentinel-5P aerosol layer; the dataset page describes 13 channels total.
- Dataset's official split is scene-based: scene1+scene2 train, scene3 validation, scene4 test (1,458 / 504 / 504). Keep this split to avoid spatial/scene leakage; do not randomly mix overlapping neighboring patches across splits.
- Pixel labels: 0 = Non-fire, 1 = Fire. Patch labels are derived from the fire-mask fraction; configurable threshold is in `config.py`.

The NPZ schema is expected to contain `image` (12,H,W), `label` (H,W), and optionally `aerosol` (H,W). The loader validates shapes and reports errors rather than silently guessing.

## Important band convention

Sen2Fire image bands are ordered B1..B12. The default five-channel stack is B4, B3, B2, B8, B12 (RGB + NIR + SWIR2). NBR is calculated as `(B8 - B12) / (B8 + B12)` for analysis and reports. This preserves raw SWIR2 as an input channel; NBR is not substituted for SWIR2. Values are normalized using train-only per-band statistics.

EfficientNetB0 was pretrained on 3-channel ImageNet RGB. This implementation uses a trainable 1×1 spectral adapter (5→3) in front of the pretrained backbone. This is an explicit, reproducible way to retain the ImageNet backbone while learning a projection from multispectral channels; it is not claimed to be a literal 5-channel pretrained stem.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_dataset.py
python -m src.inspect_dataset --data_dir data/Sen2Fire
python -m src.train --data_dir data/Sen2Fire --epochs 25
python -m src.evaluate --data_dir data/Sen2Fire
python app.py
```

TensorFlow GPU support on native Windows is limited for modern TensorFlow releases. CPU works but training may be slow. For GPU training, use WSL2 with a supported NVIDIA setup or Google Colab/Kaggle. Do not assume a GPU is available.

## Dataset layout after extraction

```
data/Sen2Fire/scene1/*.npz
 data/Sen2Fire/scene2/*.npz
 data/Sen2Fire/scene3/*.npz
 data/Sen2Fire/scene4/*.npz
```

## Pipeline

1. Inspect NPZ keys, shapes, label values, class distribution, and sample patches.
2. Preserve official scene-based train/validation/test split.
3. Derive patch labels from pixel-mask fire fraction; invalid/no-data handling is audited.
4. Resize to 224×224; train-only flips/90° rotations and mild brightness augmentation.
5. Calculate train-only mean/std for B4/B3/B2/B8/B12.
6. Train a four-block baseline CNN.
7. Train EfficientNetB0 with spectral adapter, first frozen then last 40 backbone layers fine-tuned.
8. Evaluate accuracy, per-class precision/recall/F1, macro-F1, ROC-AUC, confusion matrix, ROC curve.
9. Generate Grad-CAM overlays and NBR panels.
10. Serve single and batch NPZ predictions through Flask; export CSV alert report.

## Important limitations

- Sen2Fire labels originate from MOD14A1 fire products and are raster masks, not human-verified ground truth for every pixel.
- A patch with a tiny fire-mask fraction may be classified positive under the default `PATCH_FIRE_FRACTION_THRESHOLD=0.0001`; report this threshold and run sensitivity analysis.
- The benchmark's scene-level split is preferred over a random 80/10/10 split because adjacent patches overlap by 128 pixels. The official split reduces leakage even though its proportions differ from 80/10/10.
- NBR is commonly used for burn severity / vegetation change and is not a definitive active-fire detector. SWIR brightness and thermal fire products are more direct active-fire cues. Describe this carefully in the report.
- This is a research/educational prototype, not an operational emergency warning service. Satellite revisit, cloud, geolocation and sensor limitations prevent guaranteed early warning.

## Report and screenshots

Run training/evaluation first. Plots are saved under `artifacts/`. Then generate the report:

```powershell
python scripts/build_report.py
```

The report builder embeds all available generated figures. Missing figures are called out as not yet generated; do not submit the template as completed experimental evidence. The dataset link is intentionally the final section of the report.

## Citation

Xu, Y., Berg, A., & Haglund, L. (2024). *Sen2Fire: A Challenging Benchmark Dataset for Wildfire Detection using Sentinel Data*. IGARSS 2024. https://doi.org/10.1109/IGARSS53475.2024.10641441

Dataset DOI: https://doi.org/10.5281/zenodo.10881058
