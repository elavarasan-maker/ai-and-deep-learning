from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "Sen2Fire"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"
IMAGE_SIZE = 224
BATCH_SIZE = 8
SEED = 42
# Official scene split: preserve it to reduce spatial leakage from overlapping chips.
SPLITS = {"train": ["scene1", "scene2"], "val": ["scene3"], "test": ["scene4"]}
# Patch is positive when at least this fraction of pixels have fire label 1.
PATCH_FIRE_FRACTION_THRESHOLD = 0.0001
# 0-based indices in the 12-channel B1..B12 image cube: B4, B3, B2, B8, B12.
SELECTED_BAND_INDICES = [3, 2, 1, 7, 11]
BAND_NAMES = ["B4_Red", "B3_Green", "B2_Blue", "B8_NIR", "B12_SWIR2"]
